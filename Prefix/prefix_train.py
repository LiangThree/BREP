import os
import sys
from datasets import load_dataset, Dataset
from typing import Optional, Union

import pdb
import numpy as np
import json
import torch.optim as optim

cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-1])
sys.path.append(main_dir)

from template import *
import fire
from datasets import load_dataset, concatenate_datasets
from transformers import  AutoTokenizer, TrainingArguments, AutoConfig
from trl import SFTTrainer
from transformers.trainer_callback import TrainerCallback
import os
import torch
from transformers import AutoModelForCausalLM
from model import ActivationLLama
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from transformers.trainer_callback import TrainerCallback
import torch

MAX_INPUT_LENGTH = 512
MAX_LENGTH = 512

device_map = "auto"

def load_RED_model(model_path, op_position, layer_type):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_auth_token=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = ActivationLLama(model, op_position=op_position, layer_type=layer_type, prefix=-1)
    
    return model


class CustomModelSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_path = os.path.join(checkpoint_path, "delta_vector.pth")

        kwargs["model"].save_model(save_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path) :
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))
        if  "model.safetensors" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "model.safetensors"))


class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        
        weights = [feature.get('weight', 1.0) for feature in features]
        batch['weight'] = torch.tensor(weights, dtype=torch.float32, device=batch['input_ids'].device)
        batch['labels'] = torch.stack([torch.tensor(f["labels"]) for f in features])
        batch['input_ids'] = torch.stack([torch.tensor(f["input_ids"]) for f in features])
        batch['attention_mask'] = torch.stack([torch.tensor(f["attention_mask"]) for f in features])

        return batch


class ActivationScalingMonitor(TrainerCallback):
    
    def __init__(self, model, weight, target_bias, pid_params, weight_bounds, limit_prop):
        self.model = model
        self.target_bias = target_bias * limit_prop
        self.kp, self.ki, self.kd = pid_params
        self.min_weight, self.max_weight = weight_bounds
        
        # PID控制器状态
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_scaling = None
        self.prev_bias = None
        self.last_adjusted_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        # 获取当前的 activation_scaling 值
        current_scaling = 0
        current_bias = []

        for layer in range(len(self.model.base_model.model.layers)):
            if self.model.op_position == "attn_q":
                module = self.model.base_model.model.layers[layer].self_attn.q_proj
            elif self.model.op_position == "attn_k":
                module = self.model.base_model.model.layers[layer].self_attn.k_proj
            elif self.model.op_position == "attn_v":
                module = self.model.base_model.model.layers[layer].self_attn.v_proj
            elif self.model.op_position == "attn_o":
                module = self.model.base_model.model.layers[layer].self_attn.o_proj
            elif self.model.op_position == "ffn_up":
                module = self.model.base_model.model.layers[layer].mlp.up_proj
            elif self.model.op_position == "ffn_down":
                module = self.model.base_model.model.layers[layer].mlp.down_proj
            elif self.model.op_position == "layer":
                module = self.model.base_model.model.layers[layer]

            if hasattr(module.delta_vector, 'activation_scaling'):
                delta = module.delta_vector.activation_scaling.detach().to(torch.float).cpu().numpy()
                current_scaling += delta

            if hasattr(module.delta_vector, 'activation_bias'):
                bias = module.delta_vector.activation_bias.detach().to(torch.float).cpu().numpy()
                l2_norm = np.linalg.norm(bias)
                current_bias.append(round(l2_norm,2))
        
        mean_bias = np.mean(current_bias)
        error = self.target_bias - mean_bias

        # 防止连续调整（每10步调整一次）
        if state.global_step - self.last_adjusted_step > 3:
            self.adjust_weight(error, state.global_step)
            self.last_adjusted_step = state.global_step

        # if hasattr(module.delta_vector, 'activation_scaling'):
        #     print(f"Step {state.global_step}: scaling: {current_scaling}")

        with open(self.model.log_path, "a") as f:
            print(f"Step {state.global_step}: mean_bias:{mean_bias} activation_bias: {current_bias} error={error:.4f} weight={self.model.global_weight.item():.6f}")
            f.write(f"Step {state.global_step}: mean_bias:{mean_bias} activation_bias: {current_bias} error={error:.4f} weight={self.model.global_weight.item():.6f}\n")
    
    def adjust_weight(self, error, step):
        # 计算PID分量
        proportional = self.kp * error
        self.integral += self.ki * error
        derivative = self.kd * (error - self.prev_error)
        
        # 计算权重调整量（带平滑因子）
        adjustment = proportional + self.integral + derivative
        smoothing_factor = 5
        new_weight = self.model.global_weight.item() * (1 + smoothing_factor * adjustment)
        
        # 应用边界保护
        new_weight = max(self.min_weight, min(self.max_weight, new_weight))
        
        # 更新模型权重
        self.model.global_weight = torch.tensor(
            new_weight, dtype=torch.float32, device=self.model.device
        )
        
        # 更新PID状态
        self.prev_error = error

        with open(self.model.log_path, "a") as f:
            print(f"Step {step}: Weight adjusted to {new_weight:.6f} | P: {proportional:.6f}, I: {self.integral:.6f}, D: {derivative:.6f} | PID:{proportional+self.integral+derivative:.6f}")
            f.write(f"Step {step}: Weight adjusted to {new_weight:.6f} | P: {proportional:.6f}, I: {self.integral:.6f}, D: {derivative:.6f} | PID:{proportional+self.integral+derivative:.6f}\n")


def load_custom_dataset(
    json_path: str,
    data_num: int = None,
    split: Optional[Union[str, float]] = None,
) -> Union[Dataset, dict]:

    file_path = json_path

    dataset = load_dataset('json', data_files=file_path, split='train')
    dataset = dataset.select(range(data_num))

    if isinstance(split, float) and 0 < split < 1:
        dataset = dataset.train_test_split(test_size=1-split, shuffle=True)
        return dataset
    elif split == 'all' or split is None:
        return dataset
    else:
        raise ValueError("split参数应为float(0-1)或None")
        
def save_params_to_json(output_path: str, **kwargs):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kwargs, f, indent=4, ensure_ascii=False)

class WeightedSFTTrainer(SFTTrainer):
    def _remove_unused_columns(self, dataset, description=None):
        return dataset
    
    def _prepare_dataset(self, dataset, *args, **kwargs):
        return dataset
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        weights = inputs.pop('weight')
        labels = inputs.pop('labels') # 这里注意检查训练的labels是否是我们掩盖处理后的labels

        outputs = model(**inputs)
        logits = outputs.logits

        # 预测值：去掉最后一个位置 [batch_size, seq_len-1, vocab_size]
        shift_logits = logits[..., :-1, :].contiguous() 
        
        # 标签值：去掉第一个位置 [batch_size, seq_len-1]
        shift_labels = labels[..., 1:].contiguous()

        # 逐个位置计算损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),  # 展平为 [batch_size*(seq_len-1), vocab_size]
            shift_labels.view(-1)                          # 展平为 [batch_size*(seq_len-1)]
        )
        loss = loss.view(shift_labels.size(0), -1)         # 恢复为 [batch_size, seq_len-1]

        # 处理有效标记
        valid_tokens = (shift_labels != -100).float().sum(dim=1).clamp(min=1)
        
        # 计算样本平均损失
        sample_loss = loss.sum(dim=1) / valid_tokens

        # 加权总损失
        weighted_loss = (sample_loss * weights).sum()
        weighted_loss = weighted_loss * model.global_weight

        return (weighted_loss, outputs) if return_outputs else weighted_loss
        
def train(
        model_path: str = "",
        data_path: str = "",
        output_dir: str = "",
        data_num: int = None,
        n_prefix: int = None,
        op_position: str = "",
        learning_rate: str = None,
        num_train_epochs:  int = 3,
        layer_type: str="",
        template_index: str="",
        pid_kp: float = 0.1,  # PID比例系数
        pid_ki: float = 0.0001,  # PID积分系数
        pid_kd: float = 0.01,  # PID微分系数
        target_bias: float = 0.54,  # 目标偏置L2范数
        min_weight: float = 1e-5,  # 权重最小值
        max_weight: float = 1e-1,  # 权重最大值
        limit_prop: float = 0.8,  
):

    path = "/mnt/usercache/huggingface/"
    model_path = path + model_path

    if not os.path.exists(model_path):
        model_path = model_path.replace('usercache', 'publiccache')

    if 'math10k' in data_path:
        data_type = 'math10k'
    elif 'prm800k' in data_path:
        data_type = 'prm800k'
    elif 'commonsense' in data_path:
        data_type = 'commonsense'
    elif 'ultrafeedback' in data_path:
        data_type = 'ultrafeedback'

    output_dir = os.path.join(output_dir, f'{data_num}_{data_type}_{layer_type}_{n_prefix}_{learning_rate}_{target_bias}')

    weight = 500/data_num
    learning_rate = float(learning_rate)

    model = load_RED_model(model_path=model_path, op_position=op_position, layer_type=layer_type)
    model.config = model.base_model.config
    model.device = model.base_model.device
    model.global_weight = torch.tensor(weight, dtype=torch.float32, device=model.device)

    scaling_monitor = ActivationScalingMonitor(
        model,
        weight=weight,
        target_bias=target_bias,
        pid_params=(pid_kp, pid_ki, pid_kd),
        weight_bounds=(min_weight, max_weight),
        limit_prop=limit_prop
    )
    
    train_config = {
        "model_path": model_path,
        "data_path": data_path,
        "data_num": data_num,
        "op_position": op_position,
        "template_index": template_index,
        "pid_kp": pid_kp,
        "pid_ki": pid_ki,  # PID积分系数
        "pid_kd": pid_kd,  # PID微分系数
        "n_prefix": n_prefix,  # 前缀长度
        "learning_rate": learning_rate,  # 学习率   
        "target_bias": target_bias,  # 目标偏置L2范数
        "min_weight": min_weight,  # 权重最小值
        "max_weight": max_weight,  # 权重最大值
        "limit_prop":limit_prop
    }
    
    output_dir = os.path.join(output_dir, f"{op_position}")
    log_dir = os.path.join(output_dir, "log")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config_path = os.path.join(output_dir, "train_config.json")
    print(train_config)
    print(model)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "right"

    template = get_prompt(tokenizer, template_index)

    def process_ultra_preference(example, task_type, k=None):
        question = example["instruction"]
        output = example["output"]

        prompt = template.replace("%s", question)
        output = f"{output}\n\n "

        example["prompt"] = prompt
        example["output"] = output
        example["text"] = example["prompt"] + example["output"]

        # 分词处理
        inputs = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            add_special_tokens=True
        )
        input_ids = inputs["input_ids"]
        example["text_length"] =  len(inputs.input_ids)

        # 计算prompt长度
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False).input_ids
        prompt_length = len(prompt_ids)
        example["prompt_length"] = prompt_length

        # 初始化labels
        labels = [-100] * len(input_ids)

        # 设置output部分标签
        output_start = prompt_length
        output_end = len(input_ids) - 1  # 假设EOS已添加

        if task_type == "prefix":
            # 设置前k个token
            for i in range(output_start, output_end):
                if (i - output_start) < k:
                    labels[i] = input_ids[i]
                else:
                    labels[i] = -100
            labels[-1] = input_ids[-1]  # EOS
        else:
            # 全部output
            for i in range(output_start, len(input_ids)):
                labels[i] = input_ids[i]
        
        # 更新字段
        example.update({
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "task_type": task_type
        })

        return example

    train_data = load_custom_dataset(data_path, data_num)

    split_dataset = train_data.train_test_split(test_size=0.1, shuffle=True)
    train_prefix = split_dataset["train"]
    train_full = split_dataset["test"]
    
    # 处理前缀数据集
    train_prefix = train_prefix.map(
        lambda ex: process_ultra_preference(ex, "prefix", n_prefix),
        num_proc=8
    )

    # 处理完整输出数据集 
    train_full = train_full.map(
        lambda ex: process_ultra_preference(ex, "full"),
        num_proc=8
    )

    # 合并数据集
    combined_data = concatenate_datasets([train_prefix, train_full])
    combined_data = combined_data.filter(lambda x:x["prompt_length"] <= MAX_INPUT_LENGTH and x["text_length"] <= MAX_LENGTH)
    combined_data = combined_data.remove_columns(
        ["instruction", "output", "prompt", "text", "text_length", "prompt_length"]
    )

    columns_to_remove = [
        "instruction", "output", "prompt", "text", 
        "text_length", "prompt_length", "task_type", "answer"  # 确保包含 task_type
    ]

    # 添加条件移除字段
    if 'math10k' in data_path or 'commonsense' in data_path or 'ultrafeedback' in data_path:
        columns_to_remove.append("input")
    if 'prm800k' in data_path:
        columns_to_remove.append("index")
        
    # 计算样本权重
    n_prefix = len(train_prefix)
    n_full = len(train_full)

    def add_weight(example):
        if example["task_type"] == "prefix":
            example["weight"] = weight
        else:
            example["weight"] = weight
        return example

    combined_data = combined_data.map(add_weight)
    custom_saving_callback = CustomModelSavingCallback()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    elif os.path.isfile(output_dir+"/train.log"):
        with open(output_dir+"/train.log", 'w') as f:
            pass

    class LogSaverCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=True, **kwargs):
            if logs:
                with open(output_dir+"/train.log", "a") as f:
                    f.write(f"Step {state.global_step}: {logs}\n")
    
    model.log_path = output_dir+"/train.log"

    if "qwen3-14b" in model_path.lower():
        batch_size = 4
    else:
        batch_size = 8
        
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=log_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="no",
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        warmup_ratio=0.1,
        report_to="none",
        logging_strategy="steps",
        weight_decay=1e-2,
        remove_unused_columns=False,
    )

    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        mlm=False,
    )

    def formatting_func(example):
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "labels": example["labels"],
            "task_type": example["task_type"]
        }
    
    existing_columns = combined_data.column_names
    columns_to_remove = [col for col in columns_to_remove if col in existing_columns]
    combined_data = combined_data.remove_columns(columns_to_remove)

    trainer = WeightedSFTTrainer(
        model=model,
        # dataset_text_field='labels', # 和formatting_func参数2选1
        formatting_func=formatting_func,
        data_collator=data_collator,
        train_dataset=combined_data,
        args=training_args,
        callbacks=[custom_saving_callback, scaling_monitor, LogSaverCallback()],
    )

    trainer.train()
    model.save_model(os.path.join(output_dir, "delta_vector.pth"))
    save_params_to_json(config_path, **train_config)

if __name__ == "__main__":
    fire.Fire(train)