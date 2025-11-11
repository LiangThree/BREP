import os
import sys
from datasets import load_dataset, Dataset
from typing import Optional, Union
import pdb
import json
import torch.optim as optim

cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-1])
sys.path.append(main_dir)

from template import *
import logging
import fire
from datasets import load_dataset
from transformers import  AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from transformers.trainer_callback import TrainerCallback
import os
import torch
from transformers import AutoModelForCausalLM
from model import ActivationLLama
from transformers import DataCollatorForLanguageModeling
from functools import partial

MAX_INPUT_LENGTH = 512
MAX_LENGTH = 512

device_map = "auto"

def load_RED_model(model_path, op_position, layer_type):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
        use_auth_token=True,
        torch_dtype=torch.bfloat16,
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

from transformers.trainer_callback import TrainerCallback
import torch


class ActivationScalingMonitor(TrainerCallback):
    def __init__(self, model):
        self.model = model
        self.prev_scaling = None  # 记录上一次的 activation_scaling 值
        self.prev_bias = None
        self.num_layers = self.model.base_model.model.config.num_hidden_layers

    def on_step_end(self, args, state, control, **kwargs):
        # 获取当前的 activation_scaling 值
        current_scaling = 0
        current_bias = 0

        for layer in range(self.num_layers):
            if "attn_q" in self.model.op_position:
                module = self.model.base_model.model.layers[layer].self_attn.q_proj
            elif "attn_k" in self.model.op_position:
                module = self.model.base_model.model.layers[layer].self_attn.k_proj
            elif "attn_v" in self.model.op_position:
                module = self.model.base_model.model.layers[layer].self_attn.v_proj
            elif "attn_o" in self.model.op_position:
                module = self.model.base_model.model.layers[layer].self_attn.o_proj
            elif "ffn_up" in self.model.op_position:
                module = self.model.base_model.model.layers[layer].mlp.up_proj
            elif "ffn_down" in self.model.op_position:
                module = self.model.base_model.model.layers[layer].mlp.down_proj
            elif self.model.op_position == "layer":
                module = self.model.base_model.model.layers[layer]

            if hasattr(module.delta_vector, 'activation_scaling'):
                delta = module.delta_vector.activation_scaling.detach().to(torch.float).cpu().numpy()
                current_scaling += delta
            
            if hasattr(module.delta_vector, 'activation_bias'):
                current_bias += module.delta_vector.activation_bias.detach().to(torch.float).cpu().numpy()
                
        if hasattr(module.delta_vector, 'activation_scaling'):
            print(f"Step {state.global_step}: activation_scaling mean: {current_scaling.mean()}")
        
        if hasattr(module.delta_vector, 'activation_bias'):
            print(f"Step {state.global_step}: activation_bias mean: {current_bias.mean()}")


def load_custom_dataset(
    json_path: str,
    data_num: int = None,
    split: Optional[Union[str, float]] = None,
) -> Union[Dataset, dict]:
    """
    加载自定义JSON格式数据集
    
    参数:
        json_path (str): JSON文件路径
        split (str/float, 可选): 数据集划分比例（当需要分割时）
        convert_answer_to_float (bool): 是否将answer字段转为float类型
    
    返回:
        Dataset对象或包含多个划分的DatasetDict
    """
    data_path = "/mnt/userdata/MyProject/Prefix/" 
    if not os.path.exists(data_path):
        data_path = "/mnt/userdata/liangsirui/MyProject/Prefix/"
    file_path = data_path + json_path

    # 加载原始数据集
    dataset = load_dataset('json', data_files=file_path, split='train')
    dataset = dataset.select(range(data_num))

    # 处理数据集划分
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

def train(
        model_path: str = "",
        data_path: str = "",
        output_dir: str = "",
        data_num: int = None,
        op_position: str = "",
        learning_rate: float = 2e-5,
        num_train_epochs:  int = 3,
        layer_type: str="",
        template_index: str=""
):
    
    print("----------------------- template -----------------------")
    print(prompt_template[template_index])
    print("--------------------------------------------------------")

    path = "/mnt/usercache/huggingface/"
    if not os.path.exists(path):
        path = "/mnt/publiccache/huggingface/"
    model_path = path + model_path

    if 'math10k' in data_path:
        data_type = 'math10k'
    elif 'prm800k' in data_path:
        data_type = 'prm800k'

    output_dir = os.path.join(output_dir, f'{data_num}_{data_type}_{layer_type}_{learning_rate}')
    
    if op_position == "qk":
        op_position = ["attn_q","attn_k"]
    
    model = load_RED_model(model_path=model_path, op_position=op_position, layer_type=layer_type)
    train_config = {
        "model_path": model_path,
        "data_path": data_path,
        "data_num": data_num,
        "op_position": op_position,
        "template_index": template_index
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

    def process_ultra_preference(example):
        
        question = example["instruction"]
        output = example["output"]

        template = prompt_template[template_index] % question
        output = f"{output}\n\n "
        
        # template = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        example["prompt"] = template
        example["prompt_length"] = len(tokenizer(example["prompt"]).input_ids)
        example["output"] = output
        example["text"] = example["prompt"] + example["output"] + " </s>"
        example["text_length"] = len(tokenizer(example["text"]).input_ids)

        tokenized = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        example["input_ids"] = tokenized["input_ids"][0]
        example["attention_mask"] = tokenized["attention_mask"][0]
        example["completion"] = output

        return example

    train_data = load_custom_dataset(data_path, data_num)
    train_data = train_data.map(
        partial(process_ultra_preference),
        num_proc=8
    )
    train_data = train_data.filter(lambda x:x["prompt_length"] <= MAX_INPUT_LENGTH and x["text_length"] <= MAX_LENGTH)
    custom_saving_callback = CustomModelSavingCallback()

    logging.basicConfig(
        filename="training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

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
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    model.config = model.base_model.config

    trainer = SFTTrainer(
        model=model,
        # dataset_text_field='text',
        data_collator=data_collator,
        train_dataset=train_data,
        args=training_args,
        callbacks=[custom_saving_callback, ActivationScalingMonitor(model), LogSaverCallback()],
    )

    trainer.train()
    model.save_model(os.path.join(output_dir, "delta_vector.pth"))
    save_params_to_json(config_path, **train_config)


if __name__ == "__main__":
    # pip install transformers==4.47.1 trl==0.15.2 "pydantic<2.0"
    fire.Fire(train)