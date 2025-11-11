import argparse
import torch, transformers
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os
import json
import pdb
from datasets import load_dataset, Dataset
from typing import Optional, Union
from transformers.trainer_callback import TrainerCallback
import fire
import os

import sys
cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-1])
sys.path.append(main_dir)

from template import *
import torch

device = "cuda"

def get_data(data_files:str, data_num:int, split:Optional[Union[str, float]] = None):

    dataset = load_dataset('json', data_files=data_files, split='train')
    dataset = dataset.select(range(data_num))
    
    # 处理数据集划分
    if isinstance(split, float) and 0 < split < 1:
        dataset = dataset.train_test_split(test_size=1-split, shuffle=True)
    
    return dataset['train'], dataset['test']

def train_model(model_path:str, data_path:str, layer:int, output_dir:str, data_num:int, template_index:str, learning_rate: float = 2e-5):

    path = "/mnt/usercache/huggingface/"
    if not os.path.exists(path):
        path = "/mnt/publiccache/huggingface/"
    model_path = path + model_path

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True, use_cache=True, torch_dtype=torch.bfloat16)

    # get tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, model_max_length=2048,
        padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    template = get_prompt(tokenizer, template_index)

    peft_config = LoraConfig(
        r=4, lora_alpha=32, target_modules=["o_proj"], layers_to_transform=[layer],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    # model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    
    train_data, eval_data = get_data(data_path, data_num, 0.8)

    # 将数据转换为 Hugging Face 的 Dataset 格式
    train_dataset = Dataset.from_dict({
        "input": [template % example['instruction'] for example in train_data],
        "output": [str(example['output']) for example in train_data]
    })

    valid_dataset = Dataset.from_dict({
        "input": [template % example['instruction'] for example in eval_data],
        "output": [str(example['output']) for example in eval_data]
    })

    def tokenize_function(example):
        return tokenizer(example["input"], example["output"], truncation=True, max_length=1024)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if "math10k" in data_path:
        data_type = 'math10k'
    elif "prm800k" in data_path:
        data_type = 'prm800k'
    output_dir = os.path.join(output_dir, f"{data_num}_{data_type}_lora_{learning_rate}")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    class LogSaverCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=True, **kwargs):
            if logs:
                with open(output_dir+"/train.log", "a") as f:
                    f.write(f"Step {state.global_step}: {logs}\n")

    training_args = TrainingArguments(
        output_dir="./lora/lora_output",
        overwrite_output_dir=True, 
        bf16=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=learning_rate,
        num_train_epochs=2,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    class FixedTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            
            if self.args.n_gpu > 1:
                main_device = self.model.device
                
                return {k: v.to(main_device) if isinstance(v, torch.Tensor) else v 
                        for k, v in inputs.items()}
            
            return inputs
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            inputs = self._prepare_inputs(inputs)
            
            outputs = model(**inputs)
            
            loss = outputs.loss
            if self.args.n_gpu > 1 and loss.device != self.model.device:
                loss = loss.to(self.model.device)
                
            return (loss, outputs) if return_outputs else loss


    trainer = FixedTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        data_collator=data_collator,
        callbacks=[LogSaverCallback()],
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f'model saved in {output_dir}')


if __name__ == '__main__':
    fire.Fire(train_model)