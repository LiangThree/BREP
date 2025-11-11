import argparse
import fire
import torch, transformers
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os
import sys
cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-1])
sys.path.append(main_dir)
from Prefix.make_answer_json import make_answer
import json
import pdb
from datasets import load_dataset, Dataset
from typing import Optional, Union
from peft import PeftModel
from tqdm import *
from template import *

device = "cuda"

def load_custom_dataset(
    json_path: str,
    data_num: int = None,
    split: Optional[Union[str, float]] = None,
) -> Union[Dataset, dict]:
    
    # 加载 JSON 文件（原始格式是字典）
    with open(json_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    # 将字典转换为列表格式
    data_list = list(data_dict.values())

    if data_num == -1:
        data_num = len(data_list)
    # 限制数据条数，避免超界
    if data_num is not None:
        data_list = data_list[:min(data_num, len(data_list))]

    # 转换为 Hugging Face Dataset 格式
    dataset = Dataset.from_list(data_list)

    # # 加载原始数据集
    # dataset = load_dataset('json', data_files=json_path, split='train')
    # dataset = dataset.select(range(data_num))
    
    # 处理数据集划分
    if isinstance(split, float) and 0 < split < 1:
        dataset = dataset.train_test_split(test_size=1-split, shuffle=True)
        return dataset
    elif split == 'all' or split is None:
        return dataset
    else:
        raise ValueError("split参数应为float(0-1)或None")

def get_data(data_files:str, data_num:int, split:Optional[Union[str, float]] = None):

    dataset = load_dataset('json', data_files=data_files, split='train')
    dataset = dataset.select(range(data_num))
    
    # 处理数据集划分
    if isinstance(split, float) and 0 < split < 1:
        dataset = dataset.train_test_split(test_size=1-split, shuffle=True)
    
    return dataset['train'], dataset['test']

def read_json_file(file_path):
    try:
        # 以只读模式打开文件，使用 UTF-8 编码
        with open(file_path, 'r', encoding='utf-8') as file:
            # 加载 JSON 数据
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except json.JSONDecodeError:
        print(f"文件 {file_path} 不是有效的 JSON 格式。")

def save_list_to_json(data_list, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)  # 使用 indent 格式化 JSON
    print(f"数据已成功存储到 {file_path}")

def train_model(model_path:str, lora_path:str, data_num:int, dataset:str, template_index:str):

    path = "/mnt/usercache/huggingface/"
    if not os.path.exists(path):
        path = "/mnt/publiccache/huggingface/"
    model_path = path + model_path

    path = "/mnt/userdata/MyProject/" 
    if not os.path.exists(path):
        path = "/mnt/userdata/liangsirui/MyProject/"
    
    data_path = os.path.join(lora_path, f'{template_index}_{dataset}_eval.json')
    lora_path = os.path.join(lora_path)

    make_answer(dataset, data_path)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, trust_remote_code=True)

    # get tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, model_max_length=2048,
        padding_side="left", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    template = get_prompt(tokenizer, template_index)

    model = PeftModel.from_pretrained(model, lora_path)
    model.to(device)
    model.eval()

    def process_alpacaeval(example):
        question = example['Question']
        prompt = template.replace("%s", question)
        example["prompt"] = prompt
        return example

    dataset = load_custom_dataset(data_path, data_num)
    dataset = dataset.map(process_alpacaeval)

    answer_dict = read_json_file(data_path)

    batch_size = 64
    all_prompts = [ex["prompt"] for ex in dataset]
    total_samples = len(all_prompts)

    for batch_idx in tqdm(range(0, total_samples, batch_size)):
        batch_prompts = all_prompts[batch_idx:batch_idx + batch_size]
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to("cuda")

        # print('qwen math generate config')
        # outputs = model.generate(
        #     **inputs,
        #     max_new_tokens=1024,
        #     do_sample=False,  
        #     num_beams=1,      
        #     use_cache=True,    
        # )

        outputs = generate(inputs, model, tokenizer, data_path)
        
        completions = tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        
        for i, (prompt, completion) in enumerate(zip(batch_prompts, completions)):
            global_idx = batch_idx + i
            answer_dict[str(global_idx)]['lora'] = completion

    save_list_to_json(answer_dict, data_path)



if __name__ == '__main__':
    fire.Fire(train_model)