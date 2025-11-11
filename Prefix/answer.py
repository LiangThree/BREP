import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-1])
sys.path.append(main_dir)
from template import *

from datasets import load_dataset
from tqdm import tqdm 
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
import json
from transformers import GenerationConfig,AutoConfig

import fire
from model import ActivationLLama
from datasets import load_dataset, Dataset
from typing import Optional, Union
import pdb 
from make_answer_json import make_answer
from make_answer_prefix import make_answer_prefix

def load_custom_dataset(
    json_path: str,
    data_num: int = None,
    split: Optional[Union[str, float]] = None,
) -> Union[Dataset, dict]:
    
    with open(json_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    data_list = list(data_dict.values())

    if data_num == -1:
        data_num = len(data_list)
    if data_num is not None:
        data_list = data_list[:min(data_num, len(data_list))]

    dataset = Dataset.from_list(data_list)

    if isinstance(split, float) and 0 < split < 1:
        dataset = dataset.train_test_split(test_size=1-split, shuffle=True)
        return dataset
    elif split == 'all' or split is None:
        return dataset
    else:
        raise ValueError("split参数应为float(0-1)或None")

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
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

def save_params_to_json(output_path: str, **kwargs):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kwargs, f, indent=4, ensure_ascii=False)

def main(
        model_path: str = "",
        data_num: int = None,
        peft:bool = False,
        n_prefix: int = None,
        start: int = -1,
        end: int = -1,
        peft_path:str="",
        is_train_return:bool = True,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.2,
        dataset: str="",
        template_index:str="",
        
):

    data_path = os.path.join(peft_path, f'{template_index}_{dataset}_eval_prefix{n_prefix}.json')

    peft_path_list = peft_path.split('/')
    config_path = peft_path_list[3]
    prefix = n_prefix

    path = "/mnt/usercache/huggingface/"
    if not os.path.exists(path):
        path = "/mnt/publiccache/huggingface/"
    model_path = path + model_path
    
    config_path = os.path.join(peft_path, 'eval_config.json')
    peft_path = os.path.join(peft_path, 'delta_vector.pth')

    if os.path.exists(config_path):
        print(f'{config_path} exists')
    
    if os.path.exists(peft_path):
        print(f'{peft_path} exists')

    if "attn_o" in peft_path.lower():
        op_position = "attn_o"
    elif "ffn_up" in peft_path.lower():
        op_position = "ffn_up"
    elif "ffn_down" in peft_path.lower():
        op_position = "ffn_down"
    elif "layer" in peft_path.lower():
        op_position = "layer"
    else:
        print("Can't find op_position!")
        exit(0)

    eval_config = {
        "model_path": model_path,
        "data_path": data_path,
        "data_num": data_num,
        "op_position": op_position,
        "peft": peft,
        "start": start,
        "end": end,
        "peft_path": peft_path,
        "is_train_return": is_train_return,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "repetition_penalty": repetition_penalty,
        "template_index": template_index
    }
    save_params_to_json(config_path, **eval_config)
    print(eval_config)
    
    if peft == "lora":
        model = AutoPeftModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="auto")
    elif peft == "RED":
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="auto")
        model = ActivationLLama(model, op_position=op_position, prefix=prefix)
        model.load_model(peft_path)
    elif peft == "ft":
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="auto")

    print(model)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", torch_dtype=torch.bfloat16)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    template = get_prompt(tokenizer, template_index)

    def process_alpacaeval(example):
        question = example['Question']
        prompt = template.replace('%s', question)
        example["prompt"] = prompt
        return example

    make_answer(dataset, data_path)

    dataset = load_custom_dataset(data_path, data_num)
    # dataset = load_dataset("tatsu-lab/alpaca_eval")["eval"]
    dataset = dataset.map(process_alpacaeval)

    answer_dict = read_json_file(data_path)
    
    # A100 & max_length 512 : batch_size=128
    # A100 & max_length 2048 : batch_size=64
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
            max_length=2048,
        ).to("cuda")
        
        outputs = generate(inputs, model, tokenizer, data_path)
        
        completions = tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        
        for i, (prompt, completion) in enumerate(zip(batch_prompts, completions)):
            global_idx = batch_idx + i
            answer_dict[str(global_idx)]["red"] = completion

    save_list_to_json(answer_dict, data_path)


if __name__ == "__main__":
    fire.Fire(main)