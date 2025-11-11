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
from peft import PeftModel
from tqdm import *
from template import *
from make_answer_json import save_json_to_file
from make_answer_json import make_answer
from math import cos, pi

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parallel_size = 1
max_length = 2048
temperature_list = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8]

from vllm import LLM, SamplingParams

def load_custom_dataset(
    data_dict: dict,
    data_num: int = None,
    split: Optional[Union[str, float]] = None,
) -> Union[Dataset, dict]:

    data_list = list(data_dict.values())

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

def get_data(data_files:str, data_num:int, split:Optional[Union[str, float]] = None):

    dataset = load_dataset('json', data_files=data_files, split='train')
    dataset = dataset.select(range(data_num))
    
    if isinstance(split, float) and 0 < split < 1:
        dataset = dataset.train_test_split(test_size=1-split, shuffle=True)
    
    return dataset['train'], dataset['test']

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

def eval_one_data(model_answer: str, answer: str):
    
    if answer in model_answer:
        return True
    else:
        
        try:
            int_answer = float(answer)

            if int_answer.is_integer():
                int_answer = int(int_answer)
                formatted_answer = f"{int_answer:,}" if abs(int_answer) >= 1000 else str(int_answer)
            
            if answer in model_answer or str(int_answer) in model_answer or formatted_answer in model_answer:
                return True
            else:
                return False
            
        except:
            return False


def make_train(read_path, data_num, n_round):
    
    data = read_json_file(read_path) 
    data_dict = {}

    if type(data) == dict:
        for key in data.keys():
            one_data = data[key]
            Question = one_data['Question']
            Output = one_data['Output']
            Answer = one_data['Answer']
            model_answer = one_data['base']
            Answer_count = one_data['Answer_count']

            data_dict[key] = {'Question':Question, 'Output':Output, 'Answer':Answer, 'base':model_answer, 'Answer_count':Answer_count}
            
            if not eval_one_data(model_answer, Answer):
                data_dict[key]['Answer_count'] = n_round

    elif type(data) == list:
        for index, one_data in enumerate(data[:min(data_num,len(data))]):
            Question = one_data['instruction']
            Output = one_data['output']
            Answer = one_data['answer']
            
            data_dict[index] = {'Question':Question, 'Output':Output, 'Answer':Answer, 'Answer_count': 1}
    
    return data_dict


def dynamic_temperature_scheduler(current_round, max_rounds=8):
    peak_round = max_rounds * 0.6 
    if current_round <= peak_round:
        return 0.3 + 0.5 * (current_round / peak_round)  # 线性升温至0.8
    else:
        return 0.8 * (1 - (current_round - peak_round)/(max_rounds - peak_round))  # 线性降温
    

def first_round(model_path:str, data_num:int, output_path:str, template:str, dataset:str, vllm:bool, model, tokenizer):

    read_path = f"/mnt/userdata/liangsirui/MyProject/Prefix/dataset/{dataset}/train.json"
    if not os.path.exists(read_path):
        read_path = f"/mnt/userdata/MyProject/Prefix/dataset/{dataset}/train.json"
    
    save_path = os.path.join(output_path, dataset, f"{dataset}_1.json")
    data_dict = make_train(read_path, data_num, 1)
    
    def process_alpacaeval(example):
        question = example['Question']
        prompt = template % question
        example["prompt"] = prompt
        return example

    dataset = load_custom_dataset(data_dict, data_num)
    dataset = dataset.map(process_alpacaeval)
    answer_dict = data_dict

    if vllm:
        prompts = []
        for i in range(len(dataset)):
            question = dataset[i]['Question']
            prompt = template % question
            prompts.append(prompt)

        print(f'current temperature 0  current max_length {max_length}')
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_length,
            stop=[], 
            skip_special_tokens=True
        )

        outputs = model.generate(prompts, sampling_params)
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            answer_dict[i]['base'] = generated_text
            answer_dict[i]['eval'] = eval_one_data(answer_dict[i]['base'], answer_dict[i]['Answer'])
            
    else:
        for i in tqdm(range(len(dataset))):
            
            prompt = dataset[i]["prompt"]
            prompt_ids = tokenizer(prompt, return_tensors='pt').to(model.base_model.device)
            
            outputs = model.generate(
                **prompt_ids,
                max_new_tokens=300,
                do_sample=False,  # 禁用采样
                num_beams=1,      # 贪婪搜索
                use_cache=True,    # 启用KV缓存
            )

            completion_good = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_dict[i]['base'] = completion_good
            
    
    remove_list = []
    for key in answer_dict.keys():
        if answer_dict[key]['Answer'] == None:
            remove_list.append(key)

    for key in remove_list:
        answer_dict.pop(key)

    save_list_to_json(answer_dict, save_path)

def the_next_round(model_path:str, data_num:int, output_path:str, template:str, dataset:str, n_round:int, vllm:bool, model, tokenizer):

    pre_round = n_round-1
    read_path = os.path.join(output_path, dataset, f"{dataset}_{pre_round}.json")
    save_path = os.path.join(output_path, dataset, f"{dataset}_{n_round}.json")
    data_dict = make_train(read_path, data_num, n_round)

    def process_alpacaeval(example):
        question = example['Question']
        prompt = template % question
        example["prompt"] = prompt
        return example

    dataset = load_custom_dataset(data_dict, data_num)
    dataset = dataset.map(process_alpacaeval)
    answer_dict = data_dict

    if vllm:
        
        index = []
        prompts = []
        
        for key in answer_dict.keys():
            if answer_dict[key]['Answer_count'] == n_round:
                question = answer_dict[key]['Question']
                prompt = template % question
                
                index.append(key)
                prompts.append(prompt)

        # temperature = dynamic_temperature_scheduler(n_round, 8)
        temperature = temperature_list[n_round-1]
        print(f'current temperature {temperature}   current max_length {max_length}')

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_length,
            stop=[], 
            skip_special_tokens=True
        )

        outputs = model.generate(prompts, sampling_params)
        
        for key, output in zip(index, outputs):
            generated_text = output.outputs[0].text
            answer_dict[key]['base'] = generated_text
            answer_dict[key]['eval'] = eval_one_data(generated_text, answer_dict[key]['Answer'])
        
        save_list_to_json(answer_dict, save_path)
        
        return len(prompts)



def train_model(model_path:str, data_num:int, output_path:str, template_index:int, dataset:str, vllm:bool):

    n_round = 8

    print("template_index:", template_index)
    print("----------------------- template -----------------------")
    print(prompt_template[template_index])
    print("--------------------------------------------------------")
    template = prompt_template[template_index]

    path = "/mnt/usercache/huggingface/"
    if not os.path.exists(path):
        path = "/mnt/publiccache/huggingface/"
    model_path = path + model_path

    if vllm:
        model = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=parallel_size,  
            gpu_memory_utilization=0.9,
            # disable_custom_all_reduce=True  # 添加此参数
        )

        tokenizer = model.get_tokenizer()

    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, model_max_length=max_length,
            padding_side="right", use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

    current_count = 0
    
    for current_round in range(1, n_round+1):
        print(f'This is the {current_round} round')
        if current_round==1:
            save_path = os.path.join(output_path, dataset, f"{dataset}_1.json")
            if os.path.isfile(save_path): 
                print(f"file {save_path} is exists, gap {current_round} round")
                continue
            first_round(model_path, data_num, output_path, template, dataset, vllm, model, tokenizer)

        else:
            save_path = os.path.join(output_path, dataset, f"{dataset}_{current_round}.json")
            if os.path.isfile(save_path):
                print(f"file {save_path} is exists, gap {current_round} round")
                continue

            pre_count = current_count
            current_count = the_next_round(model_path, data_num, output_path, template, dataset, current_round, vllm, model, tokenizer)

            if pre_count == current_count:
                break 


def main():
    parser = argparse.ArgumentParser(description="A simple script that takes different arguments.")
    parser.add_argument('-model_path', '--model_path', type=str, default=None)
    parser.add_argument('-output_path', '--output_path', type=str, default=None)
    parser.add_argument('-data_num', '--data_num', type=int)
    parser.add_argument('-template_index', '--template_index', type=int, default=None)
    parser.add_argument('-dataset', '--dataset', type=str)
    parser.add_argument('-vllm', '--vllm', type=bool)
    args = parser.parse_args()

    train_model(**vars(args))


if __name__ == '__main__':
    main()