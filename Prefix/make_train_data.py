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
from make_answer_json import add_index_to_json, save_json_to_file
from eval import eval_one_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parallel_size = 4

from vllm import LLM, SamplingParams


def load_custom_dataset(
    json_path: str,
    data_num: int = None,
    split: Optional[Union[str, float]] = None,
) -> Union[Dataset, dict]:
    
    with open(json_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

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



def make_train(read_path, save_path, data_num, n_round):
    
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

            if not eval_one_data(model_answer, Answer):
                Answer_count = n_round 

            data_dict[key] = {'Question':Question, 'Output':Output, 'Answer':Answer, 'base':model_answer,'Answer_count': Answer_count}

    elif type(data) == list:
        for index, one_data in enumerate(data[:min(data_num,len(data))]):
            Question = one_data['instruction']
            Output = one_data['output']
            Answer = one_data['answer']
            
            data_dict[index] = {'Question':Question, 'Output':Output, 'Answer':Answer, 'Answer_count': 1}
    
    save_json_to_file(data_dict, save_path)

def first_round(model_path:str, data_num:int, output_path:str, template:str, dataset:str, vllm:bool, model, tokenizer):

    read_path = f"/mnt/userdata/liangsirui/MyProject/Prefix/dataset/{dataset}/train.json"
    save_path = os.path.join(output_path, dataset, f"{dataset}_1.json")
    make_train(read_path, save_path, data_num, 1)
    
    def process_alpacaeval(example):
        question = example['Question']
        prompt = template % question
        example["prompt"] = prompt
        return example

    dataset = load_custom_dataset(save_path, data_num)
    dataset = dataset.map(process_alpacaeval)
    answer_dict = read_json_file(save_path)

    if vllm:
        prompts = []
        for i in range(len(dataset)):
            question = dataset[i]['Question']
            prompt = template % question
            prompts.append(prompt)
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=300,
            stop=[], 
            skip_special_tokens=True
        )

        outputs = model.generate(prompts, sampling_params)
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            answer_dict[str(i)]['base'] = generated_text
            
    else:
        for i in tqdm(range(len(dataset))):
            
            generate_dict = {}
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
            answer_dict[str(i)]['base'] = completion_good
    
    save_list_to_json(answer_dict, save_path)

def the_next_round(model_path:str, data_num:int, output_path:str, template:str, dataset:str, n_round:int, vllm:bool, model, tokenizer):

    pre_round = n_round-1
    read_path = os.path.join(output_path, dataset, f"{dataset}_{pre_round}.json")
    save_path = os.path.join(output_path, dataset, f"{dataset}_{n_round}.json")
    make_train(read_path, save_path, data_num, n_round)

    def process_alpacaeval(example):
        question = example['Question']
        prompt = template % question
        example["prompt"] = prompt
        return example

    dataset = load_custom_dataset(save_path, data_num)
    dataset = dataset.map(process_alpacaeval)
    answer_dict = read_json_file(save_path)

    pdb.set_trace()

    if vllm:
        prompts = []
        for i in range(len(dataset)):
            question = dataset[i]['Question']
            prompt = template % question
            prompts.append(prompt)
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=300,
            stop=[], 
            skip_special_tokens=True
        )

        outputs = model.generate(prompts, sampling_params)
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            answer_dict[str(i)]['base'] = generated_text



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
            tensor_parallel_size=parallel_size,  # 暂时禁用tensor并行
            gpu_memory_utilization=0.9,
            disable_custom_all_reduce=True  # 添加此参数
        )

        tokenizer = model.get_tokenizer()

    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, model_max_length=2048,
            padding_side="right", use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

    for i in range(1, n_round+1):
        print(f'This is the {n_round} round')
        if i==1:
            save_path = os.path.join(output_path, dataset, f"{dataset}_1.json")
            if os.isfile(save_path):
                continue
            first_round(model_path, data_num, output_path, template, dataset, vllm, model, tokenizer)
        else:
            save_path = os.path.join(output_path, dataset, f"{dataset}_{i}.json")
            if os.isfile(save_path):
                continue
            the_next_round(model_path, data_num, output_path, template, dataset, n_round, vllm, model, tokenizer)


def main():
    parser = argparse.ArgumentParser(description="Train script with necessary arguments.")
    # 仅保留长选项，移除无效的短选项
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory path')
    parser.add_argument('--data_num', type=int, required=True, help='Number of data samples')
    parser.add_argument('--template_index', type=int, required=True, help='Index of the prompt template')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--n_round', type=int, default=8, help='Number of training rounds')
    # 使用 action='store_true' 处理布尔参数
    parser.add_argument('--vllm', action='store_true', help='Use vLLM for inference')
    
    args = parser.parse_args()
    train_model(**vars(args))


if __name__ == '__main__':
    main()