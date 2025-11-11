import json
import os
import fire
import pdb
import glob
import os

def find_json_files(folder_path):
    # 使用递归模式匹配所有子目录的.json文件
    search_pattern = os.path.join(folder_path, "**", "*.json")
    return glob.glob(search_pattern, recursive=True)

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
    # print(f"数据已成功存储到 {file_path}")

def extract_response(text: str, data_path:str) -> str:
    start_marker = "### Response:"
    start_idx = text.find(start_marker)
    
    if start_idx == -1:
        return ""
    
    response_text = text[start_idx + len(start_marker):].strip()
    
    end_marker = "assistant"
    end_idx = response_text.find(end_marker)

    if 'template0' in data_path:
        if end_idx != -1:
            response_text = response_text[:end_idx].strip()

    else:
        if end_idx != -1:
            second_end_idx = response_text.find(end_marker, end_idx + len(end_marker))
            if second_end_idx != -1:
                response_text = response_text[:second_end_idx].strip()
    
    response_text.replace('assistant', '')
    response_text.strip()
    
    return response_text


def eval_one_data(
    model_answer: str,
    answer: str,
):
    int_answer = float(answer)

    if int_answer.is_integer():
        int_answer = str(int(int_answer))
        formatted_answer = f"{int_answer:,}" if abs(int_answer) >= 1000 else str(int_answer)
    
    if answer in model_answer or int_answer in model_answer or formatted_answer in model_answer:
        return True
    else:
        return False

def extract_between_assistants(text, keyword="assistant"):
    indexes = []
    start = 0
    length = len(keyword)
    
    # 获取所有关键词位置
    while True:
        pos = text.find(keyword, start)
        if pos == -1: break
        indexes.append(pos)
        start = pos + length
    
    if len(indexes) < 1: return text
    
    return text[indexes[0] + length : ]



def eval_data(
        data_path: str = "",
        train_config: dict=None,
        eval_num: int = None
):
    data = read_json_file(data_path)
    file_name = data_path.split('/')[-1]

    all_key = data["0"].keys()
    eval_type = set(all_key)-set(["Question", "Answer", "eval", "Output"])
    eval_type = tuple(eval_type)[0]

    correct_count = 0
    question_count = 0

    # if 'lora' in data_path and 'math500' in data_path:
    #     pdb.set_trace()
    
    for index, key in enumerate(data.keys()):

        one_data = data[key]
        one_data["eval"] = False

        if int(eval_num) == -1:
            eval_num = len(data)
        if eval_type not in one_data.keys() or int(index) > int(eval_num):
            break

        answer = one_data["Answer"]
        model_answer = one_data[eval_type]
        model_answer = extract_between_assistants(model_answer, keyword="assistant")
        
        if 'gpqa' in data_path:
            error_answer = ['(A)', '(B)', '(C)', '(D)']
            error_answer = [i for i in error_answer if i != answer]
            error_1 = error_answer[0]
            error_2 = error_answer[1]
            error_3 = error_answer[2]
            
            one_data["eval"] = False
            
            if '{'+answer+'}' in model_answer:
                one_data["eval"] = True
            elif '{'+answer[1]+'}' in model_answer:
                one_data["eval"] = True
            elif answer in model_answer and error_1 not in model_answer and error_2 not in model_answer and error_3 not in model_answer:
                one_data["eval"] = True
        
        elif 'hellaswag' in data_path:
            error_answer = ['ending1', 'ending2', 'ending3', 'ending4']
            error_answer = [i for i in error_answer if i != answer]
            error_1 = error_answer[0]
            error_2 = error_answer[1]
            error_3 = error_answer[2]
            
            one_data["eval"] = False
            
            if '{'+answer+'}' in model_answer:
                one_data["eval"] = True
            elif answer in model_answer and error_1 not in model_answer and error_2 not in model_answer and error_3 not in model_answer:
                one_data["eval"] = True

        elif 'boolq' in data_path or 'piqa' in data_path:
            if answer in model_answer:
                one_data["eval"] = True
            else:
                one_data["eval"] = False

        if 'math500' in data_path or 'aime' in data_path:
            start = model_answer.find('boxed{')
            if start == -1:
                one_data["eval"] = False
                # model_answer = model_answer
            else:
                start += len('boxed{')
                model_answer = model_answer[start:].replace('<|endoftext|>','')
            
            remove_str = ['\\left,' '\\right', ' ']
            for s in remove_str:
                model_answer = model_answer.replace(s, '')

            if str(answer) in model_answer:
                one_data["eval"] = True
            else:
                one_data["eval"] = False
            
        elif 'gsm8k' in data_path or 'svamp' in data_path or 'mathqa' in data_path or 'amc23' in data_path:
            if str(answer) in model_answer:
                one_data["eval"] = True
            else:
                if answer == None:
                    continue

                try:
                    int_answer = float(answer)

                    if int_answer.is_integer():
                        int_answer = int(int_answer)
                        formatted_answer = f"{int_answer:,}" if abs(int_answer) >= 1000 else str(int_answer)

                    if str(answer) in model_answer or str(int_answer) in model_answer or formatted_answer in model_answer:
                        one_data["eval"] = True
                    else:
                        one_data["eval"] = False
                    
                except Exception as e:
                    one_data["eval"] = False
        
        if one_data["eval"]:
            correct_count += 1
        
        question_count += 1
    
    if question_count==0:
        return
    
    dataset = train_config["dataset"]
    data_num = train_config["data_num"]
    epoch = train_config["epoch"]
    lr = train_config["lr"]
    prefix = train_config["prefix"]

    acc= (correct_count/question_count)*100
    print(f"|{file_name}|{dataset}|{data_num}|{epoch}|{prefix}|{lr}|{correct_count}|{question_count}|{acc:.1f}|")
    
    save_list_to_json(data, data_path)


import re
from typing import List, Dict

def parse_training_config(path: str) -> Dict[str, str]:
    path_parts = path.split("/")

    config_path = path_parts[3]
    epoch = path_parts[4]

    parts = config_path.split("_")
    train_config = {
        "data_num": parts[0],
        "dataset": parts[1],
        "epoch": epoch,
        "lr": parts[-1],
        "prefix": parts[-3]
    }

    return train_config

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,default=None)
parser.add_argument('--eval_num', type=str,default=None) 
parser.add_argument('--model', type=str,default=None) 
parser.add_argument('--dataset', type=str,default=None) 
args = parser.parse_args()

if __name__ == "__main__":

    json_files = find_json_files(args.data_path)
    # print(f"找到 {len(json_files)} 个JSON文件:")
    if len(json_files) == 0:
        exit(0)

    print(f'{args.data_path}')
    print('|file_name|dataset|data_num|epoch|prefix|lr|correct_count|question_count|accuracy|')
    print('|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|')
    eval_list = []
    for file in json_files:
        test_model = args.model.lower()
        test_data = args.dataset.lower()
        
        # if 'constraint'in file.lower() and '^' not in file.lower():
        #     continue
        
        if test_model in file.lower() and test_data in file.lower() and "config" not in file.lower() and "eval" in file.lower():
            # print(file)
            train_config = parse_training_config(file)
            eval_data(file, train_config, args.eval_num)
            # try:
            #     eval_data(file, train_config, args.eval_num)
            # except Exception as e:
            #     print(f"Error processing file {file}: {e}")
            #     continue
    
    print('\n---------------------------------------------------------------------------\n')