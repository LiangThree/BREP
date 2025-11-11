import sys
import os
import pickle
import numpy as np
import json
import pdb
from tqdm import *
import random
import json, time, random
from openai import OpenAI
import time

api_key = "sk-8826f002883c4101b058aa9332b00977"

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

def use_api(question):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    input_prompt = """
    For the following question, provide the key values in the question, Replace the numbers with [NUM], and give the corresponding value.  Please note that only the variables given in the question need to be provided, and there is no need to answer the variables in the reasoning process. Answer me in json!

    For example:
    Mr Boarden is remodeling his bathroom. For every square foot, he needs 24 mosaic tiles. How many mosaic tiles would Mr Boarden need to cover two thirds of his 36 sq ft bathroom?

    {
        "Mr Boarden needs [NUM] mosaic tiles every square foot": "24",
        "Mr Boarden's bathroom is [NUM] sq": "36"
    }

    Question:
    [question]
    """

    input_prompt = input_prompt.replace("[question]", question)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": input_prompt},
        ],
        response_format={
            'type': 'json_object'
        },
        stream=False,
        temperature=0.5,
    )

    output = response.choices[0].message.content.strip()
    output = json.loads(output)

    return output


if __name__ == "__main__":

    # time.sleep(21600)
    
    data = read_json_file("dataset/math10k/train.json")
    output_path = 'Truthful/faithful_results/math10k_faithful_data.json'
    
    if os.path.exists(output_path):
        faithful_data = read_json_file(output_path)
    else:
        faithful_data = []
    
    data_start = len(faithful_data)

    for one_data in tqdm(data[data_start:5000]):
        question = one_data['instruction']

        try:
            output = use_api(question)
            one_data['num_data'] = output
            faithful_data.append(one_data)
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
        
    save_list_to_json(faithful_data, output_path)