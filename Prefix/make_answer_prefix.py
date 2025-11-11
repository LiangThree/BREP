import json
import os
import pdb
from collections import defaultdict
import re
data_name = "gsm8k"

def save_list_to_json(data_list, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)  # 使用 indent 格式化 JSON
    print(f"数据已成功存储到 {file_path}")

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
    
    return text[indexes[0] + length + 2 : ]

def tokenize(text):
    return re.findall(r"[\w\-]+|['\.,!?;:\"’“”]|[\U0001F600-\U0001F64F]", text)

def get_first_k_tokens(tokens, k):
    return tokens[:k] if k >= 0 else tokens[:]

def make_answer_prefix(read_path, save_path, prefix):
    
    data = read_json_file(read_path)
    data_dict = defaultdict()
    
    for key in data.keys():
        one_data = data[key]
        Question = one_data['Question']
        Output = one_data['Output']
        Answer = one_data['Answer']
        Index = key

        answer_prefix = extract_between_assistants(one_data['base'], keyword="assistant")
        token_list = tokenize(answer_prefix)
        result = get_first_k_tokens(token_list, prefix)
        answer_prefix = ' '.join(result)
        data_dict[Index] = {'Question':Question+answer_prefix, 'Output':Output, 'Answer':Answer}
    
    save_list_to_json(data_dict, save_path)

if __name__ == "__main__":
    prefix = 8
    make_answer_prefix('Results/Prefix/Llama-3-8B/5000_math10k_all_2e-4_RED/ffn_up/llama_gsm8k_eval.json',
                f'Results/Prefix/Llama-3-8B/5000_math10k_all_2e-4_RED/ffn_up/llama_gsm8k_eval_base{prefix}.json',prefix=prefix)