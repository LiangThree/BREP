import json
import matplotlib.pyplot as plt
import os
import sys
import fire
import pickle
import heapq
import pdb
import numpy as np

cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-1])
sys.path.append(main_dir)

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

def deal_prob(prob):
    questions = prob.keys()
    layer_sum = [0] * 32
    layers = [f'layer_{i}' for i in range(32)] 
    for layer_idx, layer in enumerate(layers):
        truth_sum = 0
        for question in questions:
            tokens = prob[question][layer].keys()
            for token in tokens:
                truth_sum += prob[question][layer][token]
        layer_sum[layer_idx] = truth_sum/(len(questions)*len(tokens))
        print(f'{round(layer_sum[layer_idx],2)}|', end='')

def draw_heatmap(output_path="Truthful/faithful_results/llama3_ffn_down"):
    base_prob =  read_json_file(f"{output_path}/truthful_probe_base.json")
    reft_prob =  read_json_file(f"{output_path}/truthful_probe_reft.json")
    prefix_prob =  read_json_file(f"{output_path}/truthful_probe_prefix.json")

    for index, prob in enumerate([base_prob, reft_prob, prefix_prob]):
        print(f"Processing ...")
        print('|0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|')
        print('|',end='')
        deal_prob(prob)
        print("\n")
        
            

if __name__ == "__main__":
    fire.Fire(draw_heatmap)