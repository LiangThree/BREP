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

def draw_heatmap(output_path="Truthful/prob_results/llama3"):
    base_prob =  read_json_file(f"{output_path}/truthful_probe_base.json")
    reft_prob =  read_json_file(f"{output_path}/truthful_probe_reft.json")
    prefix_prob =  read_json_file(f"{output_path}/truthful_probe_prefix.json")

    # 确定模型类型和层数
    if "qwen" in output_path.lower():
        layer_num = 28
    elif "llama" in output_path.lower() or 'mistral' in output_path.lower():
        layer_num = 32
    else:
        raise ValueError("Unknown model type")

    # 配置热图参数
    token_bins = 32
    models = ['Base', 'ReFT', 'BREP']
    data_map = {
        'Base': base_prob,
        'ReFT': reft_prob,
        'BREP': prefix_prob
    }

    # 创建输出目录
    output_dir = f"{output_path}/prob_heatmaps"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    heatmap_dict = {}

    for model in models:
        # 初始化存储矩阵（层数 x token区间）
        heatmap_data = np.zeros((layer_num, token_bins))
        
        # 遍历每一层
        for layer_idx in range(layer_num):
            layer_key = f"layer_{layer_idx}"
            token_acc = np.zeros(512)  # 原始token位置准确率
            
            # 遍历每个数据点
            for data_index in data_map[model].keys():
                layer_data = data_map[model][data_index][layer_key]
                cumulative_acc = 0
                
                # 计算每个token位置的累计准确率
                for index, token_pos in enumerate(layer_data.keys()):
                    cumulative_acc += layer_data[token_pos]
                    token_acc[index] += cumulative_acc / (index+1)  # 当前位置的平均准确率
            
            # 计算全局平均并分组
            avg_acc = token_acc / len(data_map[model].keys())
            
            # 将512个token位置分成32个区间求均值
            for bin_idx in range(token_bins):
                start = bin_idx * 16
                end = (bin_idx + 1) * 16
                heatmap_data[layer_idx, bin_idx] = np.mean(avg_acc[start:end])
        
        heatmap_dict[model] = heatmap_data
    
    diff_reft = heatmap_dict['ReFT'] - heatmap_dict['Base']
    diff_prefix = heatmap_dict['BREP'] - heatmap_dict['Base']

    diff_models = [
        ('ReFT', diff_reft),
        ('BREP', diff_prefix) 
    ]

    max_val_reft = max(np.abs(diff_models[0][1].max()), np.abs(diff_models[0][1].min()))
    max_val_prefix = max(np.abs(diff_models[1][1].max()), np.abs(diff_models[1][1].min()))
    max_val = max(max_val_reft, max_val_prefix)

    for model_name, diff_matrix in diff_models:
        plt.figure(figsize=(12, 8))
        vmin, vmax = -max_val, max_val
        plt.imshow(diff_matrix, cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
        plt.title(f'Accuracy Difference ({model_name} - Base)', fontsize=22)
        plt.xlabel('Token Position', fontsize=22)
        plt.ylabel('Layer Index', fontsize=22)
        plt.xticks(np.arange(0, token_bins, 4), 
                  [str(i*16) for i in range(0, token_bins, 4)], fontsize=20)
        plt.yticks(np.arange(0, layer_num, 4), 
                  [str(i) for i in range(0, layer_num, 4)], fontsize=20)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20) 
        cbar.set_label('Average Accuracy', rotation=270, labelpad=15, fontsize=20)
        
        # 保存图像
        plt.savefig(f"{output_dir}/{model_name}_heatmap.svg", 
                   bbox_inches='tight', dpi=300)
        plt.savefig(f"{output_dir}/{model_name}_heatmap.pdf", 
                   bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    fire.Fire(draw_heatmap)