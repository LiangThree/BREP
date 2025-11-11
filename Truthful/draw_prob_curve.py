import json
import matplotlib.pyplot as plt
import os
import sys
import fire
import pickle
import heapq
import pdb

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

def get_prob_layers(output_path):

    with open(f"{output_path}/base_truthfulqa_probes.pkl", "rb") as f:
        probe_data = pickle.load(f)
    base_accuracies = probe_data['accuracies']
    base_top_k_layer =  [i for i, _ in heapq.nlargest(10, enumerate(base_accuracies), key=lambda x: x[1])]
    print(f'base acc: {base_accuracies}')
    print(f'base top k: {base_top_k_layer}')

    with open(f"{output_path}/prefix_truthfulqa_probes.pkl", "rb") as f:
        probe_data = pickle.load(f)
    prefix_accuracies = probe_data['accuracies']
    prefix_top_k_layer =  [i for i, _ in heapq.nlargest(10, enumerate(prefix_accuracies), key=lambda x: x[1])]
    print(f'prefix acc: {prefix_accuracies}')
    print(f'prefix top k: {prefix_top_k_layer}')

    with open(f"{output_path}/reft_truthfulqa_probes.pkl", "rb") as f:
        probe_data = pickle.load(f)
    reft_accuracies = probe_data['accuracies']
    reft_top_k_layer =  [i for i, _ in heapq.nlargest(10, enumerate(reft_accuracies), key=lambda x: x[1])]
    print(f'reft acc: {reft_accuracies}')
    print(f'reft top k: {reft_top_k_layer}')


def draw_curve(output_path="Truthful/prob_results/llama3"):
    # 分析get_inference_hs.py的分析结果
    base_prob =  read_json_file(f"{output_path}/truthful_probe_base.json")
    reft_prob =  read_json_file(f"{output_path}/truthful_probe_reft.json")
    prefix_prob =  read_json_file(f"{output_path}/truthful_probe_prefix.json")

    data_length = len(base_prob.keys())
    get_prob_layers(output_path)

    if "qwen" in output_path.lower():
        layer_num = 28
    elif "llama" in output_path.lower() or 'mistral' in output_path.lower():
        layer_num = 32
    
    plt.figure(figsize=(6 * 4, 4 * 8))  # Adjust size as needed
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between subplots

    all_base_layer_acc = [0 for i in range(512)]
    all_reft_layer_acc = [0 for i in range(512)]
    all_prefix_layer_acc = [0 for i in range(512)]

    for layer_index in range(layer_num):
        int_layer_index = int(layer_index)
        layer_index = f"layer_{layer_index}"
        all_base_acc = [0 for i in range(512)]
        all_reft_acc = [0 for i in range(512)]
        all_prefix_acc = [0 for i in range(512)]
        
        for data_index in base_prob.keys():
            base_layer_prob = base_prob[data_index][layer_index]
            reft_layer_prob = reft_prob[data_index][layer_index]
            prefix_layer_prob = prefix_prob[data_index][layer_index]

            base_correct = 0
            reft_correct = 0
            prefix_correct = 0

            for index, token_index in enumerate(base_layer_prob.keys()):
                base_token_prob = base_layer_prob[token_index]
                base_correct += base_token_prob
                current_base_acc = base_correct / (index + 1)
                
                reft_token_prob = reft_layer_prob[token_index]
                reft_correct += reft_token_prob
                current_reft_acc = reft_correct / (index + 1)

                prefix_token_prob = prefix_layer_prob[token_index]
                prefix_correct += prefix_token_prob
                current_prefix_acc = prefix_correct / (index + 1)

                all_base_acc[index] += current_base_acc
                all_reft_acc[index] += current_reft_acc
                all_prefix_acc[index] += current_prefix_acc

        data_length = 512
        avg_base = [acc/data_length for acc in all_base_acc][:data_length]
        avg_reft = [acc/data_length for acc in all_reft_acc][:data_length]
        avg_prefix = [acc/data_length for acc in all_prefix_acc][:data_length]

        all_base_layer_acc = [x+y for x,y in zip(all_base_layer_acc[:data_length],avg_base[:data_length])]
        all_reft_layer_acc = [x+y for x,y in zip(all_reft_layer_acc[:data_length],avg_reft[:data_length])]
        all_prefix_layer_acc = [x+y for x,y in zip(all_prefix_layer_acc[:data_length],avg_prefix[:data_length])]

        # Create subplot
        ax = plt.subplot(8, 4, int_layer_index + 1)
        
        # Grid and spines
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='lightgray')
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Plot lines
        colors = ['#EF4F4F', '#7E5CAD', '#7AA2E3', '#52D3D8', '#FFE699']
        ax.plot(avg_base, label='Base Model', linestyle='-', linewidth=1.5, color=colors[0])
        ax.plot(avg_reft, label='ReFT Model', linestyle='--', linewidth=1.5, color=colors[1])
        ax.plot(avg_prefix, label='Prefix Model', linestyle=':', linewidth=1.5, color=colors[2])

        # Subplot title and labels
        ax.set_title(f'Token-wise Accuracy ({layer_index})', fontsize=10)
        ax.set_xlabel('Token Position', fontsize=8)
        ax.set_ylabel('Average Accuracy', fontsize=8)
        
        # Only show legend on the first subplot to avoid repetition
        if layer_index == 0:
            ax.legend(fontsize=8)
        
        ax.grid(True, alpha=0.4)
        ax.set_xlim(0, data_length)
        ax.set_xticks(range(0, data_length, data_length//8))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(16))

    # Save the combined figure
    if not os.path.exists(f"{output_path}/prob_curve"):
        os.mkdir(f"{output_path}/prob_curve")

    plt.savefig(f"{output_path}/prob_curve/all_layers_comparison.svg", bbox_inches='tight', dpi=300)

    plt.figure(figsize=(6, 4))  # Adjust size as needed
    ax = plt.gca()  # 明确获取当前axes

    all_base_layer_acc = [i/32 for i in all_base_layer_acc]
    all_reft_layer_acc = [i/32 for i in all_reft_layer_acc]
    all_prefix_layer_acc = [i/32 for i in all_prefix_layer_acc]

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='lightgray')
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Plot lines
    colors = ['#EF4F4F', '#7E5CAD', '#7AA2E3', '#52D3D8', '#FFE699']
    ax.plot(all_base_layer_acc, label='Base Model', linestyle='-', linewidth=1, color=colors[0])
    ax.plot(all_reft_layer_acc, label='ReFT Model', linestyle='--', linewidth=1, color=colors[1])
    ax.plot(all_prefix_layer_acc, label='Prefix Model', linestyle=':', linewidth=1, color=colors[2])

    ax.legend(fontsize=8)

    # Subplot title and labels
    ax.set_title(f'Token-wise Accuracy', fontsize=8)
    ax.set_xlabel('Token Position', fontsize=8)
    ax.set_ylabel('Average Accuracy', fontsize=8)

    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, data_length)
    ax.set_xticks(range(0, data_length, data_length//8))

    plt.savefig(f"{output_path}/prob_curve/avg_comparison.svg", bbox_inches='tight', dpi=300)
        

if __name__ == "__main__":
    fire.Fire(draw_curve)