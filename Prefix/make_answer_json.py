import json
import os
data_name = "gsm8k"
# data_name = "AQuA"

def add_index_to_json(json_data):
    for idx, item in enumerate(json_data):  # 遍历列表，同时获取索引
        item['index'] = idx  # 添加 index 属性
    return json_data

def save_json_to_file(json_data, file_path):
    with open(file_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"JSON data saved to {file_path}")

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

def make_answer(data_name, data_path):
    file_path = f"dataset/{data_name}/test.json"
    if not os.path.exists(file_path):
        file_path = f"/mnt/userdata/MyProject/Prefix/dataset/{data_name}/test.json"
    data = read_json_file(file_path) 

    data_dict = {}
    if data[0].get('index') is None:
        updated_json_data = add_index_to_json(data)
        save_json_to_file(updated_json_data, file_path)

    for one_data in data:
        Question = one_data['instruction']
        Output = one_data['output']
        Answer = one_data['answer']
        Index = one_data['index']
        
        data_dict[Index] = {'Question':Question, 'Output':Output, 'Answer':Answer}
    
    save_list_to_json(data_dict, data_path)


# save_list_to_json(data_dict, f'/mnt/userdata/MyProject/lofit/finetuned_checkpoints/math10k/Meta-Llama-3-8B-Instruct_math10k_lofit_seed42/{data_name}_eval.json')
# make_answer(data_dict, f'Results/Llama/9000_ffn_up_2e-05/template1/{data_name}_eval.json')
# save_list_to_majson(modify, 'dataset/modify.json')

