import os
import sys
import fire
cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-1])
sys.path.append(main_dir)
import heapq
import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pickle
from functools import partial
import pdb
import json
from datasets import load_dataset
from baukit import Trace, TraceDict
from model import ActivationLLama
import glob
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SEED=42
MAX_SEQ_LEN=128
BATCH_SIZE = 1


def save_list_to_json(data_list, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)  # 使用 indent 格式化 JSON
    print(f"数据已成功存储到 {file_path}")


def load_and_format_data(task):
    """加载TruthfulQA数据集并格式化为问答对"""
    
    # 读取数据
    if task == 'Truthful':
        data_path = "dataset/TruthfulQA/truthful_qa.jsonl"
    elif task == 'Faithful':
        data_path = "dataset/Faithful/faithful.jsonl"
    else:
        print('Task error!')
        exit(0)

    dataset = load_dataset("json", data_files=data_path)["train"]

    # 转换为 Dataset 对象
    print(dataset[0])
    
    formatted_data = []
    for item in dataset:
        # 处理正确答案
        for ans in item['correct_answers']:
            formatted_data.append({
                'prompt': format_truthfulqa(item['question'], ans),
                'label': 1
            })
        
        # 处理错误答案
        for ans in item['incorrect_answers']:
            formatted_data.append({
                'prompt': format_truthfulqa(item['question'], ans),
                'label': 0
            })
    
    return formatted_data

def format_truthfulqa(question, answer):
    return f"Q: {question}\nA: {answer}"

class ReftActivationExtractor:
    def __init__(self, model_name, prefix_path, prefix, start_layer, end_layer):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token_id = (0)
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map="auto")
        
        if "ffn_up" in prefix_path:
            op_position = "ffn_up"
        elif "ffn_down" in prefix_path:
            op_position = "ffn_down"
        elif "attn_o" in prefix_path:
            op_position = "attn_o"
        
        self.model = ActivationLLama(self.model, op_position=op_position, layer_type="all", prefix=prefix)
        self.model.load_model(prefix_path)
        self.num_layers = self.model.base_model.model.config.num_hidden_layers
        self.valid_layers = list(range(start_layer, end_layer))
        self.model.eval()
        
    def get_activations(self, prompts):
        """批量提取各层的注意力头激活"""
        all_layer_activations = []

        batch_size = BATCH_SIZE
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            
            # 编码文本
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                truncation=True,
                max_length=MAX_SEQ_LEN, 
                padding="max_length"
            ).to('cuda')

            # 提取激活
            with torch.no_grad():
                with TraceDict(self.model, [f"base_model.model.layers.{i}" for i in self.valid_layers]) as td:
                    outputs = self.model(**inputs)
                    
            # 收集各层各头的激活
            batch_activations = []
            for layer in self.valid_layers:
                layer_activations = td[f"base_model.model.layers.{layer}"].output[0]
                batch_activations.append(layer_activations.to(torch.float16).cpu().numpy()) # (8, 64, 4096) [batch, seq_len, features]
                # batch_activations.append(layer_activations.cpu().numpy()) # (8, 64, 4096) [batch, seq_len, features]
            
            batch_activations = np.stack(batch_activations) # 拼接: [layers, batch, seq_len, features]
            batch_activations = np.transpose(batch_activations, (1, 0, 2, 3))  # [batch, layers, seq, features]
            all_layer_activations.append(batch_activations) # [data_num, batch, layers, seq_len, features]
        
        # 合并批次: [data_num, layers, seq_len, features]
        del self.model
        return np.concatenate(all_layer_activations, axis=0), self.valid_layers

class ActivationExtractor:
    def __init__(self, model_name, start_layer, end_layer):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token_id = (0)
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map="auto").to('cuda')
        self.num_layers = self.model.config.num_hidden_layers
        self.valid_layers = list(range(start_layer, end_layer))
        self.model.eval()
        
    def get_activations(self, prompts):
        
        all_layer_activations = []

        batch_size=BATCH_SIZE
        for i in tqdm(range(0, len(prompts), batch_size), desc='Get activations'):
            batch_prompts = prompts[i:i+batch_size]
            
            # 编码文本
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                truncation=True,
                max_length=MAX_SEQ_LEN, 
                padding="max_length"
            ).to('cuda')

            # 提取激活
            with torch.no_grad():
                with TraceDict(self.model, [f"model.layers.{i}" for i in self.valid_layers]) as td:
                    outputs = self.model(**inputs)
            
            batch_activations = []
            for layer in self.valid_layers:
                layer_activations = td[f"model.layers.{layer}"].output[0]
                batch_activations.append(layer_activations.to(torch.float16).cpu().numpy()) # (8, 64, 4096) [batch, seq_len, features]
                # batch_activations.append(layer_activations.cpu().numpy()) 
            
            
            batch_activations = np.stack(batch_activations) # 拼接: [layers, batch, seq_len, features]
            batch_activations = np.transpose(batch_activations, (1, 0, 2, 3))  # [batch, layers, seq, features]
            all_layer_activations.append(batch_activations) # [data_num, batch, layers, seq_len, features]
        
        # 合并批次: [data_num, layers, seq_len, features]
        del self.model
        return np.concatenate(all_layer_activations, axis=0), self.valid_layers

class ProbeTrainer:
    def __init__(self, all_layers):
        self.all_layers = all_layers
        
    def train_probes(self, activations, labels, train_ratio=0.8):
        np.random.seed(SEED)
        
        # 数据分割
        indices = np.random.permutation(len(labels))
        split_idx = int(len(indices) * train_ratio)
        
        # 初始化存储
        probes = {}
        accuracies = {}
        
        for layer_idx, layer in tqdm(enumerate(self.all_layers), desc='Train'):
            # 提取特定头的激活特征
            head_features = activations[:, layer_idx, -1, :]  # 取最后一个token的激活
            
            # 分割数据集
            X_train = head_features[indices[:split_idx]]
            y_train = labels[indices[:split_idx]]
            X_val = head_features[indices[split_idx:]]
            y_val = labels[indices[split_idx:]]
            
            # 训练逻辑回归
            clf = LogisticRegression(max_iter=1000, random_state=SEED)
            clf.fit(X_train, y_train)
            
            # 评估并存储
            val_acc = clf.score(X_val, y_val)
            probes[(layer)] = clf
            accuracies[(layer)] = val_acc
                
        return probes, accuracies


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert to Python list
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    else:
        return obj


def merge_all_results(output_path, file_name):
    
    base_files = glob.glob(f"{output_path}/{file_name}")
    
    merged_data = {}
    merged_data['probes'] = {}
    merged_data['accuracies'] = {}

    for f in base_files:
        with open(f, "rb") as pf:
            data = pickle.load(pf)
            for layer, probe in data['probes'].items():
                merged_data['probes'][layer] = probe
            for layer, acc in data['accuracies'].items():
                merged_data['accuracies'][layer] = round(acc,3)
    
    if 'base' in file_name:
        final_path = f"{output_path}/base_truthfulqa_probes.pkl"
    elif 'prefix' in file_name:
        final_path = f"{output_path}/prefix_truthfulqa_probes.pkl"
    elif 'reft' in file_name:
        final_path = f"{output_path}/reft_truthfulqa_probes.pkl"

    with open(final_path, "wb") as f:
        pickle.dump(merged_data, f)
        
    print(f"merge results in  {final_path}")

def train_prob(task, model_name, output_path, prefix_path, reft_path):
    
    MODEL_NAME=f"/mnt/usercache/huggingface/{model_name}"  
    if not os.path.exists(MODEL_NAME):
        MODEL_NAME=f"/mnt/publiccache/huggingface/{model_name}"
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print("\nLoading data...")
    data = load_and_format_data(task)
    data = data[:3000]

    prompts = [d['prompt'] for d in data]
    labels = np.array([d['label'] for d in data])

    layer_divide = [(0,8),(8,16),(16,24),(24,32)]

    base_prob = True
    acc_dict = {}
    # ------------------- base prob -------------------------------
    if base_prob:
        print("\nExtracting base activations...")
        print("Training probes...")
        for current_layer in layer_divide:
            print('current_layer:', current_layer)
            start_layer = current_layer[0]
            end_layer = current_layer[1]
            layer_suffix = f"{start_layer}_{end_layer}"
            extractor_base = ActivationExtractor(model_name=MODEL_NAME, start_layer=start_layer, end_layer=end_layer)
            activations, layers = extractor_base.get_activations(prompts)  # [data_num, layers, seq_len, features]

            trainer = ProbeTrainer(layers)
            probes, acc_matrix = trainer.train_probes(activations, labels)
            
            with open(f"{output_path}/base_probes_{layer_suffix}.pkl", "wb") as f:
                pickle.dump({
                    'probes': probes,
                    'accuracies': acc_matrix,
                    'layer_head_mapping': (layers)
                }, f)
            
            torch.cuda.empty_cache()
        
        merge_all_results(output_path, 'base_probes_*.pkl')
        
        with open(f"{output_path}/base_truthfulqa_probes.pkl", "rb") as f:
            probe_data = pickle.load(f)
        base_accuracies = probe_data['accuracies']
        base_top_k_layer =  [i for i, _ in heapq.nlargest(10, enumerate(base_accuracies), key=lambda x: x[1])]

        print(f'base acc: {base_accuracies}')
        print(f'base top k: {base_top_k_layer}')
        acc_dict['base acc'] = convert_numpy(base_accuracies)
        acc_dict['base top k'] = base_top_k_layer
        torch.cuda.empty_cache()

    # ------------------- prefix prob -------------------------------
    if prefix_path != "":
        print("\nExtracting prefix activations...")
        print("Training probes...")
        if not os.path.isfile(prefix_path):
            prefix_path = os.path.join(prefix_path, 'delta_vector.pth')
        
        for current_layer in layer_divide:
            
            print('current_layer:', current_layer)
            start_layer = current_layer[0]
            end_layer = current_layer[1]
            layer_suffix = f"{start_layer}_{end_layer}"
            
            extractor_prefix = ReftActivationExtractor(model_name=MODEL_NAME,
                                                    start_layer=start_layer, 
                                                    end_layer=end_layer,
                                                    prefix_path=prefix_path, 
                                                    prefix=8)
            prefix_activations, layers = extractor_prefix.get_activations(prompts)  # [data_num, layers, seq_len, features]

            trainer = ProbeTrainer(layers)
            probes, acc_matrix = trainer.train_probes(prefix_activations, labels)
            
            with open(f"{output_path}/prefix_probes_{layer_suffix}.pkl", "wb") as f:
                pickle.dump({
                    'probes': probes,
                    'accuracies': acc_matrix,
                    'layer_head_mapping': (layers)
                }, f)
            
            torch.cuda.empty_cache()
        
        merge_all_results(output_path, 'prefix_probes_*.pkl')
        
        with open(f"{output_path}/prefix_truthfulqa_probes.pkl", "rb") as f:
            probe_data = pickle.load(f)
        prefix_accuracies = probe_data['accuracies']
        prefix_top_k_layer =  [i for i, _ in heapq.nlargest(10, enumerate(prefix_accuracies), key=lambda x: x[1])]

        print(f'prefix acc: {prefix_accuracies}')
        print(f'prefix top k: {prefix_top_k_layer}')
        acc_dict['prefix acc'] = convert_numpy(prefix_accuracies)
        acc_dict['prefix top k'] = prefix_top_k_layer
        torch.cuda.empty_cache()

    # ------------------- reft prob -------------------------------
    if reft_path != "":
        if not os.path.isfile(reft_path):
            reft_path = os.path.join(reft_path, 'delta_vector.pth')
        
        print("\nExtracting reft activations...")
        print("Training probes...")

        for current_layer in layer_divide:
            
            print('current_layer:', current_layer)
            start_layer = current_layer[0]
            end_layer = current_layer[1]
            layer_suffix = f"{start_layer}_{end_layer}"

            extractor_reft = ReftActivationExtractor(
                model_name=MODEL_NAME, 
                start_layer=start_layer, 
                end_layer=end_layer,
                prefix_path=reft_path, 
                prefix=-1)
            reft_activations, layers = extractor_reft.get_activations(prompts)
            
            trainer = ProbeTrainer(layers)
            probes, acc_matrix = trainer.train_probes(reft_activations, labels)
            
            with open(f"{output_path}/reft_probes_{layer_suffix}.pkl", "wb") as f:
                pickle.dump({
                    'probes': probes,
                    'accuracies': acc_matrix,
                    'layer_head_mapping': (layers)
                }, f)
            torch.cuda.empty_cache()
            
        merge_all_results(output_path, 'reft_probes_*.pkl')
        
        with open(f"{output_path}/reft_truthfulqa_probes.pkl", "rb") as f:
            probe_data = pickle.load(f)
        reft_accuracies = probe_data['accuracies']
        reft_top_k_layer =  [i for i, _ in heapq.nlargest(10, enumerate(reft_accuracies), key=lambda x: x[1])]

        print(f'reft acc: {reft_accuracies}')
        print(f'reft top k: {reft_top_k_layer}')
        acc_dict['reft acc'] = convert_numpy(reft_accuracies)
        acc_dict['reft top k'] = reft_top_k_layer
        
        torch.cuda.empty_cache()

    # avg_acc = [(x+y+z)/3 for x, y, z in zip(base_accuracies, prefix_accuracies, reft_accuracies)]
    # avg_top_k_layer =  [i for i, _ in heapq.nlargest(10, enumerate(avg_acc), key=lambda x: x[1])]
    # print(f'sum acc: {avg_acc}')
    # print(f'sum top k: {avg_top_k_layer}')

    save_path = os.path.join(output_path, 'prob_acc.json')
    save_list_to_json(acc_dict, save_path)




if __name__ == "__main__":
    fire.Fire(train_prob)