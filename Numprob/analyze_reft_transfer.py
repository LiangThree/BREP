import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-1])
sys.path.append(main_dir)
from template import *
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
from collections import defaultdict
from numpy import linalg as LA

os.environ["CUDA_VISIBLE_DEVICES"]="3"

MODEL_NAME="/mnt/usercache/huggingface/Meta-Llama-3-8B-Instruct"  
if not os.path.exists(MODEL_NAME):
    MODEL_NAME="/mnt/publiccache/huggingface/Meta-Llama-3-8B-Instruct"  
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
NUM_LAYERS=32  # LLaMA-7B的层数
NUM_HEADS=32   # 每层的注意力头数
SEED=42
MAX_SEQ_LEN=256

def load_and_format_data():
    """加载TruthfulQA数据集并格式化为问答对"""
    
    data_path = "dataset/gsm8k/test.json"
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    return data


class ReftActivationExtractor:
    def __init__(self, model_name, reft_path, prefix):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token_id = (0)
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(DEVICE)
        self.model = ActivationLLama(self.model, op_position="ffn_up", layer_type="all", prefix=prefix)
        self.model.load_model(reft_path)
        self.model.eval()
        
    def get_activations(self, prompts):
        """批量提取各层的注意力头激活"""
        all_layer_activations = []

        batch_size = 8
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            
            # 编码文本
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                truncation=True,
                max_length=MAX_SEQ_LEN, 
                padding="max_length"
            ).to(DEVICE)

            # 提取激活
            with torch.no_grad():
                with TraceDict(self.model, [f"base_model.model.layers.{i}" for i in range(NUM_LAYERS)]) as td:
                    outputs = self.model(**inputs)
                    
            # 收集各层各头的激活
            batch_activations = []
            for layer in range(NUM_LAYERS):
                layer_activations = td[f"base_model.model.layers.{layer}"].output[0]
                batch_activations.append(layer_activations.to(torch.float32).cpu().numpy()) # (8, 64, 4096) [batch, seq_len, features]
                # batch_activations.append(layer_activations.cpu().numpy()) # (8, 64, 4096) [batch, seq_len, features]
            
            batch_activations = np.stack(batch_activations) # 拼接: [layers, batch, seq_len, features]
            batch_activations = np.transpose(batch_activations, (1, 0, 2, 3))  # [batch, layers, seq, features]
            all_layer_activations.append(batch_activations) # [data_num, batch, layers, seq_len, features]
        
        # 合并批次: [data_num, layers, seq_len, features]
        del self.model
        return np.concatenate(all_layer_activations, axis=0)

class ActivationExtractor:
    def __init__(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token_id = (0)
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(DEVICE)
        self.model.eval()
        
    def get_activations(self, prompts):
        
        all_layer_activations = []

        batch_size=16
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            
            # 编码文本
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                truncation=True,
                max_length=MAX_SEQ_LEN, 
                padding="max_length"
            ).to(DEVICE)

            # 提取激活
            with torch.no_grad():
                with TraceDict(self.model, [f"model.layers.{i}" for i in range(NUM_LAYERS)]) as td:
                    outputs = self.model(**inputs)
            
            batch_activations = []
            for layer in range(NUM_LAYERS):
                layer_activations = td[f"model.layers.{layer}"].output[0]
                batch_activations.append(layer_activations.to(torch.float32).cpu().numpy()) # (8, 64, 4096) [batch, seq_len, features]
                # batch_activations.append(layer_activations.cpu().numpy()) 
            
            
            batch_activations = np.stack(batch_activations) # 拼接: [layers, batch, seq_len, features]
            batch_activations = np.transpose(batch_activations, (1, 0, 2, 3))  # [batch, layers, seq, features]
            all_layer_activations.append(batch_activations) # [data_num, batch, layers, seq_len, features]
        
        # 合并批次: [data_num, layers, seq_len, features]
        del self.model
        return np.concatenate(all_layer_activations, axis=0)


if __name__ == "__main__":
    print("\nLoading data...")
    data = load_and_format_data()
    data = data[:100]  

    template = prompt_template['llama3']
    prompts = [template % d['instruction'] for d in data]
    labels = [d['answer'] for d in data]

    base_prob = True
    reft_prob = True

    # ------------------- base prob -------------------------------
    if base_prob:
        print("\nExtracting base activations...")

        extractor_base = ActivationExtractor(MODEL_NAME)
        activations = extractor_base.get_activations(prompts)  # [data_num, layers, seq_len, features]

    # ------------------- reft prob -------------------------------
    if reft_prob:
        print("\nExtracting reft activations...") # Prefix/Results/RED/Llama3/9000_math10k_all_2e-4/ffn_down

        extractor_reft = ReftActivationExtractor(model_name=MODEL_NAME, reft_path="Results/RED/Llama3/9000_math10k_all_2e-4/ffn_down/delta_vector.pth", prefix=-1)
        reft_activations = extractor_reft.get_activations(prompts)
    
    
    modify_degree = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for data_index in range(activations.shape[0]):
        for layer_index in range(activations.shape[1]):

            prob_path = f"/mnt/userdata/liangsirui/MyProject/NumProbe/model/llama3_instruct/ridge/layer_{layer_index}_prober_for_golden.bin"
            with open(prob_path, 'rb') as fin:
                g_prober = pickle.load(fin)
            G = g_prober[-1].coef_ / LA.norm(g_prober[-1].coef_)

            for token_index in range(activations.shape[2]):
                hs_vec = activations[data_index, layer_index, token_index] - reft_activations[data_index, layer_index, token_index]
                dot_product = np.dot(hs_vec, G)
                # if abs(dot_product) > 0.05:
                #     modify_degree[f'data_{data_index}'][f'layer_{layer_index}'][f'token_{token_index}'] = round(float(dot_product),5)
                modify_degree[f'data_{data_index}'][f'layer_{layer_index}'][f'token_{token_index}'] = round(float(dot_product),5)
    
    with open("Numprob/modify_degree.json", "w", encoding="utf-8") as f:
        json.dump(modify_degree, f, indent=4)
        

    
    
    