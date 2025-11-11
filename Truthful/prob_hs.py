import sys
import os
import pickle
import numpy as np

cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-1])
sys.path.append(main_dir)
from template import *
from datasets import load_dataset
from tqdm import tqdm 
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
import json
from transformers import GenerationConfig,AutoConfig
import fire
from model import ActivationLLama
from datasets import load_dataset, Dataset
from typing import Optional, Union
import pdb 
from Prefix.make_answer_json import make_answer
from collections import defaultdict

MAX_SEQ_LEN = 512
BATCH_SIZE = 64

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

def save_params_to_json(output_path: str, **kwargs):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kwargs, f, indent=4, ensure_ascii=False)

def main(
    model_path: str = "",
    data_num: int = None,
    peft:bool = False,
    n_prefix: int = None,
    start: int = -1,
    end: int = -1,
    peft_path:str="",
    is_train_return:bool = True,
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.2,
    template_index: int = None,
    dataset: str="",
    prob_layers: list = None,
    output_path: str="",
):
    
    data_path = os.path.join(peft_path, f'{dataset}_eval.json')
    
    print("----------------------- template -----------------------")
    print(prompt_template[template_index])
    print("--------------------------------------------------------")

    peft_path_list = peft_path.split('/')
    config_path = peft_path_list[3]
    prefix = n_prefix
    print('prefix:', prefix)

    config_path = os.path.join(peft_path, 'eval_config.json')
    peft_path = os.path.join(peft_path, 'delta_vector.pth')

    if os.path.exists(config_path):
        print(f'{config_path} exists')
    
    if os.path.exists(peft_path):
        print(f'{peft_path} exists')

    if "ffn_up" in peft_path:
        op_position = "ffn_up"
    elif "ffn_down" in peft_path:
        op_position = "ffn_down"
    elif "attn_o" in peft_path:
        op_position = "attn_o"

    eval_config = {
        "model_path": model_path,
        "data_path": data_path,
        "data_num": data_num,
        "op_position": op_position,
        "peft": peft,
        "start": start,
        "end": end,
        "peft_path": peft_path,
        "is_train_return": is_train_return,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "repetition_penalty": repetition_penalty,
        "template_index": template_index
    }
    # save_params_to_json(config_path, **eval_config)
    print(eval_config)

    def process_alpacaeval(example):
        question = example['Question']
        prompt = prompt_template[template_index] % question
        example["prompt"] = prompt
        return example
    
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="auto")
    model = ActivationLLama(model, op_position=op_position, prefix=prefix)
    model.load_model(peft_path)

    print(model)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", torch_dtype=torch.bfloat16)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    if not os.path.exists(data_path):
        make_answer(dataset, data_path)

    dataset = load_custom_dataset(data_path, data_num)
    dataset = dataset.map(process_alpacaeval)

    generate_list = []
    
    generation_config = GenerationConfig(
            do_sample=False,
            no_repeat_ngram_size=no_repeat_ngram_size if no_repeat_ngram_size > 0 else 0,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # pad_token_id=0,
            # eos_token_id=2,
            bos_token_id=1,
        )
    
    answer_dict = read_json_file(data_path)

    # A100 * 1: batch_size=128
    batch_size = BATCH_SIZE
    all_prompts = [ex["prompt"] for ex in dataset]
    total_samples = len(all_prompts)

    hidden_activations = defaultdict(list)
    
    """
    hidden_activations[i][j]: i layer j tokens (shape: [batch_size, seq_len, hidden_size])
    """

    # 定义前向钩子
    def hook_fn(module, input, output, layer_idx):
        # 保存输出的 hidden states（去掉批次维度）
        hidden_activations[layer_idx].append(output[0].detach().cpu())

    # 注册钩到每个 o_proj 层
    for idx, layer in enumerate(model.base_model.model.layers):
        layer.register_forward_hook(
            lambda module, input, output, idx=idx: hook_fn(module, input, output, idx)
        )

    if n_prefix == 0:
        with open(f"{output_path}/base_truthfulqa_probes.pkl", "rb") as f:
            probe_data = pickle.load(f)
    elif n_prefix == -1:
        with open(f"{output_path}/reft_truthfulqa_probes.pkl", "rb") as f:
            probe_data = pickle.load(f)
    else:
        with open(f"{output_path}/prefix_truthfulqa_probes.pkl", "rb") as f:
            probe_data = pickle.load(f)

    probes = probe_data['probes']
    accuracies = probe_data['accuracies']
    # top_k_layer =  [i for i, _ in heapq.nlargest(5, enumerate(accuracies), key=lambda x: x[1])]
    top_k_layer =  prob_layers
    truthful_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for batch_idx in tqdm(range(0, total_samples, batch_size)):
        
        hidden_activations.clear()
        batch_prompts = all_prompts[batch_idx:batch_idx + batch_size]
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_SEQ_LEN,
            generation_config=generation_config
        )
        
        completions = tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        
        for i, (prompt, completion) in enumerate(zip(batch_prompts, completions)):
            global_idx = batch_idx + i
            answer_dict[str(global_idx)][op_position] = completion
        
        for layer in top_k_layer:
            current_layer_hs = hidden_activations[layer]
            for index, current_hs in enumerate(current_layer_hs):
                for i, one_batch_data in enumerate(current_hs):
                    probe_global_idx = batch_idx + i
                    last_token_activation = one_batch_data[-1, :]
                    last_token_activation = last_token_activation.reshape(1, -1).cpu().float().numpy()
                    pred = probes[layer].predict(last_token_activation)

                    if isinstance(pred, np.ndarray):
                        pred = pred.tolist()
                    pred = pred[0]

                    truthful_dict[str(probe_global_idx)][f"layer_{layer}"][f"token_{index}"] = pred

    if n_prefix == 0:
        save_list_to_json(truthful_dict, f"{output_path}/truthful_probe_base.json")
    elif n_prefix == -1:
        save_list_to_json(truthful_dict, f"{output_path}/truthful_probe_reft.json")
    else:
        save_list_to_json(truthful_dict, f"{output_path}/truthful_probe_prefix.json")


def get_hs(model_name, output_path, prefix_path, reft_path, layer_type):

    if 'llama' in model_name.lower():
        prob_layers = [i for i in range(32)]
        template_index="llama3"
    elif 'qwen' in model_name.lower():
        prob_layers = [i for i in range(28)]
        template_index="qwen_base"
    elif 'mistral' in model_name.lower():
        prob_layers = [i for i in range(32)]
        template_index="mistral"
    
    model_path=f"/mnt/usercache/huggingface/{model_name}"  
    if not os.path.exists(model_path):
        model_path=f"/mnt/publiccache/huggingface/{model_name}"

    data_num=500
    peft="RED"
    start=-1
    end=-1
    is_train_return=True
    no_repeat_ngram_size=0
    repetition_penalty=1.2
    
    dataset="gsm8k"

    print('------------------------- get base hidden states -------------------------')
    
    n_prefix=0
    main(model_path=model_path,
        data_num=data_num,
        peft=peft,
        n_prefix=n_prefix,
        start=start,
        end=end,
        peft_path=reft_path,
        is_train_return=is_train_return,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        template_index=template_index,
        dataset=dataset,
        prob_layers=prob_layers,
        output_path=output_path)
    
    print('------------------------- get prefix hidden states -------------------------')
    
    n_prefix=8
    main(model_path=model_path,
        data_num=data_num,
        peft=peft,
        n_prefix=n_prefix,
        start=start,
        end=end,
        peft_path=prefix_path,
        is_train_return=is_train_return,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        template_index=template_index,
        dataset=dataset,
        prob_layers=prob_layers,
        output_path=output_path)

    print('------------------------- get reft hidden states -------------------------')
    
    n_prefix=-1
    main(model_path=model_path,
        data_num=data_num,
        peft=peft,
        n_prefix=n_prefix,
        start=start,
        end=end,
        peft_path=reft_path,
        is_train_return=is_train_return,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        template_index=template_index,
        dataset=dataset,
        prob_layers=prob_layers,
        output_path=output_path)

if __name__ == "__main__":
    fire.Fire(get_hs)