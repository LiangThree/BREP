import random
from collections import defaultdict
from typing import Dict, List
import blobfile as bf
import gzip
import orjson
import json
import numpy as np
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple
import pdb
from tqdm import *

Sample = Dict[str, Any]

def json_loads(s: str) -> Dict:
    try:
        return orjson.loads(s)
    except Exception:
        return json.loads(s)  # fallback

def open_jsonl(file: str):
    if file.endswith(".gz"):
        return gzip.open(bf.BlobFile(file, "rb"))
    return bf.BlobFile(file, "r")

def _read_jsonl(file: str) -> List[Dict]:
    with open_jsonl(file) as f:
        return [json_loads(l) for l in f.readlines() if l]

def _key_by_problem(samples: List[Dict]):
    grouped_samples = defaultdict(list)
    for sample in samples:
        grouped_samples[sample["problem"]].append(sample)
    return grouped_samples

def _get_answer(sample: Sample) -> Optional[str]:
    return sample.get("answer", sample.get("given_answer", None))

def _choose_sample_by_score(samples: List[Sample], key: str) -> Optional[Sample]:
    if len(samples) == 0:
        return None
    return max(samples, key=lambda x: x[key])

def save_list_to_json(data_list, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)  # 使用 indent 格式化 JSON
    print(f"数据已成功存储到 {file_path}")

data = _read_jsonl('prm800k/phase2_train.jsonl')
data_list = []

for index, one_data in tqdm(enumerate(data)):
    
    instruction = one_data['question']['problem']
    asnwer = one_data['question']['ground_truth_answer']
    output = one_data['question']['ground_truth_solution']
    
    one_data_dict = {
        'instruction': instruction,
        'answer': asnwer,
        'output': output
    }

    data_list.append(one_data_dict)

save_list_to_json(data_list, 'prm800k/train.json')


"""
phase1_train.jsonl
data_num:949
dict_keys(['labeler', 'timestamp', 'generation', 'is_quality_control_question','is_initial_screening_question', 'question', 'label'])

Question - data[0]['question']: ['problem', 'ground_truth_answer']
Steps - data[0]['label']: ['steps'( data[0]['label']['steps'][0] list存储steps), 'total_time', 'finish_reason']

phase2_train.jsonl
data_num: 97782
dict_keys(['labeler', 'timestamp', 'generation', 'is_quality_control_question', 'is_initial_screening_question', 'question', 'label'])

Question - data[0]['question']: 
dict_keys(['problem', 'ground_truth_solution', 'ground_truth_answer', 'pre_generated_steps', 'pre_generated_answer', 'pre_generated_verifier_score'])
"""
