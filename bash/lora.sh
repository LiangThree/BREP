#!/bin/bash

# llama3/Qwen2.5/Qwen-math: pip install transformers==4.37.2 trl==0.7.10 "pydantic<2.0.0" accelerate==0.25.0 -U
# LLama3.1/Qwen3: pip install tokenizers==0.21.1 transformers==4.52.3 trl==0.17.0 torchvision==0.16 protobuf==3.20.3

# Meta-Llama-3-8B-Instruct | Qwen2.5-7B-Instruct | Qwen2.5-Math-7B-Instruct | Qwen3-8B | DeepSeek-R1-Distill-Qwen-7B
# template: llama | qwen_base | qwen_math 

condition=$1

if [ "$condition" == "train" ]; then
    
    # CUDA_VISIBLE_DEVICES=0 python lora/lora_train.py \
    # --model_path "Meta-Llama-3.1-8B-Instruct" \
    # --data_path  "dataset/prm800k/train.json" \
    # --output_dir "Results/Lora/Llama-3.1-8B/" \
    # --learning_rate 1e-6 \
    # --data_num 5000 \
    # --template_index "llama" \
    # --layer 15

    # CUDA_VISIBLE_DEVICES=0 python lora/lora_eval.py \
    # --model_path "Meta-Llama-3-8B-Instruct" \
    # --lora_path "Results/Lora/Llama-3.1-8B/5000_prm800k_lora_1e-06"\
    # --dataset "math500" \
    # --data_num -1 \
    # --template_index "llama"

    # CUDA_VISIBLE_DEVICES=2 python lora/lora_train.py \
    # --model_path "Qwen3-14B" \
    # --data_path  "dataset/prm800k/train.json" \
    # --output_dir "Results/Lora/Qwen3-14B/" \
    # --learning_rate 1e-6 \
    # --data_num 5000 \
    # --template_index "qwen3_nothink" \
    # --layer 15

    # CUDA_VISIBLE_DEVICES=2 python lora/lora_eval.py \
    # --model_path "Qwen3-14B" \
    # --lora_path "Results/Lora/Qwen3-14B/5000_prm800k_lora_1e-06"\
    # --dataset "math500" \
    # --data_num -1 \
    # --template_index "qwen3_nothink" 

    CUDA_VISIBLE_DEVICES=2 python lora/lora_eval.py \
    --model_path "Qwen3-14B" \
    --lora_path "Results/Lora/Qwen3-14B/5000_prm800k_lora_1e-06"\
    --dataset "amc23" \
    --data_num -1 \
    --template_index "qwen3_nothink" 


elif [ "$condition" == "eval" ]; then

    # CUDA_VISIBLE_DEVICES=0 python lora/lora_eval.py \
    # --model_path "Meta-Llama-3.1-8B-Instruct" \
    # --lora_path "Results/Lora/Llama-3.1-8B/5000_math10k_lora_1e-05"\
    # --dataset "gsm8k" \
    # --data_num -1 \
    # --template_index "llama"

    # CUDA_VISIBLE_DEVICES=0 python lora/lora_eval.py \
    # --model_path "Meta-Llama-3.1-8B-Instruct" \
    # --lora_path "Results/Lora/Llama-3.1-8B/5000_math10k_lora_1e-05"\
    # --dataset "svamp" \
    # --data_num -1 \
    # --template_index "llama"

    # CUDA_VISIBLE_DEVICES=0 python lora/lora_eval.py \
    # --model_path "Meta-Llama-3.1-8B-Instruct" \
    # --lora_path "Results/Lora/Llama-3.1-8B/5000_math10k_lora_1e-05"\
    # --dataset "mathqa" \
    # --data_num -1 \
    # --template_index "llama"    

    python lora/lora_eval.py \
    --model_path "Qwen2.5-Math-7B-Instruct" \
    --lora_path "Results/Lora/Qwen2.5-Math-7B/5000_prm800k_lora_1e-05"\
    --dataset "amc23" \
    --data_num -1 \
    --template_index "qwen_math"

fi