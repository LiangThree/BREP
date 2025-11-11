#!/bin/bash
# layer_type: scaling/bias/all 当学习率达到0.01量级时scaling才会发生变化 
# watch --color -n1 gpustat -cpu

# llama3/Qwen2.5/Qwen-math: pip install transformers==4.37.2 trl==0.7.10 "pydantic<2.0.0" accelerate==0.25.0 -U
# LLama3.1/Qwen3: pip install tokenizers==0.21.1 transformers==4.52.3 trl==0.17.0 torchvision==0.16 protobuf==3.20.3 torch==2.1.0

# deepspeed配置
# conda install -c conda-forge cuda-nvcc=11.7   
# conda install cuda-toolkit  
# export CUDA_HOME=/root/conda/envs/python/
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# pip install deepspeed==0.13.1

# Meta-Llama-3-8B-Instruct | Qwen2.5-Math-7B-Instruct | Qwen3-8B | DeepSeek-R1-Distill-Qwen-7B
# template: llama3 | qwen_base | qwen_base_fewshot | mistral | none

# n_prefix 0:原始模型不调整 -1:reft模型所有位置均调整 n:调整问题及前n个token
# 训练时n_prefix内置为-1

condition=$1

if [ "$condition" == "red" ]; then

    CUDA_VISIBLE_DEVICES=1 python Prefix/red_train.py \
    --model_path "Qwen3-14B" \
    --data_path  "dataset/prm800k/train.json" \
    --output_dir "Results/RED/Qwen3-14B" \
    --data_num 5000 \
    --op_position "ffn_up" \
    --learning_rate 2e-4 \
    --template_index "qwen3_nothink" \
    --layer_type "all" \
    --num_train_epochs 3

    CUDA_VISIBLE_DEVICES=1 python Prefix/answer.py \
    --model_path "Qwen3-14B" \
    --dataset "amc23" \
    --peft "RED" \
    --n_prefix -1 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/RED/Qwen3-14B/5000_prm800k_all_2e-4/ffn_up" \
    --repetition_penalty 1.1 \
    --template_index "qwen3_nothink" \
    --data_num -1

elif [ "$condition" == "weight" ]; then
    
    CUDA_VISIBLE_DEVICES=0 python Prefix/prefix_train_weight.py \
    --model_path "Qwen3-14B" \
    --data_path  "dataset/prm800k/train.json" \
    --output_dir "Results/Test_Weight/Qwen3-14B/" \
    --data_num 5000 \
    --n_prefix 64 \
    --op_position "ffn_up" \
    --learning_rate 2e-5 \
    --layer_type "all" \
    --template_index 'qwen3_nothink' \
    --num_train_epochs 2 \
    --weight 1e-2 \

elif [ "$condition" == "prefix" ]; then

    for n in 1.4 1.7 1.8 1.9 2.0; do
        CUDA_VISIBLE_DEVICES=1 python Prefix/prefix_train.py \
        --model_path "Qwen3-14B" \
        --data_path  "dataset/math10k/train.json" \
        --output_dir "Results/Test_Constraint/Qwen3-14B/" \
        --data_num 5000 \
        --n_prefix 64 \
        --op_position "ffn_up" \
        --learning_rate 2e-5 \
        --layer_type "all" \
        --template_index 'qwen3_nothink' \
        --num_train_epochs 2 \
        --pid_kp 0.1 \
        --pid_ki 0.0001 \
        --pid_kd 0.01 \
        --target_bias $n \
        --min_weight 1e-5 \
        --max_weight 1e-1 \
        --limit_prop 0.8
    done

elif [ "$condition" == "deepspeed" ]; then
    
    for n in 1.5; do
        deepspeed --num_gpus=2 Prefix/prefix_train_deepspeed.py \
        --model_path "Qwen3-14B" \
        --data_path  "dataset/math10k/train.json" \
        --output_dir "Results/Test_Constraint/Qwen3-14B/" \
        --data_num 5000 \
        --n_prefix 64 \
        --op_position "ffn_up" \
        --learning_rate 2e-4 \
        --layer_type "all" \
        --template_index 'qwen3_nothink' \
        --num_train_epochs 2 \
        --pid_kp 0.1 \
        --pid_ki 0.0001 \
        --pid_kd 0.01 \
        --target_bias $n \
        --min_weight 1e-5 \
        --max_weight 1e-1 \
        --limit_prop 0.8
    done

elif [ "$condition" == "eval" ]; then

    data="eval"

    python Prefix/eval.py \
    --data_path "Results/Base/base/9000_Qwen3-14B_all_0_0" \
    --model "qwen" \
    --dataset ${data} \
    --eval_num -1

    python Prefix/eval.py \
    --data_path "Results/Lora/Qwen3-14B" \
    --model "qwen" \
    --dataset ${data} \
    --eval_num -1

    python Prefix/eval.py \
    --data_path "Results/RED/Qwen3-14B" \
    --model "qwen" \
    --dataset ${data} \
    --eval_num -1

    python Prefix/eval.py \
    --data_path "Results/BREP/Qwen3-14B" \
    --model "qwen" \
    --dataset ${data} \
    --eval_num -1

    python Prefix/eval.py \
    --data_path "Results/Test_Constraint/Qwen3-14B" \
    --model "qwen" \
    --dataset ${data} \
    --eval_num -1

elif [ "$condition" == "answer" ]; then

    CUDA_VISIBLE_DEVICES=2 python Prefix/answer.py \
    --model_path "Qwen3-14B" \
    --dataset "amc23" \
    --peft "RED" \
    --n_prefix 6 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Test_Constraint/Qwen3-14B/5000_prm800k_all_64_2e-05_1.2/ffn_up" \
    --repetition_penalty 1.1 \
    --template_index "qwen3_nothink" \
    --data_num -1


elif [ "$condition" == "base_eval" ]; then

    CUDA_VISIBLE_DEVICES=0 python Prefix/base_eval.py \
    --model_path "Qwen3-14B" \
    --output_path "Results/Base/base/9000_Qwen3-14B_all_0_0" \
    --dataset "piqa" \
    --data_num -1 \
    --template_index "qwen3_nothink" \
    --vllm False

fi