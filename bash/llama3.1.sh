#!/bin/bash
# layer_type: scaling/bias/all 当学习率达到0.01量级时scaling才会发生变化 

# llama3/Qwen2.5/Qwen-math: pip install transformers==4.37.2 trl==0.7.10 "pydantic<2.0.0" accelerate==0.25.0 -U
# LLama3.1/Qwen3: pip install tokenizers==0.21.1 transformers==4.52.3 trl==0.17.0 torchvision==0.16 protobuf==3.20.3

# Meta-Llama-3-8B-Instruct | Qwen2.5-7B-Instruct | Qwen2.5-Math-7B-Instruct | Qwen3-8B | DeepSeek-R1-Distill-Qwen-7B
# template: llama3 | qwen_base | qwen_base_fewshot | mistral | none

# n_prefix 0:原始模型不调整 -1:reft模型所有位置均调整 n:调整问题及前n个token
# 训练时n_prefix内置为-1

condition=$1

if [ "$condition" == "red" ]; then

    CUDA_VISIBLE_DEVICES=1 python Prefix/red_train.py \
    --model_path "Meta-Llama-3.1-8B-Instruct" \
    --data_path  "dataset/prm800k/train.json" \
    --output_dir "Results/RED/Llama-3.1-8B" \
    --data_num 5000 \
    --op_position "ffn_up" \
    --learning_rate 2e-5 \
    --template_index "llama" \
    --layer_type "all" \
    --num_train_epochs 3

    CUDA_VISIBLE_DEVICES=1 python Prefix/answer.py \
    --model_path "Meta-Llama-3.1-8B-Instruct" \
    --dataset "amc23" \
    --peft "RED" \
    --n_prefix -1 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/RED/Llama-3.1-8B/5000_prm800k_all_2e-05/ffn_up" \
    --repetition_penalty 1.1 \
    --template_index "llama" \
    --data_num -1

elif [ "$condition" == "eval" ]; then

    python Prefix/eval.py \
    --data_path "Results/Base/base/9000_Llama-3.1-8B_all_0_0" \
    --model "Llama" \
    --dataset "" \
    --eval_num -1

    python Prefix/eval.py \
    --data_path "Results/Lora/Llama-3.1-8B" \
    --model "Llama" \
    --dataset "" \
    --eval_num -1

    python Prefix/eval.py \
    --data_path "Results/RED/Llama-3.1-8B" \
    --model "Llama" \
    --dataset "" \
    --eval_num -1

    python Prefix/eval.py \
    --data_path "Results/BREP/Llama-3.1-8B" \
    --model "Llama" \
    --dataset "" \
    --eval_num -1

    # python Prefix/eval.py \
    # --data_path "Results/Test_Constraint/Llama-3.1-8B" \
    # --model "Llama" \
    # --dataset "" \
    # --eval_num -1

elif [ "$condition" == "eval_3" ]; then

    data='eval'

    python Prefix/eval.py \
    --data_path "Results/Base/base/9000_Llama3-8B_all_0_0" \
    --model "Llama" \
    --dataset ${data} \
    --eval_num -1

    python Prefix/eval.py \
    --data_path "Results/Lora/Llama3-8B" \
    --model "Llama" \
    --dataset ${data} \
    --eval_num -1

    python Prefix/eval.py \
    --data_path "Results/RED/Llama3-8B" \
    --model "Llama" \
    --dataset ${data} \
    --eval_num -1

    # python Prefix/eval.py \
    # --data_path "Results/Ablation/Llama-3-8B" \
    # --model "Llama" \
    # --dataset ${data} \
    # --eval_num -1

    # python Prefix/eval.py \
    # --data_path "Results/Prefix/Llama-3-8B" \
    # --model "Llama" \
    # --dataset ${data} \
    # --eval_num -1
    
    python Prefix/eval.py \
    --data_path "Results/Test_Constraint/Llama-3-8B" \
    --model "Llama" \
    --dataset ${data} \
    --eval_num -1

elif [ "$condition" == "answer" ]; then
    # Prefix: Results/Test_Length/Llama3/5000_math10k_all_64_0.0002_0.82/ffn_up
    # RED: Results/RED/Llama3/5000_math10k_all_2e-4/ffn_up

    CUDA_VISIBLE_DEVICES=1 python Prefix/answer.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --dataset "amc23" \
    --peft "RED" \
    --n_prefix 8 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Test_Constraint/Llama-3-8B/5000_prm800k_all_64_2e-4_1.0/ffn_up" \
    --repetition_penalty 1.1 \
    --template_index "llama" \
    --data_num 500


elif [ "$condition" == "length" ]; then

    CUDA_VISIBLE_DEVICES=2 python Prefix/answer.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --dataset "gsm8k" \
    --peft "RED" \
    --n_prefix 4 \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --peft_path "Results/Test_Length/Llama-3-8B/5000_math10k_all_64_2e-4_1/ffn_up" \
    --repetition_penalty 1.1 \
    --template_index "llama" \
    --data_num -1

elif [ "$condition" == "prefix_compare" ]; then
    
    for n in 1 4 8 16 32 64 128 256; do
        
        CUDA_VISIBLE_DEVICES=1 python Prefix/answer.py \
        --model_path "Meta-Llama-3-8B-Instruct" \
        --dataset "gsm8k" \
        --peft "RED" \
        --n_prefix $n \
        --is_train_return False \
        --no_repeat_ngram_size 5 \
        --peft_path "Results/Prefix/Llama-3-8B/5000_math10k_all_2e-4_RED/ffn_up" \
        --repetition_penalty 1.1 \
        --template_index "llama" \
        --data_num 500

        CUDA_VISIBLE_DEVICES=1 python Prefix/answer.py \
        --model_path "Meta-Llama-3-8B-Instruct" \
        --dataset "gsm8k" \
        --peft "RED" \
        --n_prefix $n \
        --is_train_return False \
        --no_repeat_ngram_size 5 \
        --peft_path "Results/Prefix/Llama-3-8B/5000_math10k_all_64_2e-4_1_BREP/ffn_up" \
        --repetition_penalty 1.1 \
        --template_index "llama" \
        --data_num 500
    
    done

    
elif [ "$condition" == "base_eval" ]; then


    CUDA_VISIBLE_DEVICES=2 python Prefix/base_eval.py \
    --model_path "Meta-Llama-3.1-8B-Instruct" \
    --output_path "Results/Base/base/9000_Llama-3.1-8B_all_0_0" \
    --dataset "boolq" \
    --data_num -1 \
    --template_index "llama" \
    --vllm False

    CUDA_VISIBLE_DEVICES=2 python Prefix/base_eval.py \
    --model_path "Meta-Llama-3.1-8B-Instruct" \
    --output_path "Results/Base/base/9000_Llama-3.1-8B_all_0_0" \
    --dataset "piqa" \
    --data_num -1 \
    --template_index "llama" \
    --vllm False


fi