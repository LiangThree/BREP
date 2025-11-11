#!/bin/bash
# layer_type: scaling/bias/all 当学习率达到0.01量级时scaling才会发生变化 

# llama3/Qwen2.5/Qwen-math: pip install transformers==4.37.2 trl==0.7.10 "pydantic<2.0.0" accelerate==0.25.0 -U
# LLama3.1/Qwen3: pip install tokenizers==0.21.1 transformers==4.52.3 trl==0.17.0 torchvision==0.16 protobuf==3.20.3

# Meta-Llama-3-8B-Instruct | Meta-Llama-3.1-8B-Instruct | Qwen2.5-Math-7B-Instruct | Qwen3-8B | 3Qwen-14B
# template: llama3 | qwen_base | qwen_base_fewshot | mistral | none
# n_prefix 0:原始模型不调整 -1:reft模型所有位置均调整 n:调整问题及前n个token 训练时n_prefix内置为-1
# Trainset: dataset/prm800k/train.json  dataset/math10k/train.json

condition=$1

if [ "$condition" == "red" ]; then

    # CUDA_VISIBLE_DEVICES=1 python Prefix/red_train.py \
    # --model_path "Meta-Llama-3-8B-Instruct" \
    # --data_path  "dataset/prm800k/train.json" \
    # --output_dir "Results/RED/Llama-3-8B" \
    # --data_num 5000 \
    # --op_position "ffn_up" \
    # --learning_rate 2e-5 \
    # --template_index "llama" \
    # --layer_type "all" \
    # --num_train_epochs 3

    for prefix in 1 4 8 16 32 64; do
        CUDA_VISIBLE_DEVICES=0 python Prefix/answer.py \
        --model_path "Meta-Llama-3-8B-Instruct" \
        --dataset "gsm8k" \
        --peft "RED" \
        --n_prefix ${prefix} \
        --is_train_return False \
        --no_repeat_ngram_size 5 \
        --peft_path "Results/RED/Llama-3-8B/5000_math10k_all_2e-4/ffn_up" \
        --repetition_penalty 1.1 \
        --template_index "llama" \
        --data_num 500

        # CUDA_VISIBLE_DEVICES=1 python Prefix/answer.py \
        # --model_path "Meta-Llama-3-8B-Instruct" \
        # --dataset "math500" \
        # --peft "RED" \
        # --n_prefix ${prefix} \
        # --is_train_return False \
        # --no_repeat_ngram_size 5 \
        # --peft_path "Results/RED/Llama-3-8B/5000_prm800k_all_2e-5/ffn_up" \
        # --repetition_penalty 1.1 \
        # --template_index "llama" \
        # --data_num 500

    done

    

elif [ "$condition" == "weight" ]; then

    for n in 5e-2 5e-3 5e-4 5e-5 1e-5 5e-6 1e-6 5; do
        
        # CUDA_VISIBLE_DEVICES=0 python Prefix/prefix_train_weight.py \
        # --model_path "Meta-Llama-3-8B-Instruct" \
        # --data_path  "dataset/math10k/train.json" \
        # --output_dir "Results/Test_Weight/Llama-3-8B/" \
        # --data_num 5000 \
        # --n_prefix 64 \
        # --op_position "ffn_up" \
        # --learning_rate 2e-4 \
        # --layer_type "all" \
        # --num_train_epochs 2 \
        # --template_index "llama" \
        # --weight ${n} \

        CUDA_VISIBLE_DEVICES=3 python Prefix/answer.py \
        --model_path "Meta-Llama-3-8B-Instruct" \
        --dataset "gsm8k" \
        --peft "RED" \
        --n_prefix 8 \
        --is_train_return False \
        --no_repeat_ngram_size 5 \
        --peft_path "Results/Test_Weight/Llama-3-8B/5000_math10k_all_64_2e-4_${n}/ffn_up" \
        --repetition_penalty 1.1 \
        --template_index "llama" \
        --data_num 500
        
    done

elif [ "$condition" == "prefix" ]; then

    CUDA_VISIBLE_DEVICES=0 python Prefix/prefix_train.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --data_path  "dataset/math10k/train.json" \
    --output_dir "Results/Test_Constraint/Llama-3-8B/" \
    --data_num 5000 \
    --n_prefix 64 \
    --op_position "ffn_up" \
    --learning_rate 2e-5 \
    --layer_type "all" \
    --num_train_epochs 2 \
    --template_index "llama" \
    --pid_kp 0.1 \
    --pid_ki 0.0001 \
    --pid_kd 0.01 \
    --min_weight 1e-5 \
    --max_weight 1e-1 \
    --limit_prop 0.8 \
    --target_bias 1

elif [ "$condition" == "eval" ]; then

    dataset="eval" # This can be set: gsm8k/svamp/mathqa/math500/amc23 to eval single dataset

    python Prefix/eval.py \
    --data_path "Results/Base/base/9000_Llama-3-8B_all_0_0" \
    --model "Llama" \
    --dataset ${dataset} \
    --eval_num -1

    python Prefix/eval.py \
    --data_path "Results/Lora/Llama-3-8B" \
    --model "Llama" \
    --dataset ${dataset} \
    --eval_num -1

    python Prefix/eval.py \
    --data_path "Results/RED/Llama-3-8B" \
    --model "Llama" \
    --dataset ${dataset} \
    --eval_num -1

    python Prefix/eval.py \
    --data_path "Results/Test_Weight/Llama-3-8B" \
    --model "Llama" \
    --dataset ${dataset} \
    --eval_num -1

    # python Prefix/eval.py \
    # --data_path "Results/Test_Constraint/Llama-3.1-8B" \
    # --model "Llama" \
    # --dataset "" \
    # --eval_num -1


elif [ "$condition" == "answer" ]; then

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

    
elif [ "$condition" == "base_eval" ]; then

    CUDA_VISIBLE_DEVICES=0 python Prefix/base_eval.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --output_path "Results/Base/base/9000_Llama3-8B_all_0_0" \
    --dataset "gsm8k" \
    --data_num -1 \
    --template_index "llama" \
    --vllm False

fi