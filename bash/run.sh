#!/bin/bash
# layer_type: scaling/bias/all 当学习率达到0.01量级时scaling才会发生变化 

# llama3/Qwen2.5/Qwen-math: pip install transformers==4.37.2 trl==0.7.10 "pydantic<2.0.0" accelerate==0.25.0 -U
# LLama3.1/Qwen3: pip install tokenizers==0.21.1 transformers==4.52.3 trl==0.17.0 torchvision==0.16 protobuf==3.20.3

# Meta-Llama-3-8B-Instruct | Meta-Llama-3.1-8B-Instruct | Qwen2.5-Math-7B-Instruct | Qwen3-8B | 3Qwen-14B
# template: llama3 | qwen_base | qwen_base_fewshot | mistral | none
# n_prefix 0:原始模型不调整 -1:reft模型所有位置均调整 n:调整问题及前n个token 训练时n_prefix内置为-1
# Trainset: dataset/prm800k/train.json  dataset/math10k/train.json

condition=$1

if [ "$condition" == "train" ]; then

    # dataset="ultrafeedback"
    # device=2

    dataset="commonsense"
    device=1

    CUDA_VISIBLE_DEVICES=${device} python Prefix/prefix_train.py \
    --model_path "Meta-Llama-3-8B-Instruct" \
    --data_path  "dataset/${dataset}/train.json" \
    --output_dir "Results/BREP/Llama-3-8B/" \
    --data_num 5000 \
    --n_prefix 64 \
    --op_position "ffn_up" \
    --learning_rate 2e-4 \
    --layer_type "all" \
    --num_train_epochs 2 \
    --template_index "llama" \
    --pid_kp 0.1 \
    --pid_ki 0.0001 \
    --pid_kd 0.01 \
    --min_weight 1e-5 \
    --max_weight 1e-1 \
    --limit_prop 0.8 \
    --target_bias 2

    CUDA_VISIBLE_DEVICES=${device} python Prefix/prefix_train.py \
    --model_path "Meta-Llama-3.1-8B-Instruct" \
    --data_path  "dataset/${dataset}/train.json" \
    --output_dir "Results/BREP/Llama-3.1-8B/" \
    --data_num 5000 \
    --n_prefix 64 \
    --op_position "ffn_up" \
    --learning_rate 2e-4 \
    --layer_type "all" \
    --num_train_epochs 2 \
    --template_index "llama" \
    --pid_kp 0.1 \
    --pid_ki 0.0001 \
    --pid_kd 0.01 \
    --min_weight 1e-5 \
    --max_weight 1e-1 \
    --limit_prop 0.8 \
    --target_bias 2

    CUDA_VISIBLE_DEVICES=${device} python Prefix/prefix_train.py \
    --model_path "Qwen3-8B" \
    --data_path  "dataset/${dataset}/train.json" \
    --output_dir "Results/BREP/Qwen3-8B/" \
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
    --target_bias 2 \
    --min_weight 1e-5 \
    --max_weight 1e-1 \
    --limit_prop 0.8

    CUDA_VISIBLE_DEVICES=${device} python Prefix/prefix_train.py \
    --model_path "Qwen3-14B" \
    --data_path  "dataset/${dataset}/train.json" \
    --output_dir "Results/BREP/Qwen3-14B/" \
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
    --target_bias 2 \
    --min_weight 1e-5 \
    --max_weight 1e-1 \
    --limit_prop 0.8


elif [ "$condition" == "base" ]; then

    for data in 'hellaswag'; do
        
        data_num=1000

        CUDA_VISIBLE_DEVICES=0 python Prefix/base_eval.py \
        --model_path "Meta-Llama-3-8B-Instruct" \
        --output_path "Results/Base/base/9000_Llama-3-8B_all_0_0" \
        --dataset ${data} \
        --data_num ${data_num} \
        --template_index "llama" \
        --vllm False

        CUDA_VISIBLE_DEVICES=0 python Prefix/base_eval.py \
        --model_path "Meta-Llama-3.1-8B-Instruct" \
        --output_path "Results/Base/base/9000_Llama-3.1-8B_all_0_0" \
        --dataset ${data} \
        --data_num ${data_num} \
        --template_index "llama" \
        --vllm False

        CUDA_VISIBLE_DEVICES=0 python Prefix/base_eval.py \
        --model_path "Qwen3-8B" \
        --output_path "Results/Base/base/9000_Qwen3-8B_all_0_0" \
        --dataset ${data} \
        --data_num ${data_num} \
        --template_index "qwen3_nothink" \
        --vllm False

        CUDA_VISIBLE_DEVICES=0 python Prefix/base_eval.py \
        --model_path "Qwen3-14B" \
        --output_path "Results/Base/base/9000_Qwen3-14B_all_0_0" \
        --dataset ${data} \
        --data_num ${data_num} \
        --template_index "qwen3_nothink" \
        --vllm False

    done

elif [ "$condition" == "lora" ]; then

    # Meta-Llama-3-8B-Instruct | Meta-Llama-3.1-8B-Instruct | Qwen2.5-Math-7B-Instruct | Qwen3-8B | 3Qwen-14B

    for data in 'hellaswag'; do

        data_num=1000
        
        CUDA_VISIBLE_DEVICES=1 python lora/lora_eval.py \
        --model_path "Meta-Llama-3-8B-Instruct" \
        --lora_path "Results/Lora/Llama-3-8B/5000_prm800k_lora_1e-5"\
        --dataset ${data} \
        --data_num ${data_num} \
        --template_index "llama"

        CUDA_VISIBLE_DEVICES=1 python lora/lora_eval.py \
        --model_path "Meta-Llama-3.1-8B-Instruct" \
        --lora_path "Results/Lora/Llama-3.1-8B/5000_prm800k_lora_1e-05"\
        --dataset ${data} \
        --data_num ${data_num} \
        --template_index "llama"

        CUDA_VISIBLE_DEVICES=1 python lora/lora_eval.py \
        --model_path "Qwen3-8B" \
        --lora_path "Results/Lora/Qwen3-8B/5000_prm800k_lora_1e-06"\
        --dataset ${data} \
        --data_num ${data_num} \
        --template_index "qwen3_nothink"

        CUDA_VISIBLE_DEVICES=1 python lora/lora_eval.py \
        --model_path "Qwen3-14B" \
        --lora_path "Results/Lora/Qwen3-14B/5000_prm800k_lora_1e-06"\
        --dataset ${data} \
        --data_num ${data_num} \
        --template_index "qwen3_nothink"

    done

elif [ "$condition" == "red" ]; then

    # Meta-Llama-3-8B-Instruct | Meta-Llama-3.1-8B-Instruct | Qwen2.5-Math-7B-Instruct | Qwen3-8B | 3Qwen-14B
    
    for prefix in 0 1 4 8 16 32 64 128; do

        data='gsm8k'
        data_num=500
        
        CUDA_VISIBLE_DEVICES=0 python Prefix/answer.py \
        --model_path "Meta-Llama-3-8B-Instruct" \
        --dataset ${data} \
        --peft "RED" \
        --n_prefix ${prefix} \
        --is_train_return False \
        --no_repeat_ngram_size 5 \
        --peft_path "Results/RED/Llama-3-8B/5000_prm800k_all_2e-5/ffn_up" \
        --repetition_penalty 1.1 \
        --template_index "llama" \
        --data_num ${data_num}

        # CUDA_VISIBLE_DEVICES=2 python Prefix/answer.py \
        # --model_path "Meta-Llama-3.1-8B-Instruct" \
        # --dataset ${data} \
        # --peft "RED" \
        # --n_prefix -1 \
        # --is_train_return False \
        # --no_repeat_ngram_size 5 \
        # --peft_path "Results/RED/Llama-3.1-8B/5000_prm800k_all_2e-05/ffn_up" \
        # --repetition_penalty 1.1 \
        # --template_index "llama" \
        # --data_num ${data_num}

        # CUDA_VISIBLE_DEVICES=2 python Prefix/answer.py \
        # --model_path "Qwen3-8B" \
        # --dataset ${data} \
        # --peft "RED" \
        # --n_prefix -1 \
        # --is_train_return False \
        # --no_repeat_ngram_size 5 \
        # --peft_path "Results/RED/Qwen3-8B/5000_prm800k_all_2e-4/ffn_up" \
        # --repetition_penalty 1.1 \
        # --template_index "qwen3_nothink" \
        # --data_num ${data_num}

        # CUDA_VISIBLE_DEVICES=2 python Prefix/answer.py \
        # --model_path "Qwen3-14B" \
        # --dataset ${data} \
        # --peft "RED" \
        # --n_prefix -1 \
        # --is_train_return False \
        # --no_repeat_ngram_size 5 \
        # --peft_path "Results/RED/Qwen3-14B/5000_prm800k_all_2e-4/ffn_up" \
        # --repetition_penalty 1.1 \
        # --template_index "qwen3_nothink" \
        # --data_num ${data_num}

    done

elif [ "$condition" == "brep" ]; then

    # Meta-Llama-3-8B-Instruct | Meta-Llama-3.1-8B-Instruct | Qwen2.5-Math-7B-Instruct | Qwen3-8B | 3Qwen-14B
    
    for data in 'hellaswag'; do

        data_num=1000
        
        CUDA_VISIBLE_DEVICES=3 python Prefix/answer.py \
        --model_path "Meta-Llama-3-8B-Instruct" \
        --dataset ${data} \
        --peft "RED" \
        --n_prefix 8 \
        --is_train_return False \
        --no_repeat_ngram_size 5 \
        --peft_path "Results/BREP/Llama-3-8B/5000_prm800k_all_64_2e-4_1.0/ffn_up" \
        --repetition_penalty 1.1 \
        --template_index "llama" \
        --data_num ${data_num}

        CUDA_VISIBLE_DEVICES=3 python Prefix/answer.py \
        --model_path "Meta-Llama-3.1-8B-Instruct" \
        --dataset ${data} \
        --peft "RED" \
        --n_prefix 8 \
        --is_train_return False \
        --no_repeat_ngram_size 5 \
        --peft_path "Results/BREP/Llama-3.1-8B/5000_prm800k_all_64_2e-4_0.8/ffn_up" \
        --repetition_penalty 1.1 \
        --template_index "llama" \
        --data_num ${data_num}

        CUDA_VISIBLE_DEVICES=3 python Prefix/answer.py \
        --model_path "Qwen3-8B" \
        --dataset ${data} \
        --peft "RED" \
        --n_prefix 8 \
        --is_train_return False \
        --no_repeat_ngram_size 5 \
        --peft_path "Results/BREP/Qwen3-8B/5000_prm800k_all_64_2e-05_1.2/ffn_up" \
        --repetition_penalty 1.1 \
        --template_index "qwen3_nothink" \
        --data_num ${data_num}

        CUDA_VISIBLE_DEVICES=3 python Prefix/answer.py \
        --model_path "Qwen3-14B" \
        --dataset ${data} \
        --peft "RED" \
        --n_prefix 8 \
        --is_train_return False \
        --no_repeat_ngram_size 5 \
        --peft_path "Results/BREP/Qwen3-14B/5000_prm800k_all_64_2e-05_1.4/ffn_up" \
        --repetition_penalty 1.1 \
        --template_index "qwen3_nothink" \
        --data_num ${data_num}

    done

elif [ "$condition" == "ablation" ]; then

    python Prefix/eval.py \
    --data_path "Results/Ablation/Llama-3-8B" \
    --model "" \
    --dataset "eval" \
    --eval_num -1

    python Prefix/eval.py \
    --data_path "Results/Ablation/Qwen3-8B" \
    --model "" \
    --dataset "eval" \
    --eval_num -1

elif [ "$condition" == "eval" ]; then
    
    data="eval"
    type="ReFT"
    eval_num=-1

    # python Prefix/eval.py \
    # --data_path "Results/Base/base" \
    # --model "" \
    # --dataset ${data} \
    # --eval_num ${eval_num}

    python Prefix/eval.py \
    --data_path "Results/${type}/Llama-3-8B" \
    --model "" \
    --dataset ${data} \
    --eval_num ${eval_num}

    python Prefix/eval.py \
    --data_path "Results/${type}/Llama-3.1-8B" \
    --model "" \
    --dataset ${data} \
    --eval_num ${eval_num}

    python Prefix/eval.py \
    --data_path "Results/${type}/Qwen2.5-Math-7B" \
    --model "" \
    --dataset ${data} \
    --eval_num ${eval_num}

    python Prefix/eval.py \
    --data_path "Results/${type}/Qwen3-8B" \
    --model "" \
    --dataset ${data} \
    --eval_num ${eval_num}

    python Prefix/eval.py \
    --data_path "Results/${type}/Qwen3-14B" \
    --model "" \
    --dataset ${data} \
    --eval_num ${eval_num}
fi