model=$1

# Meta-Llama-3-8B-Instruct  
# Meta-Llama-3.1-8B-Instruct 
# Qwen2.5-7B-Instruct
# Mistral-7B-Instruct-v0.2 
# DeepSeek-R1-Distill-Qwen-7B

if [ "$model" == "truthful" ]; then

    task="Truthful"
    model_name="Meta-Llama-3-8B-Instruct"
    output_path="Truthful/truthful_results/llama3_ffn_down"
    prefix_path="Results/Test/Llama3/9000_math10k_all_32_2e-4/ffn_down"
    reft_path="Results/RED/Llama3/9000_math10k_all_2e-4/ffn_down"

    # python Truthful/train_probe.py \
    # --task ${task} \
    # --model_name ${model_name} \
    # --output_path ${output_path} \
    # --prefix_path ${prefix_path} \
    # --reft_path ${reft_path} 

    # python Truthful/prob_hs.py \
    # --model_name ${model_name} \
    # --output_path ${output_path} \
    # --prefix_path ${prefix_path} \
    # --reft_path ${reft_path} \
    # --layer_type "all" 

    python Truthful/draw_prob_curve.py \
    --output_path ${output_path} 

    python Truthful/draw_heat_map.py \
    --output_path ${output_path} 

elif [ "$model" == "faithful" ]; then

    task="Faithful"
    model_name="Meta-Llama-3-8B-Instruct"
    output_path="Truthful/faithful_results/llama3_ffn_down"
    prefix_path="Results/Test/Llama3/9000_math10k_all_32_2e-4/ffn_down"
    reft_path="Results/RED/Llama3/9000_math10k_all_2e-4/ffn_down"

    # python Truthful/train_probe.py \
    # --task ${task} \
    # --model_name ${model_name} \
    # --output_path ${output_path} \
    # --prefix_path ${prefix_path} \
    # --reft_path ${reft_path} 

    # python Truthful/prob_hs.py \
    # --model_name ${model_name} \
    # --output_path ${output_path} \
    # --prefix_path ${prefix_path} \
    # --reft_path ${reft_path} \
    # --layer_type "all" 

    # python Truthful/draw_prob_curve.py \
    # --output_path ${output_path} 

    python Truthful/draw_heat_map.py \
    --output_path ${output_path} 

fi
