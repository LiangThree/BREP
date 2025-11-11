condition=$1

if [ "$condition" == "get_embeds" ]; then
    python Numprob/get_embeds.py \
        --data_path dataset/cal_raw/raw.jsonl \
        --output_path Numprob/embeddings/llama3 \
        --model_path /mnt/usercache/huggingface/Meta-Llama-3-8B-Instruct \
        --max_new_tokens 30 \
        --num_layers 32 \
        --save_embeds

elif [ "$condition" == "prober" ]; then
    python Numprob/prober.py \
    --data_path Numprob/embeddings/llama3 \
    --output_path Numprob/model/llama3 \
    --num_layers 32 \
    --penalty ridge \
    --logscale_prediction \
    --alpha 0.1

elif [ "$condition" == "analysis" ]; then
    
    python Numprob/analysis.py \
    --data_path Numprob/embeddings/llama3 \
    --format pdf \
    --model_path Numprob/model/llama3 \
    --output_path Numprob/model/llama3/figures \
    --split test \
    --penalty ridge \
    --layer_count 32 \
    --logscale_prediction \
    --target all

elif [ "$condition" == "reft_intervene" ]; then
    # pip install jaxtyping better_abc
    
    python Numprob/reft_intervene.py \
        --data_path dataset/cal_4/4.jsonl \
        --probe_path Numprob/model/llama3 \
        --penalty ridge \
        --output_path Numprob/intervene/reft/ \
        --model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --model_path /mnt/usercache/huggingface/Meta-Llama-3-8B-Instruct \
        --max_new_tokens 15 \
        --num_layers 32 \
        --num_examples 1000 \
        --task_type reft \
        --start_layer 14 \
        --end_layer 15\
        --task_param 6 \
        --random_intervention False \
        --null_intervention False \
        --delta '[0.03, 0.05, 0.1, 0.2]'

elif [ "$condition" == "analyze_reft_transfer" ]; then

    python Numprob/analyze_reft_transfer.py

fi