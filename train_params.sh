#!/bin/bash

USE_LORA=true
MODEL_PATH="/data/home/xmju/models/Qwen3-Embedding-0.6B"
DEVICE="cuda"
OUTPUT_DIR="./checkpoints"
EVAL_EPOCHS=1
MAX_LENGTH=2048
ANCHORS=53
CUTTOFF=0.02
EPOCHS=5
BATCH_SIZE=4
SEED=42

datasets=("citeseer")
temperatures=(0.07 0.1 0.15 0.2)
lrs=(2e-6 1e-6 5e-7 1e-7)

export MASTER_PORT=$((10000 + RANDOM % 50000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

for dataset in "${datasets[@]}"; do
    RESULTS_FILE="./datasets/${dataset}/contrastive_training_results.json"
    if [ ! -f "$RESULTS_FILE" ]; then
        echo "[]" > "$RESULTS_FILE"
    fi

    for temperature in "${temperatures[@]}"; do
        for lr in "${lrs[@]}"; do
            echo "开始训练: dataset=$dataset, temp=$temperature, lr=$lr"
            echo "=========================================="

            config_tag="${dataset}-temp${temperature}-lr${lr}"
            config_output_dir="./datasets/${dataset}/checkpoints/${config_tag}"
            
            deepspeed --num_gpus=2 contrastive_train.py \
                --deepspeed ds_config1.json \
                --dataset $dataset \
                --anchors $ANCHORS \
                --cuttoff $CUTTOFF \
                --seed $SEED \
                --batch_size $BATCH_SIZE \
                --epochs $EPOCHS \
                --lr $lr \
                --temperature $temperature \
                --model_path $MODEL_PATH \
                --device $DEVICE \
                --output_dir $config_output_dir \
                --eval_epochs $EVAL_EPOCHS \
                --max_length $MAX_LENGTH \
                --results_file $RESULTS_FILE \
                --config_tag $config_tag \
                $(if [ "$USE_LORA" = "true" ]; then echo "--use_lora --lora_r 4 --lora_alpha 8 --lora_dropout 0.1"; fi)
            
            echo "配置训练完成: $config_tag"
            echo ""
        done
    done
done

echo "所有配置训练完成！各数据集结果已写入各自 datasets/<dataset>/contrastive_training_results.json"