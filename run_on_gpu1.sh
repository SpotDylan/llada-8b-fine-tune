#!/bin/bash

# Define variables
GPU_ID=0
MODEL="GSAI-ML/LLaDA-8B-Base"
OUTPUT_DIR="sft_output"
BATCH_SIZE=1
NUM_EPOCHS=3
LEARNING_RATE=2.5e-5
GRADIENT_ACCUMULATION_STEPS=8

# Print configuration
echo "===== Configuration ====="
echo "GPU ID: $GPU_ID"
echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Number of epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "========================="

# Get GPU info
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU_ID 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "Using GPU $GPU_ID: $GPU_INFO"
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo "Set CUDA_VISIBLE_DEVICES=$GPU_ID"
else
    echo "Warning: GPU $GPU_ID not found or nvidia-smi failed"
    unset CUDA_VISIBLE_DEVICES
    echo "CUDA_VISIBLE_DEVICES is unset, will use all available GPUs"
fi

# Step 1: Process SFT data
echo "===== Step 1: Preprocessing SFT data ====="
python process_sft_data.py

# Step 2: Fine-tune model
echo "===== Step 2: Fine-tuning LLaDA ====="
python finetune_llada.py \
    --model_name $MODEL \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --gpu_id $GPU_ID \
    --mixed_precision \
    --gradient_checkpointing \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS

# Step 3: Test model
echo "===== Step 3: Testing fine-tuned model ====="
python test_llada.py --model_path "$OUTPUT_DIR/final"