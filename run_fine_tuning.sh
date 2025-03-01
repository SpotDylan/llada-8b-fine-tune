#!/bin/bash

# Enable memory optimization to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set variables
MODEL_NAME="GSAI-ML/LLaDA-8B-Base"  # Change to your model name/path
OUTPUT_DIR="sft_output"
BATCH_SIZE=4  # Adjust based on your GPU memory
LEARNING_RATE=2.5e-5
NUM_EPOCHS=3
SAVE_STEPS=500
GPU_ID=0  # Set to the GPU ID you want to use (default: 0)

# Create output directory
mkdir -p $OUTPUT_DIR

echo "===== Step 1: Preprocessing SFT data ====="
python preprocess_sft_data.py

echo "===== Step 2: Fine-tuning LLaDA ====="
python finetune_llada.py \
    --model_name $MODEL_NAME \
    --data_path "sft_data/processed_data.pt" \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --save_steps $SAVE_STEPS \
    --gpu_id $GPU_ID

echo "===== Step 3: Testing fine-tuned model ====="
python inference_example.py \
    --model_path "$OUTPUT_DIR/final" \
    --steps 128 \
    --gen_length 128 \
    --block_length 32 \
    --temperature 0.0 \
    --cfg_scale 0.0 \
    --remasking "low_confidence" \
    --gpu_id $GPU_ID

echo "===== Fine-tuning pipeline completed ====="
