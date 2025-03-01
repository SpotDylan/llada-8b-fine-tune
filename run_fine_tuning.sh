#!/bin/bash

# Enable memory optimization to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set variables
MODEL_NAME="GSAI-ML/LLaDA-8B-Base"  # Change to your model name/path
OUTPUT_DIR="sft_output"
BATCH_SIZE=1  # Adjust based on your GPU memory
LEARNING_RATE=2.5e-5
NUM_EPOCHS=3
SAVE_STEPS=500
MEMORY_LIMIT=65.0  # Memory limit in GB across all GPUs

# Create output directory
mkdir -p $OUTPUT_DIR

echo "===== Step 1: Preprocessing SFT data ====="
python preprocess_sft_data.py

echo "===== Step 2: Fine-tuning LLaDA with Distributed Training ====="
echo "Using up to 8 GPUs with memory limit of $MEMORY_LIMIT GB"
python finetune_llada.py \
    --model_name $MODEL_NAME \
    --data_path "sft_data/processed_data.pt" \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --save_steps $SAVE_STEPS \
    --memory_limit $MEMORY_LIMIT

echo "===== Step 3: Testing fine-tuned model ====="
python inference_example.py \
    --model_path "$OUTPUT_DIR/final" \
    --steps 128 \
    --gen_length 128 \
    --block_length 32 \
    --temperature 0.0 \
    --cfg_scale 0.0 \
    --remasking "low_confidence" \
    --gpu_id 0  # Use first GPU for inference

echo "===== Fine-tuning pipeline completed ====="
