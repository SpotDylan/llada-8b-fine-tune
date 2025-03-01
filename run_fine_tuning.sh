#!/bin/bash

# run_fine_tuning.sh
# End-to-end SFT workflow for LLaDA-8B-Instruct

set -e  # Exit immediately on error

# Configuration
DATA_PATH="fake_sft_data.jsonl"
PROCESSED_DATA="processed_sft_data.pt"
MODEL_SAVE_PATH="llada-8b-instruct-sft.pt"
MAX_SEQ_LENGTH=4096  # Should match model's max context length

# Step 1: Data preprocessing
echo "Starting data preprocessing..."
python preprocess_sft_data.py \
    --data_path "$DATA_PATH" \
    --max_length $MAX_SEQ_LENGTH \
    --save_path "$PROCESSED_DATA"

# Step 2: Supervised Fine-Tuning
echo "Starting fine-tuning..."
python finetune_llada.py \
    --processed_data "$PROCESSED_DATA" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --batch_size 2 \
    --epochs 3 \
    --learning_rate 2.5e-5

# Step 3: Inference test
echo "Running inference test..."
python inference_example.py \
    --model_path "$MODEL_SAVE_PATH" \
    --prompt "Calculate 3 + 5." \
    --max_length 64

echo "Fine-tuning workflow completed successfully!"