#!/bin/bash

# Script to run the entire LLaDA fine-tuning pipeline
# This script preprocesses the data, fine-tunes the model, and runs an example inference

# Exit on error
set -e

# Configuration
MODEL_NAME_OR_PATH="llada-8b"  # Path to the LLaDA model or model name
INPUT_FILE="sft_data/conversations.json"
OUTPUT_DIR="models/llada-sft"
PROCESSED_DATA_DIR="sft_data/processed"
MAX_SEQ_LENGTH=4096
BATCH_SIZE=2
LEARNING_RATE=2.5e-5
WEIGHT_DECAY=0.1
NUM_EPOCHS=3
WARMUP_RATIO=0.03
SEED=42

# Create directories
mkdir -p $PROCESSED_DATA_DIR
mkdir -p $OUTPUT_DIR

echo "=== LLaDA Supervised Fine-Tuning Pipeline ==="
echo "Model: $MODEL_NAME_OR_PATH"
echo "Input data: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Processed data directory: $PROCESSED_DATA_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Number of epochs: $NUM_EPOCHS"
echo ""

# Step 1: Preprocess the conversation data
echo "Step 1: Preprocessing conversation data..."
python preprocess_sft_data.py \
    --input_file $INPUT_FILE \
    --output_dir $PROCESSED_DATA_DIR \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --max_seq_length $MAX_SEQ_LENGTH \
    --pad_to_max_length

# Step 2: Fine-tune the LLaDA model
echo "Step 2: Fine-tuning the LLaDA model..."
python finetune_llada.py \
    --data_path $PROCESSED_DATA_DIR/sft_data.pt \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --num_epochs $NUM_EPOCHS \
    --warmup_ratio $WARMUP_RATIO \
    --seed $SEED

# Step 3: Run an example inference
echo "Step 3: Running example inference..."
python inference_example.py \
    --model_path $OUTPUT_DIR \
    --prompt "What is the capital of France?" \
    --sampling_method semi_autoregressive_padding \
    --remask_strategy low_confidence \
    --temperature 0.7 \
    --top_p 0.9

echo "Fine-tuning pipeline completed successfully!"
echo "The fine-tuned model is saved in: $OUTPUT_DIR"
echo ""
echo "To run interactive inference with the model, use:"
echo "python inference_example.py --model_path $OUTPUT_DIR --interactive"
