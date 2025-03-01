#!/bin/bash

# Run the alignment fine-tuning pipeline for LLaDA 8B
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set variables
MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
INPUT_FILE="sft_data/llama_logits.json"
PROCESSED_FILE="sft_data/processed_alignment.pt"
OUTPUT_DIR="alignment_output"
BATCH_SIZE=2
LEARNING_RATE=2.5e-5
WEIGHT_DECAY=0.1
EPOCHS=3
WARMUP_STEPS=50
MAX_LENGTH=256
TEMPERATURE=1.0
LOG_INTERVAL=100

# Create directories
mkdir -p alignment_data
mkdir -p $OUTPUT_DIR

echo "=== LLaDA 8B Alignment Fine-Tuning Pipeline ==="
echo "Model: $MODEL_NAME"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Temperature: $TEMPERATURE"
echo ""

# Step 1: Preprocess data
echo "Step 1: Preprocessing alignment data..."
python preprocess_alignment_data.py \
    --input_file $INPUT_FILE \
    --output_file $PROCESSED_FILE \
    --model_name $MODEL_NAME \
    --max_length $MAX_LENGTH

# Check if preprocessing was successful
if [ ! -f "$PROCESSED_FILE" ]; then
    echo "Error: Preprocessing failed. $PROCESSED_FILE not found."
    exit 1
fi

# Step 2: Fine-tune model
echo "Step 2: Fine-tuning model for alignment..."
python finetune_llada_alignment.py \
    --data_path $PROCESSED_FILE \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --epochs $EPOCHS \
    --warmup_steps $WARMUP_STEPS \
    --temperature $TEMPERATURE \
    --log_interval $LOG_INTERVAL

# Check if fine-tuning was successful
if [ ! -d "$OUTPUT_DIR/final" ]; then
    echo "Error: Fine-tuning failed. $OUTPUT_DIR/final not found."
    exit 1
fi

# Step 3: Test the fine-tuned model
echo "Step 3: Testing fine-tuned model..."
python inference_example.py \
    --model_path "$OUTPUT_DIR/final" \
    --prompt "What is the capital of France?" \
    --gen_length 128 \
    --steps 128 \
    --block_length 32

echo ""
echo "Alignment fine-tuning pipeline completed successfully!"
echo "Fine-tuned model saved to: $OUTPUT_DIR/final"
