#!/bin/bash

# Set variables
MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"  # or "GSAI-ML/LLaDA-8B-Instruct"
OUTPUT_DIR="output"
BATCH_SIZE=2
EPOCHS=3
LEARNING_RATE=2.5e-5
WEIGHT_DECAY=0.1
MAX_LENGTH=512
USE_BF16=""  # Set to "--use_bf16" if you have GPU with bfloat16 support

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p sft_data

echo "========================================"
echo "LLaDA Supervised Fine-Tuning Pipeline"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "========================================"

# Step 1: Preprocess the data
echo "Step 1: Preprocessing data..."
python preprocess_sft_data.py \
  --model_name $MODEL_NAME \
  --output_dir sft_data \
  --max_length $MAX_LENGTH

# Check if preprocessing was successful
if [ ! -f "sft_data/sft_data.pt" ]; then
  echo "Error: Preprocessing failed. sft_data.pt not found."
  exit 1
fi

# Step 2: Fine-tune the model
echo "Step 2: Fine-tuning model..."
python finetune_llada.py \
  --model_name $MODEL_NAME \
  --data_path sft_data/sft_data.pt \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  $USE_BF16

# Check if fine-tuning was successful
if [ ! -d "$OUTPUT_DIR/final" ]; then
  echo "Error: Fine-tuning failed. Final model not found."
  exit 1
fi

# Step 3: Run inference with the fine-tuned model
echo "Step 3: Testing the fine-tuned model..."
INSTRUCT_FLAG=""
if [[ $MODEL_NAME == *"Instruct"* ]]; then
  INSTRUCT_FLAG="--instruct"
fi

python inference_example.py \
  --model_path $OUTPUT_DIR/final \
  --prompts "What is the capital of France?" "Explain the concept of supervised fine-tuning." \
  $INSTRUCT_FLAG \
  $USE_BF16

echo "========================================"
echo "Fine-tuning pipeline completed successfully!"
echo "The fine-tuned model is saved at: $OUTPUT_DIR/final"
echo "========================================"
