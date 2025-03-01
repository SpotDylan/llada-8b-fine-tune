#!/bin/bash

# Run Fine-Tuning Pipeline for LLaDA
# This script runs the entire supervised fine-tuning workflow for LLaDA

# Set variables
NUM_EXAMPLES=500
MULTI_TURN_RATIO=0.3
MAX_LENGTH=2048
NUM_EPOCHS=3
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-5
WARMUP_RATIO=0.1
STEPS=128
GEN_LENGTH=128
BLOCK_LENGTH=32

# Create directories
mkdir -p sft_data/preprocessed
mkdir -p sft_output

# Print header
echo "========================================================"
echo "LLaDA Supervised Fine-Tuning Pipeline"
echo "========================================================"
echo ""

# Step 1: Generate fake dataset
echo "Step 1: Generating fake dataset..."
python generate_fake_sft_data.py \
    --num_examples $NUM_EXAMPLES \
    --output sft_data/conversations.json \
    --multi_turn_ratio $MULTI_TURN_RATIO

if [ $? -ne 0 ]; then
    echo "Error generating fake dataset. Exiting."
    exit 1
fi
echo "Fake dataset generated successfully."
echo ""

# Step 2: Preprocess the dataset
echo "Step 2: Preprocessing the dataset..."
python preprocess_sft_data.py \
    --input sft_data/conversations.json \
    --output_dir sft_data/preprocessed \
    --tokenizer GSAI-ML/LLaDA-8B-Base \
    --max_length $MAX_LENGTH

if [ $? -ne 0 ]; then
    echo "Error preprocessing dataset. Exiting."
    exit 1
fi
echo "Dataset preprocessed successfully."
echo ""

# Step 3: Fine-tune the model
echo "Step 3: Fine-tuning the model..."
echo "This step may take a long time depending on your hardware."
python finetune_llada.py \
    --data sft_data/preprocessed/preprocessed_data.pt \
    --tokenizer sft_data/preprocessed/tokenizer \
    --model GSAI-ML/LLaDA-8B-Base \
    --output_dir sft_output \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --use_fp8

if [ $? -ne 0 ]; then
    echo "Error fine-tuning model. Exiting."
    exit 1
fi
echo "Model fine-tuned successfully."
echo ""

# Step 4: Run inference examples
echo "Step 4: Running inference examples..."
python inference_example.py \
    --model sft_output/final_model \
    --tokenizer sft_data/preprocessed/tokenizer \
    --mode examples \
    --steps $STEPS \
    --gen_length $GEN_LENGTH \
    --block_length $BLOCK_LENGTH \
    --use_bf16

if [ $? -ne 0 ]; then
    echo "Error running inference examples. Exiting."
    exit 1
fi
echo "Inference examples completed successfully."
echo ""

# Print completion message
echo "========================================================"
echo "LLaDA Supervised Fine-Tuning Pipeline Completed"
echo "========================================================"
echo ""
echo "The fine-tuned model is saved in: sft_output/final_model"
echo "The best model (based on training loss) is saved in: sft_output/best_model"
echo ""
echo "To run interactive chat with the fine-tuned model, use:"
echo "python inference_example.py --model sft_output/final_model --tokenizer sft_data/preprocessed/tokenizer --mode interactive"
echo ""
echo "To run inference examples with the fine-tuned model, use:"
echo "python inference_example.py --model sft_output/final_model --tokenizer sft_data/preprocessed/tokenizer --mode examples"
echo ""
