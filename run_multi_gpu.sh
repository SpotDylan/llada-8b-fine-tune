#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -g, --gpu-id <id>          Specify GPU ID to use (default: 0)"
    echo "  -l, --list-gpus            List available GPUs"
    echo "  -b, --batch-size <size>    Set batch size (default: 4)"
    echo "  -e, --epochs <num>         Set number of epochs (default: 3)"
    echo "  -o, --output-dir <dir>     Set output directory (default: sft_output)"
    echo "  -m, --model <name/path>    Set model name/path (default: GSAI-ML/LLaDA-8B-Base)"
    echo ""
    echo "Example:"
    echo "  $0 --gpu-id 1 --batch-size 2 --epochs 5"
    exit 0
}

# Function to list available GPUs
list_gpus() {
    echo "Checking available GPUs..."
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU information:"
        nvidia-smi --list-gpus
        echo ""
        echo "GPU memory usage:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
    else
        echo "nvidia-smi command not found. Unable to list GPUs."
    fi
    exit 0
}

# Enable memory optimization to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default values
GPU_ID=0
MODEL_NAME="GSAI-ML/LLaDA-8B-Base"
OUTPUT_DIR="sft_output"
BATCH_SIZE=4
NUM_EPOCHS=3
SAVE_STEPS=500
LEARNING_RATE=2.5e-5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            ;;
        -g|--gpu-id)
            GPU_ID="$2"
            shift
            shift
            ;;
        -l|--list-gpus)
            list_gpus
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        -e|--epochs)
            NUM_EPOCHS="$2"
            shift
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        -m|--model)
            MODEL_NAME="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Create output directory
mkdir -p $OUTPUT_DIR

# Display configuration
echo "===== Configuration ====="
echo "GPU ID: $GPU_ID"
echo "Model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Number of epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "========================="

# Check if the specified GPU is available
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_ID" -ge "$NUM_GPUS" ]; then
        echo "Error: GPU ID $GPU_ID is not available. Available GPUs: 0-$((NUM_GPUS-1))"
        exit 1
    fi
    
    echo "Using GPU $GPU_ID: $(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU_ID)"
else
    echo "Warning: nvidia-smi command not found. Unable to verify GPU availability."
fi

# Set CUDA_VISIBLE_DEVICES to ensure only the selected GPU is used
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Set CUDA_VISIBLE_DEVICES=$GPU_ID"

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
    --gpu_id 0  # Use 0 since we're setting CUDA_VISIBLE_DEVICES

echo "===== Step 3: Testing fine-tuned model ====="
python inference_example.py \
    --model_path "$OUTPUT_DIR/final" \
    --steps 128 \
    --gen_length 128 \
    --block_length 32 \
    --temperature 0.0 \
    --cfg_scale 0.0 \
    --remasking "low_confidence" \
    --gpu_id 0  # Use 0 since we're setting CUDA_VISIBLE_DEVICES

echo "===== Fine-tuning pipeline completed ====="
