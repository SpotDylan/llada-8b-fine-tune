#!/bin/bash

# Enable memory optimization to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Force the use of GPU 1 by setting CUDA_VISIBLE_DEVICES
# This makes GPU 1 appear as device 0 to the process
export CUDA_VISIBLE_DEVICES=1
echo "Set CUDA_VISIBLE_DEVICES=1 (GPU 1 will appear as device 0 to the process)"

# Run the fine-tuning script with GPU ID 0 (which is actually GPU 1)
python finetune_llada.py --gpu_id 0
