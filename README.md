# LLaDA-8B Fine-Tuning

This repository contains code and sample data for supervised fine-tuning of the LLaDA-8B model. LLaDA (Large Language Diffusion with mAsking) is a diffusion-based language model that offers an alternative to traditional autoregressive models.

## Repository Structure

- `sft_data/conversations.json`: Sample conversation data for fine-tuning
- `preprocess_sft_data.py`: Script to preprocess the conversation data
- `finetune_llada.py`: Main script for fine-tuning LLaDA
- `inference_example.py`: Script to test the fine-tuned model
- `run_fine_tuning.sh`: Shell script to run the entire fine-tuning pipeline

## How to Use

1. **Prepare Your Data**: 
   - Place your conversation data in the `sft_data/conversations.json` file
   - The data should follow the format in the sample file

2. **Run the Fine-Tuning Pipeline**:
   ```bash
   chmod +x run_fine_tuning.sh
   ./run_fine_tuning.sh
   ```

3. **Customize Fine-Tuning Parameters**:
   - Edit `run_fine_tuning.sh` to adjust parameters like model name, batch size, learning rate, etc.

## Fine-Tuning Process

The fine-tuning process follows the guidelines from the LLaDA paper:

1. **Data Preprocessing**:
   - Format data as prompt-response pairs
   - Handle multi-turn dialogues
   - Pad with EOS tokens for equal lengths

2. **Forward Process**:
   - Apply noise only to the response part
   - Keep the prompt unchanged

3. **Loss Calculation**:
   - Calculate loss only on masked tokens in the response
   - Normalize by answer length

4. **Sampling Strategies**:
   - Semi-autoregressive sampling with low-confidence remasking
   - Divide generation into blocks for better control

## Requirements

- PyTorch
- Transformers (version 4.38.2 or later)
- SetFit (version 1.0.0 or later)
- Sentence-Transformers (version 2.2.2 or later)
- CUDA-capable GPU (recommended)

## Implementation Details

The fine-tuning process has been implemented using the Hugging Face SetFit trainer, which provides:

1. **Simplified Training Loop**:
   - Streamlined training process with built-in support for callbacks
   - Automatic handling of training phases and optimization

2. **Distributed Training**:
   - Multi-GPU support for faster training
   - Efficient data parallelism

3. **Forward Process Integration**:
   - Custom callback for LLaDA's forward process
   - Maintains the same noise application strategy as the original implementation

## Reference

For more details on LLaDA, refer to the [original paper](https://arxiv.org/abs/2502.09992) and the [official repository](https://github.com/ML-GSAI/LLaDA).
