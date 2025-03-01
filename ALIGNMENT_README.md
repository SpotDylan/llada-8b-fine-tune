# LLaDA Alignment Fine-Tuning

This repository contains scripts for fine-tuning LLaDA to align its output and probability distribution with those of the LLaMA 3.3 70B model. This alignment is crucial for using LLaDA as a draft model in speculative decoding with LLaMA 3.3 70B.

## Overview

Unlike traditional LLaDA supervised fine-tuning (SFT), which trained the model to demask tokens and reconstruct responses, this alignment fine-tuning focuses exclusively on aligning LLaDA's probability distributions with LLaMA 3.3 70B's, without any demasking objective. The training uses KL divergence loss to match LLaDA's output distributions with LLaMA's.

## Data Format

The training data is provided in a JSON file containing examples, where each example includes:

- `prompt`: Input text
- `response`: Generated text from LLaMA 3.3 70B
- `logits`: A detailed mapping of the top-5 highest-likelihood tokens, their IDs, and logits for each token in the response

Example JSON format:

```json
{
  "prompt": "<BOS><start_id>user<end_id>\nWhat is the capital of France?<eot_id><start_id>assistant<end_id>\n",
  "response": "Paris.<EOS>",
  "logits": [
    {
      "position": 0,
      "chosen_token": "Paris",
      "chosen_token_id": 4874,
      "top_5": [
        {"token": "Paris", "token_id": 4874, "logit": 14.5000},
        {"token": "France", "token_id": 2763, "logit": 12.7500},
        {"token": "Yes", "token_id": 4378, "logit": 11.2500},
        {"token": "The", "token_id": 791, "logit": 10.8750},
        {"token": "It", "token_id": 1123, "logit": 9.6250}
      ],
      "full_logits": [0.1, 14.5, 2.3, ...]  // Shape: [vocab_size]
    },
    ...
  ]
}
```

A sample file is provided at `sft_data/sample_llama_logits.json`.

## Scripts

### 1. Data Preprocessing

`preprocess_alignment_data.py` processes the JSON data for alignment fine-tuning:

```bash
python preprocess_alignment_data.py \
    --input_file sft_data/llama_logits.json \
    --output_file sft_data/processed_alignment.pt \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --max_length 4096
```

### 2. Alignment Fine-Tuning

`finetune_llada_alignment.py` fine-tunes LLaDA using KL divergence loss:

```bash
python finetune_llada_alignment.py \
    --data_path sft_data/processed_alignment.pt \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --output_dir alignment_output \
    --batch_size 2 \
    --learning_rate 2.5e-5 \
    --weight_decay 0.1 \
    --epochs 3 \
    --warmup_steps 50 \
    --temperature 1.0 \
    --log_interval 100
```

### 3. Complete Pipeline

`run_alignment_fine_tuning.sh` runs the complete alignment fine-tuning pipeline:

```bash
bash run_alignment_fine_tuning.sh
```

## Key Differences from Traditional SFT

1. **Loss Function**: Uses KL divergence instead of cross-entropy to align probability distributions
2. **Training Objective**: Focuses on distribution alignment rather than demasking
3. **Data Format**: Includes full logit arrays from LLaMA 3.3 70B for each response token
4. **Application**: Optimized for speculative decoding as a draft model

## Technical Details

### Data Loading and Preprocessing

- Loads each example's "prompt" and "response" and concatenates them into a single sequence
- Tokenizes the sequence using LLaDA's tokenizer
- Constructs `llama_logits` as a tensor of shape [sequence_length, vocab_size]
- For prompt tokens, sets logits to zeros (no logits provided)
- For response tokens, sets logits to the "full_logits" array from the JSON

### Loss Calculation

- Generates LLaDA's logits using its diffusion process
- Computes KL divergence over all response positions
- Normalizes the loss by the number of response tokens

### Temperature Scaling

- Applies temperature scaling to both LLaDA and LLaMA logits before computing KL divergence
- Adjustable temperature parameter (default: 1.0)

## Usage for Speculative Decoding

After fine-tuning, the aligned LLaDA model can be used as a draft model for speculative decoding with LLaMA 3.3 70B. This approach allows for faster inference by:

1. Using LLaDA to generate multiple tokens in parallel
2. Verifying these tokens with LLaMA 3.3 70B
3. Accepting tokens that match LLaMA's distribution
4. Rejecting and regenerating tokens that don't match

The alignment fine-tuning ensures high acceptance rates in this process, making the speculative decoding more efficient.
