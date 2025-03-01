# Fine-tuning LLaDA with KL Divergence Loss

This document explains how to fine-tune LLaDA to align its output and probability distribution with those of the LLaMA 3.3 70B model using KL divergence loss. The goal is to adapt LLaDA to serve as a draft model in speculative decoding.

## Overview

The fine-tuning process has been modified to support training with logits from LLaMA 3.3 70B. The key changes include:

1. A new preprocessing script for handling logits data
2. Support for KL divergence loss to align probability distributions
3. Options for hybrid loss combining cross-entropy and KL divergence
4. Temperature softening for probability distributions

## Data Preparation

To fine-tune LLaDA with KL divergence, you need to prepare a JSON file containing examples with prompts, responses, and logits from LLaMA 3.3 70B.

### JSON Format

Each entry in the JSON file should have the following structure:

```json
{
  "prompt": "<BOS><start_id>user<end_id>\nWhat is the capital of France?<eot_id><start_id>assistant<end_id>\n",
  "response": "Paris.<EOS>",
  "logits": [[logit_for_P, logit_for_a, ...], [logit_for_a, logit_for_r, ...], ..., [logit_for_<EOS>, ...]],
  "token": [["P", "a", "r", "i", "s"], ["a", "r", "i", "s", "."], ...],
  "tokenID": [[1000, 1001, 1002, 1003, 1004], [1001, 1002, 1003, 1004, 1005], ...]
}
```

- **prompt**: The input text, including any special tokens used by LLaMA
- **response**: The generated text from LLaMA 3.3 70B
- **logits**: A list of arrays, where each array has shape `[vocab_size]` and represents the logits for the next token in the response
- **token** (optional): A list of arrays containing the corresponding tokens for each position in the logits
- **tokenID** (optional): A list of arrays containing the corresponding token IDs for each position in the logits

See `sft_data/llama_logits_example.json` for a complete example.

### Collecting Logits from LLaMA 3.3 70B

To collect logits from LLaMA 3.3 70B, you'll need to:

1. Run inference with LLaMA 3.3 70B on your prompts
2. Capture the logits for each token in the generated response
3. Format the data as described above

## Running the Fine-tuning Pipeline

The fine-tuning pipeline has been updated to support both standard cross-entropy training and KL divergence training.

### Configuration

Edit `run_fine_tuning.sh` to configure the training process:

```bash
# Choose which data to use (conversations or logits)
USE_LOGITS=true

# Standard SFT data
INPUT_FILE="sft_data/conversations.json"
PROCESSED_FILE="sft_data/processed_conversations.pt"

# Logits data for KL divergence
LOGITS_FILE="sft_data/llama_logits.json"
PROCESSED_LOGITS_FILE="sft_data/processed_logits.pt"

# KL divergence parameters
LOSS_TYPE="auto"  # auto, ce, kl, or hybrid
TEMPERATURE=1.0
ALPHA=0.5  # Weight for CE loss in hybrid mode
BETA=0.5   # Weight for KL loss in hybrid mode
```

### Loss Types

- **auto**: Automatically select KL divergence if logits are available, otherwise use cross-entropy
- **ce**: Use only cross-entropy loss (standard SFT)
- **kl**: Use only KL divergence loss
- **hybrid**: Use a weighted combination of cross-entropy and KL divergence losses

### Temperature

The temperature parameter controls how "soft" the probability distributions are for KL divergence calculation. Higher values (e.g., 2.0) make the distributions more uniform, while lower values (e.g., 0.5) make them more peaked.

### Running the Pipeline

```bash
chmod +x run_fine_tuning.sh
./run_fine_tuning.sh
```

The script will:
1. Preprocess the data (either standard conversations or logits data)
2. Fine-tune the model with the specified loss type
3. Test the fine-tuned model

## Monitoring Training

Training progress is logged to the console and to `training_log.log`. The logs include:

- Loss values for each batch
- Memory usage
- Gradient statistics
- Training time

## Using the Fine-tuned Model

After fine-tuning, the model will be saved to `sft_output/final`. You can use this model as a draft model for speculative decoding with LLaMA 3.3 70B.

## Troubleshooting

### Common Issues

- **Mismatched logits and response lengths**: Ensure that the number of logit arrays matches the number of tokens in the response
- **Out of memory errors**: Try reducing the batch size or using gradient accumulation
- **NaN or inf values in loss**: Check for extreme values in the logits or try adjusting the temperature

### Debugging

For more detailed logging, set the `--log_level` parameter to `DEBUG` in `run_fine_tuning.sh`:

```bash
python finetune_llada.py \
    --data_path $DATA_PATH \
    --log_level DEBUG \
    # other parameters...
