#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocess conversation data for supervised fine-tuning of LLaDA.
This script converts conversation data from JSON format to the format required for SFT.
"""

import json
import argparse
import os
import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer

# Special tokens
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
START_ID_TOKEN = "<start_id>"
END_ID_TOKEN = "<end_id>"
EOT_ID_TOKEN = "<eot_id>"
MASK_TOKEN_ID = 126336  # As specified in the guidelines

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess conversation data for LLaDA SFT")
    parser.add_argument("--input_file", type=str, default="sft_data/conversations.json",
                        help="Path to the input JSON file containing conversations")
    parser.add_argument("--output_dir", type=str, default="sft_data/processed",
                        help="Directory to save the processed data")
    parser.add_argument("--model_name_or_path", type=str, default="llada-8b",
                        help="Path to the LLaDA model or model name")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--pad_to_max_length", action="store_true",
                        help="Whether to pad all samples to max_seq_length")
    return parser.parse_args()

def format_conversation(conversation, tokenizer):
    """
    Format a conversation into the required format for LLaDA SFT.
    
    Format: <BOS><start_id>user<end_id>\nPrompt<eot_id><start_id>assistant<end_id>\nResponse<EOS>
    """
    formatted_text = BOS_TOKEN
    
    for i, message in enumerate(conversation):
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            formatted_text += f"{START_ID_TOKEN}user{END_ID_TOKEN}\n{content}"
            # Add EOT token if there's a next message and it's from the assistant
            if i + 1 < len(conversation) and conversation[i + 1]["role"] == "assistant":
                formatted_text += f"{EOT_ID_TOKEN}"
        elif role == "assistant":
            formatted_text += f"{START_ID_TOKEN}assistant{END_ID_TOKEN}\n{content}"
            # Add EOS token at the end of assistant's message
            formatted_text += EOS_TOKEN
    
    return formatted_text

def preprocess_conversations(args):
    """Preprocess conversations from JSON file for SFT."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Load conversations
    with open(args.input_file, 'r', encoding='utf-8') as f:
        conversations_data = json.load(f)
    
    # Process each conversation
    all_input_ids = []
    all_prompt_lengths = []
    
    for conv_data in tqdm(conversations_data, desc="Processing conversations"):
        conversation = conv_data["conversations"]
        
        # Format the conversation
        formatted_text = format_conversation(conversation, tokenizer)
        
        # Tokenize
        tokenized = tokenizer(formatted_text, truncation=True, max_length=args.max_seq_length)
        input_ids = tokenized["input_ids"]
        
        # Calculate prompt length (everything up to the assistant's response)
        prompt_text = ""
        for i, message in enumerate(conversation):
            if message["role"] == "user":
                prompt_text += f"{START_ID_TOKEN}user{END_ID_TOKEN}\n{message['content']}"
                if i + 1 < len(conversation) and conversation[i + 1]["role"] == "assistant":
                    prompt_text += f"{EOT_ID_TOKEN}"
        
        prompt_tokenized = tokenizer(prompt_text, truncation=True, max_length=args.max_seq_length)
        prompt_length = len(prompt_tokenized["input_ids"])
        
        # Add BOS token to prompt length
        prompt_length += 1  # +1 for BOS token
        
        # Pad with EOS tokens if needed
        if args.pad_to_max_length and len(input_ids) < args.max_seq_length:
            padding_length = args.max_seq_length - len(input_ids)
            input_ids.extend([tokenizer.eos_token_id] * padding_length)
        
        all_input_ids.append(input_ids)
        all_prompt_lengths.append(prompt_length)
    
    # Save processed data
    torch.save({
        "input_ids": all_input_ids,
        "prompt_lengths": all_prompt_lengths
    }, os.path.join(args.output_dir, "sft_data.pt"))
    
    print(f"Processed {len(all_input_ids)} conversations.")
    print(f"Data saved to {os.path.join(args.output_dir, 'sft_data.pt')}")
    
    # Save a sample for inspection
    with open(os.path.join(args.output_dir, "sample.txt"), 'w', encoding='utf-8') as f:
        sample_idx = random.randint(0, len(all_input_ids) - 1)
        f.write(f"Sample conversation {sample_idx}:\n")
        f.write(f"Input IDs: {all_input_ids[sample_idx]}\n")
        f.write(f"Prompt Length: {all_prompt_lengths[sample_idx]}\n")
        
        # Decode the input IDs for better readability
        decoded_text = tokenizer.decode(all_input_ids[sample_idx])
        f.write(f"Decoded Text: {decoded_text}\n")

if __name__ == "__main__":
    args = parse_args()
    preprocess_conversations(args)
