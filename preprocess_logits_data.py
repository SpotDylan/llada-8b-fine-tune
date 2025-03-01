import json
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer

def load_logits_data(file_path):
    """Load data with logits from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_logits_data(data, tokenizer, max_length=4096):
    """
    Preprocess data with logits for LLaDA fine-tuning.
    
    For each example:
    1. Tokenize prompt and response
    2. Combine them into a single sequence
    3. Align LLaMA's logits with the tokenized sequence
    4. Store the processed data
    """
    processed_data = []
    
    for example in data:
        # Extract fields
        prompt = example["prompt"]
        response = example["response"]
        llama_logits = example["logits"]
        llama_tokens = example.get("token", None)  # Optional
        llama_token_ids = example.get("tokenID", None)  # Optional
        
        # Tokenize prompt
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        prompt_length = prompt_ids.shape[0]
        
        # Tokenize response
        response_ids = tokenizer(response, return_tensors="pt").input_ids[0]
        response_length = response_ids.shape[0]
        
        # Combine prompt and response
        combined_ids = torch.cat([prompt_ids, response_ids])
        
        # Ensure we don't exceed max length
        if combined_ids.shape[0] > max_length:
            print(f"Skipping example: combined length {combined_ids.shape[0]} exceeds max_length {max_length}")
            continue
        
        # Convert LLaMA logits to tensor
        # The logits should be a list of arrays, one for each token in the response
        llama_logits_tensor = torch.tensor(llama_logits, dtype=torch.float32)
        
        # Ensure logits match the response length
        if llama_logits_tensor.shape[0] != response_length:
            print(f"Warning: Logits length {llama_logits_tensor.shape[0]} doesn't match response length {response_length}")
            # Skip examples where lengths don't match
            continue
        
        # Process token and tokenID if available
        token_info = None
        token_id_info = None
        
        if llama_tokens is not None:
            # Convert token info to a list of lists of strings
            token_info = llama_tokens
            
            # Verify dimensions match logits
            if len(token_info) != llama_logits_tensor.shape[0]:
                print(f"Warning: Token info length {len(token_info)} doesn't match logits length {llama_logits_tensor.shape[0]}")
                token_info = None
        
        if llama_token_ids is not None:
            # Convert token ID info to a list of lists of integers
            token_id_info = llama_token_ids
            
            # Verify dimensions match logits
            if len(token_id_info) != llama_logits_tensor.shape[0]:
                print(f"Warning: Token ID info length {len(token_id_info)} doesn't match logits length {llama_logits_tensor.shape[0]}")
                token_id_info = None
        
        # Create a tensor to hold all logits (including placeholders for prompt positions)
        # For prompt positions, we'll use zeros as placeholders
        full_logits = torch.zeros((combined_ids.shape[0], llama_logits_tensor.shape[1]), dtype=torch.float32)
        
        # Fill in the logits for response positions
        full_logits[prompt_length:, :] = llama_logits_tensor
        
        # Create the processed example
        processed_example = {
            "input_ids": combined_ids,
            "prompt_length": prompt_length,
            "llama_logits": full_logits
        }
        
        # Add token and token ID info if available
        if token_info is not None:
            processed_example["llama_tokens"] = token_info
        
        if token_id_info is not None:
            processed_example["llama_token_ids"] = token_id_info
        
        processed_data.append(processed_example)
    
    print(f"Processed {len(processed_data)} examples")
    return processed_data

def save_processed_data(processed_data, output_file):
    """Save processed data to a file."""
    torch.save(processed_data, output_file)
    print(f"Saved {len(processed_data)} processed examples to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess logits data for LLaDA fine-tuning")
    parser.add_argument("--input_file", type=str, default="sft_data/llama_logits.json", 
                        help="Path to input JSON file with prompts, responses, and logits")
    parser.add_argument("--output_file", type=str, default="sft_data/processed_logits.pt",
                        help="Path to output processed data file")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="Model name or path for tokenizer")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_logits_data(args.input_file)
    print(f"Loaded {len(data)} examples")
    
    # Preprocess data
    print("Preprocessing data...")
    processed_data = preprocess_logits_data(data, tokenizer, args.max_length)
    
    # Save processed data
    save_processed_data(processed_data, args.output_file)

if __name__ == "__main__":
    main()
