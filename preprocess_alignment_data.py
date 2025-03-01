import json
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def load_alignment_data(file_path):
    """Load alignment data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_alignment_data(data, tokenizer, max_length=4096):
    """
    Preprocess alignment data for LLaDA fine-tuning.
    
    For each example:
    1. Tokenize the prompt and response
    2. Extract the logits for each token in the response
    3. Prepare the data for training with KL divergence loss
    """
    processed_data = []
    
    for example in tqdm(data, desc="Processing examples"):
        # Get prompt and response
        prompt = example["prompt"]
        response = example["response"]
        logits_data = example["logits"]
        
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
            print(f"Skipping example with length {combined_ids.shape[0]} > {max_length}")
            continue
        
        # Extract logits for each token in the response
        # Initialize a tensor to store the full logits for each response token
        vocab_size = len(tokenizer)
        llama_logits = torch.zeros((combined_ids.shape[0], vocab_size), dtype=torch.float32)
        
        # Verify that the number of logits entries matches the response length
        if len(logits_data) != response_length:
            print(f"Warning: Mismatch between response length ({response_length}) and logits data length ({len(logits_data)})")
            # Instead of skipping, we'll use the minimum length to avoid losing examples
            # This ensures we only process tokens that have corresponding logits
            response_length = min(response_length, len(logits_data))
            response_ids = response_ids[:response_length]
        
        # Fill in the logits for the response tokens
        for i, logit_entry in enumerate(logits_data[:response_length]):
            position = prompt_length + i
            
            # Verify token consistency
            token_id = logit_entry["chosen_token_id"]
            if token_id != response_ids[i].item():
                print(f"Warning: Token ID mismatch at position {i}. Expected {response_ids[i].item()}, got {token_id}")
            
            # Extract full logits array
            if "full_logits" in logit_entry:
                # Convert the full logits array to a tensor
                full_logits = torch.tensor(logit_entry["full_logits"], dtype=torch.float32)
                llama_logits[position] = full_logits
            else:
                # If full_logits is not available, use top_5 to create a sparse logits tensor
                # Initialize with a very low value (effectively zero probability)
                sparse_logits = torch.full((vocab_size,), -100.0, dtype=torch.float32)
                
                # Fill in the top-5 logits
                for top_entry in logit_entry["top_5"]:
                    token_id = top_entry["token_id"]
                    logit_value = top_entry["logit"]
                    sparse_logits[token_id] = logit_value
                
                llama_logits[position] = sparse_logits
        
        # Store the processed data
        processed_data.append({
            "input_ids": combined_ids,
            "prompt_length": prompt_length,
            "llama_logits": llama_logits
        })
    
    return processed_data

def save_processed_data(processed_data, output_file):
    """Save processed data to a file."""
    torch.save(processed_data, output_file)
    print(f"Saved {len(processed_data)} processed examples to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess alignment data for LLaDA fine-tuning")
    parser.add_argument("--input_file", type=str, default="sft_data/llama_logits.json", 
                        help="Path to input JSON file with alignment data")
    parser.add_argument("--output_file", type=str, default="sft_data/processed_alignment.pt",
                        help="Path to output processed data file")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="Model name or path for tokenizer")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Load alignment data
    print(f"Loading alignment data from {args.input_file}...")
    data = load_alignment_data(args.input_file)
    print(f"Loaded {len(data)} examples")
    
    # Preprocess alignment data
    print("Preprocessing alignment data...")
    processed_data = preprocess_alignment_data(data, tokenizer, args.max_length)
    
    # Save processed data
    save_processed_data(processed_data, args.output_file)

if __name__ == "__main__":
    main()
