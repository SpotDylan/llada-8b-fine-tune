#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for inference with a fine-tuned LLaDA model.
This script demonstrates how to use the fine-tuned model for generating responses.
"""

import argparse
import torch
import logging
import random
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Constants
MASK_TOKEN_ID = 126336  # As specified in the guidelines
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
START_ID_TOKEN = "<start_id>"
END_ID_TOKEN = "<end_id>"
EOT_ID_TOKEN = "<eot_id>"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inference with fine-tuned LLaDA model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="models/llada-sft",
                        help="Path to the fine-tuned model")
    
    # Generation arguments
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum length of generated text")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Number of sequences to return")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--remask_strategy", type=str, default="low_confidence",
                        choices=["random", "low_confidence"],
                        help="Strategy for remasking tokens during generation")
    parser.add_argument("--remask_ratio", type=float, default=0.5,
                        help="Ratio of tokens to remask in each iteration")
    parser.add_argument("--sampling_method", type=str, default="semi_autoregressive_padding",
                        choices=["fixed_length", "semi_autoregressive_origin", "semi_autoregressive_padding"],
                        help="Sampling method to use")
    parser.add_argument("--block_size", type=int, default=64,
                        help="Block size for semi-autoregressive sampling")
    
    # Other arguments
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt for non-interactive mode")
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_prompt(prompt, tokenizer):
    """
    Format a prompt for LLaDA.
    
    Format: <BOS><start_id>user<end_id>\nPrompt<eot_id><start_id>assistant<end_id>\n
    """
    formatted_text = f"{BOS_TOKEN}{START_ID_TOKEN}user{END_ID_TOKEN}\n{prompt}{EOT_ID_TOKEN}{START_ID_TOKEN}assistant{END_ID_TOKEN}\n"
    return formatted_text

def generate_response(model, tokenizer, prompt, args, device):
    """
    Generate a response using the LLaDA model.
    
    Args:
        model: The LLaDA model.
        tokenizer: Tokenizer for the model.
        prompt: User prompt.
        args: Generation arguments.
        device: Device to use for generation.
        
    Returns:
        Generated response.
    """
    # Format the prompt
    formatted_prompt = format_prompt(prompt, tokenizer)
    
    # Tokenize the prompt
    prompt_tokens = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    prompt_length = prompt_tokens.input_ids.shape[1]
    
    # Initialize the sequence with the prompt
    if args.sampling_method == "fixed_length":
        # Fixed-length sampling
        sequence = torch.full(
            (1, args.max_length), 
            tokenizer.pad_token_id, 
            dtype=torch.long, 
            device=device
        )
        sequence[:, :prompt_length] = prompt_tokens.input_ids
        
        # Initialize mask for tokens to predict
        mask = torch.zeros_like(sequence, dtype=torch.bool)
        mask[:, prompt_length:] = True
        
        # Generate tokens iteratively
        with torch.no_grad():
            for _ in tqdm(range(10), desc="Generating"):  # Fixed number of iterations
                # Forward pass for masked positions
                masked_sequence = sequence.clone()
                masked_sequence[mask] = MASK_TOKEN_ID
                
                outputs = model(input_ids=masked_sequence)
                logits = outputs.logits
                
                # Sample from the distribution for masked tokens
                probs = torch.softmax(logits[mask] / args.temperature, dim=-1)
                
                if args.top_p > 0:
                    # Top-p sampling
                    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > args.top_p
                    # Keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create a mask for indices to keep
                    indices_to_keep = torch.zeros_like(probs, dtype=torch.bool)
                    for i in range(probs.shape[0]):
                        indices = sorted_indices[i][~sorted_indices_to_remove[i]]
                        indices_to_keep[i, indices] = True
                    
                    # Zero out removed indices
                    probs[~indices_to_keep] = 0
                    
                    # Renormalize
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sample from the filtered distribution
                next_tokens = torch.multinomial(probs, 1).squeeze(-1)
                
                # Update sequence
                sequence[mask] = next_tokens
                
                # Remask tokens based on strategy
                if args.remask_strategy == "random":
                    # Randomly remask tokens
                    num_to_remask = int(mask.sum().item() * args.remask_ratio)
                    remask_indices = torch.randperm(mask.sum().item())[:num_to_remask]
                    
                    # Create a new mask
                    new_mask = torch.zeros_like(mask)
                    masked_positions = torch.nonzero(mask, as_tuple=True)
                    for idx in remask_indices:
                        new_mask[masked_positions[0][idx], masked_positions[1][idx]] = True
                    
                    mask = new_mask
                else:  # low_confidence
                    # Remask tokens with lowest confidence
                    token_probs = torch.gather(probs, 1, next_tokens.unsqueeze(-1)).squeeze(-1)
                    num_to_remask = int(mask.sum().item() * args.remask_ratio)
                    
                    # Sort by confidence (probability)
                    _, sorted_indices = torch.sort(token_probs)
                    remask_indices = sorted_indices[:num_to_remask]
                    
                    # Create a new mask
                    new_mask = torch.zeros_like(mask)
                    masked_positions = torch.nonzero(mask, as_tuple=True)
                    for idx in remask_indices:
                        new_mask[masked_positions[0][idx], masked_positions[1][idx]] = True
                    
                    mask = new_mask
                
                # Check if we've reached the EOS token
                if tokenizer.eos_token_id in sequence[0, prompt_length:]:
                    break
    
    elif args.sampling_method == "semi_autoregressive_origin":
        # Semi-autoregressive sampling (origin method)
        max_new_tokens = args.max_length - prompt_length
        
        # Start with just the prompt
        sequence = prompt_tokens.input_ids
        
        # Generate tokens block by block
        for block_start in tqdm(range(0, max_new_tokens, args.block_size), desc="Generating blocks"):
            block_size = min(args.block_size, max_new_tokens - block_start)
            
            # Extend sequence with mask tokens
            block_masks = torch.full((1, block_size), MASK_TOKEN_ID, dtype=torch.long, device=device)
            extended_sequence = torch.cat([sequence, block_masks], dim=1)
            
            # Initialize mask for the new block
            mask = torch.zeros_like(extended_sequence, dtype=torch.bool)
            mask[:, -block_size:] = True
            
            # Generate tokens for this block
            for _ in range(10):  # Fixed number of iterations per block
                # Forward pass
                outputs = model(input_ids=extended_sequence)
                logits = outputs.logits
                
                # Sample from the distribution for masked tokens
                probs = torch.softmax(logits[mask] / args.temperature, dim=-1)
                
                if args.top_p > 0:
                    # Apply top-p sampling
                    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > args.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_keep = torch.zeros_like(probs, dtype=torch.bool)
                    for i in range(probs.shape[0]):
                        indices = sorted_indices[i][~sorted_indices_to_remove[i]]
                        indices_to_keep[i, indices] = True
                    
                    probs[~indices_to_keep] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sample from the filtered distribution
                next_tokens = torch.multinomial(probs, 1).squeeze(-1)
                
                # Update sequence
                extended_sequence[mask] = next_tokens
                
                # Remask tokens based on strategy
                if args.remask_strategy == "random":
                    # Randomly remask tokens
                    num_to_remask = int(mask.sum().item() * args.remask_ratio)
                    remask_indices = torch.randperm(mask.sum().item())[:num_to_remask]
                    
                    new_mask = torch.zeros_like(mask)
                    masked_positions = torch.nonzero(mask, as_tuple=True)
                    for idx in remask_indices:
                        new_mask[masked_positions[0][idx], masked_positions[1][idx]] = True
                    
                    mask = new_mask
                else:  # low_confidence
                    # Remask tokens with lowest confidence
                    token_probs = torch.gather(probs, 1, next_tokens.unsqueeze(-1)).squeeze(-1)
                    num_to_remask = int(mask.sum().item() * args.remask_ratio)
                    
                    _, sorted_indices = torch.sort(token_probs)
                    remask_indices = sorted_indices[:num_to_remask]
                    
                    new_mask = torch.zeros_like(mask)
                    masked_positions = torch.nonzero(mask, as_tuple=True)
                    for idx in remask_indices:
                        new_mask[masked_positions[0][idx], masked_positions[1][idx]] = True
                    
                    mask = new_mask
                
                # Check if all tokens in the block are EOS
                if (extended_sequence[0, -block_size:] == tokenizer.eos_token_id).all():
                    break
            
            # Update sequence for next block
            sequence = extended_sequence
            
            # Check if we've reached the EOS token
            if tokenizer.eos_token_id in sequence[0, prompt_length:]:
                break
    
    else:  # semi_autoregressive_padding
        # Semi-autoregressive sampling with padding
        max_new_tokens = args.max_length - prompt_length
        
        # Initialize sequence with prompt and padding
        sequence = torch.full(
            (1, prompt_length + max_new_tokens), 
            tokenizer.pad_token_id, 
            dtype=torch.long, 
            device=device
        )
        sequence[:, :prompt_length] = prompt_tokens.input_ids
        
        # Add EOS tokens as padding
        sequence[:, prompt_length:] = tokenizer.eos_token_id
        
        # Generate tokens block by block
        for block_start in tqdm(range(0, max_new_tokens, args.block_size), desc="Generating blocks"):
            block_size = min(args.block_size, max_new_tokens - block_start)
            block_end = prompt_length + block_start + block_size
            
            # Initialize mask for the current block
            mask = torch.zeros_like(sequence, dtype=torch.bool)
            mask[:, prompt_length + block_start:block_end] = True
            
            # Replace block with mask tokens
            masked_sequence = sequence.clone()
            masked_sequence[mask] = MASK_TOKEN_ID
            
            # Generate tokens for this block
            for _ in range(10):  # Fixed number of iterations per block
                # Forward pass
                outputs = model(input_ids=masked_sequence)
                logits = outputs.logits
                
                # Sample from the distribution for masked tokens
                probs = torch.softmax(logits[mask] / args.temperature, dim=-1)
                
                if args.top_p > 0:
                    # Apply top-p sampling
                    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > args.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_keep = torch.zeros_like(probs, dtype=torch.bool)
                    for i in range(probs.shape[0]):
                        indices = sorted_indices[i][~sorted_indices_to_remove[i]]
                        indices_to_keep[i, indices] = True
                    
                    probs[~indices_to_keep] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sample from the filtered distribution
                next_tokens = torch.multinomial(probs, 1).squeeze(-1)
                
                # Update sequence
                masked_sequence[mask] = next_tokens
                
                # Remask tokens based on strategy
                if args.remask_strategy == "random":
                    # Randomly remask tokens
                    num_to_remask = int(mask.sum().item() * args.remask_ratio)
                    remask_indices = torch.randperm(mask.sum().item())[:num_to_remask]
                    
                    new_mask = torch.zeros_like(mask)
                    masked_positions = torch.nonzero(mask, as_tuple=True)
                    for idx in remask_indices:
                        new_mask[masked_positions[0][idx], masked_positions[1][idx]] = True
                    
                    mask = new_mask
                else:  # low_confidence
                    # Remask tokens with lowest confidence
                    token_probs = torch.gather(probs, 1, next_tokens.unsqueeze(-1)).squeeze(-1)
                    num_to_remask = int(mask.sum().item() * args.remask_ratio)
                    
                    _, sorted_indices = torch.sort(token_probs)
                    remask_indices = sorted_indices[:num_to_remask]
                    
                    new_mask = torch.zeros_like(mask)
                    masked_positions = torch.nonzero(mask, as_tuple=True)
                    for idx in remask_indices:
                        new_mask[masked_positions[0][idx], masked_positions[1][idx]] = True
                    
                    mask = new_mask
            
            # Update sequence with generated tokens
            sequence[:, prompt_length + block_start:block_end] = masked_sequence[:, prompt_length + block_start:block_end]
            
            # Check if we've reached a real EOS token (not padding)
            eos_pos = (sequence[0, prompt_length:prompt_length + block_start + block_size] == tokenizer.eos_token_id).nonzero()
            if len(eos_pos) > 0:
                # Keep only up to the first EOS token
                first_eos = eos_pos[0].item() + prompt_length
                sequence = sequence[:, :first_eos + 1]
                break
    
    # Decode the generated sequence
    output = tokenizer.decode(sequence[0, prompt_length:], skip_special_tokens=True)
    
    # Remove any remaining EOS tokens from the output
    output = output.replace(tokenizer.eos_token, "")
    
    return output.strip()

def main():
    """Main function for inference."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    if args.interactive:
        logger.info("Running in interactive mode. Type 'exit' to quit.")
        while True:
            prompt = input("\nUser: ")
            if prompt.lower() == "exit":
                break
            
            logger.info("Generating response...")
            response = generate_response(model, tokenizer, prompt, args, device)
            print(f"\nAssistant: {response}")
    else:
        if args.prompt is None:
            logger.error("Please provide a prompt using --prompt or use --interactive mode.")
            return
        
        logger.info("Generating response...")
        response = generate_response(model, tokenizer, args.prompt, args, device)
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
