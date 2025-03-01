import json
import argparse
import os
from typing import Dict, List, Any, Tuple
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

def preprocess_conversation(conversation: Dict[str, Any], tokenizer, max_length: int = 2048) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocess a single conversation for LLaDA SFT.
    
    Args:
        conversation: A dictionary containing the conversation data
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        A tuple of (input_ids, prompt_length, total_length)
    """
    # Extract the conversation turns
    # Handle both old and new format
    if "conversations" in conversation:
        # Old format
        turns = conversation["conversations"]
        user_role, assistant_role = "human", "gpt"
        content_key = "value"
    else:
        # New format (list of messages)
        turns = conversation
        user_role, assistant_role = "user", "assistant"
        content_key = "content"
    
    # Format the conversation as a prompt-response pair
    formatted_text = ""
    
    # Process each turn in the conversation
    for i in range(0, len(turns), 2):
        if i + 1 < len(turns):  # Make sure we have a response
            user_msg = turns[i][content_key]
            assistant_msg = turns[i + 1][content_key]
            
            # Format according to LLaDA guidelines
            formatted_text += f"<start_id>user<end_id>\n{user_msg}<eot_id><start_id>assistant<end_id>\n{assistant_msg}"
            
            # Add EOS token after each turn
            formatted_text += "<EOS>"
    
    # Tokenize the formatted text
    tokenized = tokenizer(formatted_text, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]
    
    # Find the position of the first assistant token to determine prompt length
    prompt_text = ""
    for i in range(0, min(2, len(turns))):
        if turns[i].get("role", turns[i].get("from")) == user_role:
            prompt_text += f"<start_id>user<end_id>\n{turns[i][content_key]}<eot_id>"
    
    tokenized_prompt = tokenizer(prompt_text, return_tensors="pt")
    prompt_length = tokenized_prompt["input_ids"].shape[1]
    
    # Calculate the total length (without padding)
    total_length = input_ids.shape[0]
    
    # If the sequence is too long, truncate it
    if total_length > max_length:
        input_ids = input_ids[:max_length]
        total_length = max_length
    
    # If the sequence is too short, pad it with EOS tokens
    if total_length < max_length:
        padding_length = max_length - total_length
        padding = torch.full((padding_length,), tokenizer.eos_token_id, dtype=input_ids.dtype)
        input_ids = torch.cat((input_ids, padding), dim=0)
    
    return input_ids, torch.tensor(prompt_length), torch.tensor(total_length)

def preprocess_dataset(input_file: str, output_dir: str, tokenizer_name: str = "GSAI-ML/LLaDA-8B-Base", max_length: int = 2048):
    """
    Preprocess the entire dataset for LLaDA SFT.
    
    Args:
        input_file: Path to the input JSON file
        output_dir: Directory to save the preprocessed data
        tokenizer_name: Name or path of the tokenizer to use
        max_length: Maximum sequence length
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # Add special tokens if they don't exist
    special_tokens = {
        "additional_special_tokens": [
            "<start_id>", "<end_id>", "<eot_id>", "<EOS>"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Load the dataset
    print(f"Loading dataset from {input_file}")
    with open(input_file, "r") as f:
        data = json.load(f)
    
    # Handle both old and new format
    if isinstance(data, list):
        # Old format: list of conversation objects
        dataset = data
    elif "messages" in data:
        # New format: object with "messages" array
        dataset = data["messages"]
    else:
        raise ValueError(f"Unrecognized data format in {input_file}")
    
    # Preprocess each conversation
    preprocessed_data = []
    print(f"Preprocessing {len(dataset)} conversations")
    for conversation in tqdm(dataset):
        try:
            input_ids, prompt_length, total_length = preprocess_conversation(conversation, tokenizer, max_length)
            
            # For category, use "unknown" as default since new format doesn't have categories
            category = "unknown"
            if isinstance(conversation, dict) and "category" in conversation:
                category = conversation["category"]
            
            preprocessed_data.append({
                "input_ids": input_ids,
                "prompt_length": prompt_length,
                "total_length": total_length,
                "category": category
            })
        except Exception as e:
            print(f"Error preprocessing conversation: {e}")
            continue
    
    # Save the preprocessed data
    output_file = os.path.join(output_dir, "preprocessed_data.pt")
    print(f"Saving {len(preprocessed_data)} preprocessed conversations to {output_file}")
    torch.save(preprocessed_data, output_file)
    
    # Save the tokenizer for later use
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    print(f"Saving tokenizer to {tokenizer_dir}")
    tokenizer.save_pretrained(tokenizer_dir)
    
    print("Preprocessing complete!")

def main():
    parser = argparse.ArgumentParser(description="Preprocess conversation data for LLaDA SFT")
    parser.add_argument("--input", type=str, default="sft_data/transformed_conversations.json", help="Input JSON file")
    parser.add_argument("--output_dir", type=str, default="sft_data/preprocessed", help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="GSAI-ML/LLaDA-8B-Base", help="Tokenizer name or path")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    args = parser.parse_args()
    
    preprocess_dataset(args.input, args.output_dir, args.tokenizer, args.max_length)

if __name__ == "__main__":
    main()
