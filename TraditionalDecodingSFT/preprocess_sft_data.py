import json
import torch
import argparse
from transformers import AutoTokenizer

def load_conversations(file_path):
    """Load conversations from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['messages']

def preprocess_conversations(conversations, tokenizer, max_length=256):
    """
    Preprocess conversations for LLaDA fine-tuning.
    
    For each conversation:
    1. Format as prompt-response pairs
    2. Handle multi-turn dialogues by treating them as separate pairs with context
    3. Tokenize and prepare for training
    """
    processed_data = []
    
    for conversation in conversations:
        # Process each conversation as individual turns
        for i in range(len(conversation) - 1):
            if i == 0:
                # First turn: just user prompt and assistant response
                user_message = conversation[i]
                assistant_message = conversation[i+1]
                
                # Create chat template
                messages = [{"role": user_message["role"], "content": user_message["content"]}]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                
                # Tokenize prompt
                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
                prompt_length = prompt_ids.shape[0]
                
                # Tokenize response
                response = assistant_message["content"]
                response_ids = tokenizer(response, return_tensors="pt").input_ids[0]
                
                # Combine prompt and response
                combined_ids = torch.cat([prompt_ids, response_ids])
                
                # Ensure we don't exceed max length
                if combined_ids.shape[0] > max_length:
                    continue
                
                processed_data.append({
                    "input_ids": combined_ids,
                    "prompt_length": prompt_length
                })
            else:
                # Multi-turn: include previous context
                context_messages = conversation[:i+1]
                assistant_message = conversation[i+1]
                
                # Create chat template with context
                messages = [{"role": msg["role"], "content": msg["content"]} for msg in context_messages]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                
                # Tokenize prompt with context
                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
                prompt_length = prompt_ids.shape[0]
                
                # Tokenize response
                response = assistant_message["content"]
                response_ids = tokenizer(response, return_tensors="pt").input_ids[0]
                
                # Combine prompt and response
                combined_ids = torch.cat([prompt_ids, response_ids])
                
                # Ensure we don't exceed max length
                if combined_ids.shape[0] > max_length:
                    continue
                
                processed_data.append({
                    "input_ids": combined_ids,
                    "prompt_length": prompt_length
                })
    
    return processed_data

def save_processed_data(processed_data, output_file):
    """Save processed data to a file."""
    torch.save(processed_data, output_file)
    print(f"Saved {len(processed_data)} processed conversations to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess conversation data for LLaDA fine-tuning")
    parser.add_argument("--input_file", type=str, default="sft_data/conversations.json", 
                        help="Path to input JSON file with conversations")
    parser.add_argument("--output_file", type=str, default="sft_data/processed_conversations.pt",
                        help="Path to output processed data file")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="Model name or path for tokenizer")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Load conversations
    print(f"Loading conversations from {args.input_file}...")
    conversations = load_conversations(args.input_file)
    print(f"Loaded {len(conversations)} conversations")
    
    # Preprocess conversations
    print("Preprocessing conversations...")
    processed_data = preprocess_conversations(conversations, tokenizer, args.max_length)
    
    # Save processed data
    save_processed_data(processed_data, args.output_file)

if __name__ == "__main__":
    main()
