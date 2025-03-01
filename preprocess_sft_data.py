import json
import os
import torch
from transformers import AutoTokenizer

def preprocess_sft_data(input_file, output_dir, model_name="GSAI-ML/LLaDA-8B-Base", max_length=512):
    """
    Preprocess conversation data for supervised fine-tuning of LLaDA.
    
    Args:
        input_file: Path to the input JSON file containing conversations
        output_dir: Directory to save the preprocessed data
        model_name: Name of the LLaDA model to use for tokenization
        max_length: Maximum sequence length for padding
    """
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load conversation data
    print(f"Loading conversation data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    # Process each conversation
    all_input_ids = []
    all_prompt_lengths = []
    
    for conv in conversations:
        # Extract user prompt and assistant response
        user_prompt = None
        assistant_response = None
        
        for i, message in enumerate(conv['conversations']):
            if message['role'] == 'user':
                user_prompt = message['content']
            elif message['role'] == 'assistant' and user_prompt is not None:
                assistant_response = message['content']
                
                # Format as a chat template
                messages = [{"role": "user", "content": user_prompt}]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                
                # Tokenize prompt and response
                prompt_ids = tokenizer(prompt, return_tensors="pt")['input_ids'][0]
                response_ids = tokenizer(assistant_response, return_tensors="pt")['input_ids'][0]
                
                # Remove BOS token from response if present
                if response_ids[0] == tokenizer.bos_token_id:
                    response_ids = response_ids[1:]
                
                # Add EOS token to response if not present
                if response_ids[-1] != tokenizer.eos_token_id:
                    response_ids = torch.cat([response_ids, torch.tensor([tokenizer.eos_token_id])])
                
                # Combine prompt and response
                input_ids = torch.cat([prompt_ids, response_ids])
                prompt_length = len(prompt_ids)
                
                # Pad or truncate to max_length
                if len(input_ids) > max_length:
                    # Keep prompt intact and truncate response if needed
                    input_ids = torch.cat([prompt_ids, response_ids[:max_length - prompt_length]])
                else:
                    # Pad with EOS tokens
                    padding = torch.tensor([tokenizer.eos_token_id] * (max_length - len(input_ids)))
                    input_ids = torch.cat([input_ids, padding])
                
                all_input_ids.append(input_ids.tolist())
                all_prompt_lengths.append(prompt_length)
                
                # Reset for next turn if this is a multi-turn conversation
                user_prompt = None
    
    # Save preprocessed data
    print(f"Saving {len(all_input_ids)} preprocessed examples...")
    torch.save({
        'input_ids': torch.tensor(all_input_ids),
        'prompt_lengths': torch.tensor(all_prompt_lengths)
    }, os.path.join(output_dir, 'sft_data.pt'))
    
    print("Preprocessing complete!")
    print(f"Total examples: {len(all_input_ids)}")
    print(f"Average prompt length: {sum(all_prompt_lengths) / len(all_prompt_lengths):.2f} tokens")
    print(f"Max prompt length: {max(all_prompt_lengths)} tokens")
    print(f"Min prompt length: {min(all_prompt_lengths)} tokens")

if __name__ == "__main__":
    preprocess_sft_data(
        input_file="sft_data/conversations.json",
        output_dir="sft_data",
        model_name="GSAI-ML/LLaDA-8B-Base",
        max_length=512
    )
