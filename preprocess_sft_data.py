import json
import os
import torch
from transformers import AutoTokenizer

def preprocess_sft_data(input_file, output_dir, tokenizer_name="GSAI-ML/LLaDA-8B-Base"):
    """
    Preprocess SFT data according to LLaDA guidelines.
    
    Args:
        input_file: Path to the input JSON file containing conversations
        output_dir: Directory to save the processed data
        tokenizer_name: Name of the tokenizer to use
    """
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load conversations
    print(f"Loading conversations from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process conversations
    processed_data = []
    
    for item in data:
        conversations = item["conversations"]
        
        # Handle single-turn dialogue
        if len(conversations) == 2 and conversations[0]["role"] == "user" and conversations[1]["role"] == "assistant":
            prompt = conversations[0]["content"]
            response = conversations[1]["content"]
            
            # Format as chat template
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            # Tokenize
            prompt_ids = tokenizer(formatted_prompt)['input_ids']
            response_ids = tokenizer(response)['input_ids']
            
            processed_data.append({
                "input_ids": prompt_ids + response_ids,
                "prompt_length": len(prompt_ids)
            })
        
        # Handle multi-turn dialogue
        elif len(conversations) > 2:
            # For multi-turn dialogues, we treat it as multiple single-turn dialogue pairs
            # as described in Appendix B.1
            history = []
            
            for i in range(0, len(conversations), 2):
                if i + 1 < len(conversations):
                    user_msg = conversations[i]["content"]
                    assistant_msg = conversations[i+1]["content"]
                    
                    # Add to history
                    history.append({"role": "user", "content": user_msg})
                    
                    # Format as chat template with history
                    formatted_prompt = tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
                    
                    # Tokenize
                    prompt_ids = tokenizer(formatted_prompt)['input_ids']
                    response_ids = tokenizer(assistant_msg)['input_ids']
                    
                    processed_data.append({
                        "input_ids": prompt_ids + response_ids,
                        "prompt_length": len(prompt_ids)
                    })
                    
                    # Add assistant response to history for next turn
                    history.append({"role": "assistant", "content": assistant_msg})
    
    # Save processed data
    output_file = os.path.join(output_dir, "processed_data.pt")
    print(f"Saving processed data to {output_file}")
    torch.save(processed_data, output_file)
    
    print(f"Processed {len(processed_data)} examples")
    
    return processed_data

if __name__ == "__main__":
    input_file = "sft_data/conversations.json"
    output_dir = "sft_data"
    
    preprocess_sft_data(input_file, output_dir)
