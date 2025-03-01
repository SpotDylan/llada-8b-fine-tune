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
    
    # New format has a "messages" array where each element is a conversation
    conversations = data.get("messages", [])
    print(f"Found {len(conversations)} conversations")
    
    for conversation in conversations:
        # Handle single-turn dialogue
        if len(conversation) == 2 and conversation[0]["role"] == "user" and conversation[1]["role"] == "assistant":
            prompt = conversation[0]["content"]
            response = conversation[1]["content"]
            
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
        elif len(conversation) > 2:
            # For multi-turn dialogues, we treat it as multiple single-turn dialogue pairs
            # as described in Appendix B.1
            history = []
            
            for i in range(0, len(conversation), 2):
                if i + 1 < len(conversation):
                    user_msg = conversation[i]["content"]
                    assistant_msg = conversation[i+1]["content"]
                    
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
    
    # Also save a transformed version of the conversations in the original format for compatibility
    transformed_conversations = []
    for conversation in conversations:
        transformed_conversations.append({
            "conversations": conversation
        })
    
    transformed_file = os.path.join(output_dir, "transformed_conversations.json")
    print(f"Saving transformed conversations to {transformed_file}")
    with open(transformed_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_conversations, f, indent=2)
    
    return processed_data

if __name__ == "__main__":
    input_file = "sft_data/conversations.json"
    output_dir = "sft_data"
    
    preprocess_sft_data(input_file, output_dir)
