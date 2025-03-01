#!/usr/bin/env python3
import os
import argparse
import csv
import json
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
import torch
import tqdm
import pandas as pd
import time
from torch.nn.attention import SDPBackend, sdpa_kernel
# Only enable flash attention backend

# Comment out or modify this line if the path doesn't exist
os.environ['HF_HOME'] = '/mount/model-cache'

def main():
    print(torch.backends.cuda.flash_sdp_enabled())
    # True
    print(torch.backends.cuda.mem_efficient_sdp_enabled())
    # True
    print(torch.backends.cuda.math_sdp_enabled())
    # True
    
    parser = argparse.ArgumentParser(description="Generate model outputs")
    parser.add_argument("--model_name", type=str, default="casperhansen/Llama-3.3-70B-instruct-awq", help="Hugging Face model name or path")
    parser.add_argument("--dataset_path", type=str, default='limo_data/limo_train.parquet', help="Path to the dataset file in parquet format")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--name", type=str, default="llama70b", help="Model string name")
    parser.add_argument("--max_length", type=int, default=2080, help="Max generation length")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for processing prompts")
    args = parser.parse_args()

    # Prepare the results DataFrame
    results = []
    # Load model and tokenizer
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use flash attention if available
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if torch.backends.cuda.flash_sdp_enabled() else torch.no_grad():
        # Load model with a more direct approach to bypass the rope_scaling issue
        try:
            # First try loading with default settings
            model = LlamaForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
        except ValueError as e:
            if 'rope_scaling' in str(e):
                print("Encountered rope_scaling error, trying alternative loading method...")
                # If we get a rope_scaling error, try loading with a custom config
                # that ignores the problematic rope_scaling
                from huggingface_hub import hf_hub_download
                import json
                
                # Download the config file
                try:
                    config_path = hf_hub_download(args.model_name, "config.json")
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    
                    # Remove or fix the rope_scaling
                    if 'rope_scaling' in config_dict:
                        # Option 1: Remove rope_scaling completely
                        # del config_dict['rope_scaling']
                        
                        # Option 2: Fix rope_scaling to have only type and factor
                        config_dict['rope_scaling'] = {
                            'type': 'linear',
                            'factor': config_dict['rope_scaling'].get('factor', 1.0)
                        }
                    
                    # Save the modified config
                    temp_config_path = os.path.join(os.getcwd(), 'temp_config.json')
                    with open(temp_config_path, 'w') as f:
                        json.dump(config_dict, f)
                    
                    # Load with the modified config
                    config = AutoConfig.from_pretrained(temp_config_path)
                    model = LlamaForCausalLM.from_pretrained(
                        args.model_name,
                        config=config,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    ).to('cuda' if torch.cuda.is_available() else 'cpu')
                    
                    # Clean up
                    if os.path.exists(temp_config_path):
                        os.remove(temp_config_path)
                except Exception as inner_e:
                    print(f"Alternative loading method failed: {inner_e}")
                    raise e
            else:
                raise e
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.eval()
    print(f"cuda is available: {torch.cuda.is_available()}")

    # params for generate
    num_beams = 4
    no_repeat_ngram_size = 3
    early_stopping = True

    # Prepare the output files
    gpu_id = 0
    messages_output_path = f'{args.name}__messages_output{gpu_id}.csv'
    time_output_path = f'{args.name}__time_output{gpu_id}.csv'

    # Create files with headers
    with open(messages_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'response', 'logits_data'])
    
    with open(time_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'generation_time', 'tokens_per_second'])

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    df = pd.read_parquet(args.dataset_path)
    print(f"Dataset loaded with {len(df)} rows")
    total_start_time = time.time()

    # Process data and generate output for each GPU
    for index, row in df.iterrows():
        if index < args.start_index:
            continue
        question = row['question']
        question = f"""<|begin_of_text|>
<|system|>
You are a helpful assistant.
<|end_of_system|>

{question}

<|end_of_text|>"""
        # Tokenize the input question
        inputs = tokenizer(question, return_tensors='pt').to(model.device)
        
        # Generate the solution with logits
        start_time = time.time()
        
        # Custom generation to capture logits
        input_ids = inputs.input_ids
        attention_mask = torch.ones_like(input_ids)
        generated_tokens = []
        logits_data = []
        top_k = 5  # Number of top tokens to save
        
        with torch.no_grad():
            # Start with the input sequence
            current_ids = input_ids.clone()
            
            # Generate until max length or end token
            for position in range(args.max_length - input_ids.size(1)):
                # Get model outputs
                outputs = model(input_ids=current_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Get top-k tokens and their logits
                topk_logits, topk_indices = torch.topk(next_token_logits, top_k, dim=-1)
                
                # Select the next token (top-1 for greedy decoding)
                next_token = topk_indices[0, 0].unsqueeze(0).unsqueeze(0)
                chosen_token_id = next_token.item()
                chosen_token = tokenizer.decode([chosen_token_id])
                
                # Convert to lists for storage
                top_5_data = []
                for i in range(top_k):
                    token_id = topk_indices[0, i].item()
                    token = tokenizer.decode([token_id])
                    logit = topk_logits[0, i].item()
                    top_5_data.append({
                        "token": token,
                        "token_id": token_id,
                        "logit": logit
                    })
                
                # Get full logits (convert to Python list)
                full_logits = next_token_logits[0].detach().cpu().tolist()
                
                # Store the information for this position
                position_data = {
                    "position": position,
                    "chosen_token": chosen_token,
                    "chosen_token_id": chosen_token_id,
                    "top_5": top_5_data,
                    "full_logits": full_logits
                }
                logits_data.append(position_data)
                
                # Select the next token (top-1 for greedy decoding)
                next_token = topk_indices[0, 0].unsqueeze(0).unsqueeze(0)
                generated_tokens.append(next_token.item())
                
                # Update the input sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
                
                # Check if we've generated an end token
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        generation_time = time.time() - start_time
        
        # Decode the generated solution
        solution = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # Open the files in append mode using 'with' statement
        with open(messages_output_path, 'a', newline='') as messages_file, open(time_output_path, 'a', newline='') as time_file:
            # Create CSV writers
            messages_writer = csv.writer(messages_file)
            time_writer = csv.writer(time_file)
            
            # Format prompt with special tokens
            formatted_prompt = f"<BOS><start_id>user<end_id>\n{question}<eot_id><start_id>assistant<end_id>\n"
            
            # Format response with special tokens
            formatted_response = f"{solution}<EOS>"
            
            # Write to CSV
            logits_data_json = json.dumps(logits_data)
            messages_writer.writerow([formatted_prompt, formatted_response, logits_data_json])
            
            print(f"Wrote user and assistant messages with token info to CSV")

            # Calculate tokens per second
            num_tokens = inputs.input_ids.numel()
            tokens_per_second = num_tokens / generation_time

            # Record and save time elapsed
            time_writer.writerow([question, generation_time, tokens_per_second])
            
            # Print the current index for easy resuming
            print(f"Processed index {index}. To resume from next entry, use --start_index {index+1}")

    # Log the average generation time
    total_generation_time = time.time() - total_start_time
    avg_generation_time = total_generation_time / len(df.index)
    print(f"Average generation time per question: {avg_generation_time:.2f} seconds")



if __name__ == '__main__':
    main()
