import torch
import argparse
from transformers import AutoModel, AutoTokenizer

def generate(model, prompt, steps=128, gen_length=128, block_length=32, temperature=0., 
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    """
    Generate text using the fine-tuned LLaDA model.
    
    Args:
        model: The fine-tuned LLaDA model
        prompt: Tokenized prompt
        steps: Number of sampling steps
        gen_length: Length of generated text
        block_length: Block length for semi-autoregressive sampling
        temperature: Temperature for sampling
        cfg_scale: Classifier-free guidance scale
        remasking: Remasking strategy ('low_confidence' or 'random')
        mask_id: ID of the mask token
    
    Returns:
        Generated text
    """
    device = model.device
    
    # Initialize with mask tokens
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    prompt_index = (x != mask_id)
    
    # Ensure gen_length is divisible by block_length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    # Adjust steps to be divisible by num_blocks
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    
    for num_block in range(num_blocks):
        # Get mask indices for current block
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        # Calculate number of tokens to transfer at each step
        mask_num = block_mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps_per_block
        remainder = mask_num % steps_per_block
        
        num_transfer_tokens = torch.zeros(mask_num.size(0), steps_per_block, device=device, dtype=torch.int64) + base
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1
        
        for i in range(steps_per_block):
            # Get current mask indices
            mask_index = (x == mask_id)
            
            # Apply classifier-free guidance if needed
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            
            # Sample from logits
            if temperature > 0:
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                x0 = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), -1)
            else:
                x0 = torch.argmax(logits, dim=-1)
            
            # Apply remasking strategy
            if remasking == 'low_confidence':
                # Get confidence scores
                p = torch.nn.functional.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=device)
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking}")
            
            # Don't consider tokens outside current block
            x0_p[:, :block_start] = -float('inf')
            x0_p[:, block_end:] = -float('inf')
            
            # Replace masked tokens with predictions
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float('inf'))
            
            # Select tokens to transfer based on confidence
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            # Update x with selected tokens
            x[transfer_index] = x0[transfer_index]
    
    return x

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned LLaDA model")
    parser.add_argument("--model_path", type=str, default="sft_output/final", help="Path to fine-tuned model")
    parser.add_argument("--steps", type=int, default=128, help="Number of sampling steps")
    parser.add_argument("--gen_length", type=int, default=128, help="Length of generated text")
    parser.add_argument("--block_length", type=int, default=32, help="Block length for semi-autoregressive sampling")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="Classifier-free guidance scale")
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"], 
                        help="Remasking strategy")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (default: 0)")
    
    args = parser.parse_args()
    
    # Check available GPUs and set device
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        
        if args.gpu_id >= 0 and args.gpu_id < num_gpus:
            device = torch.device(f"cuda:{args.gpu_id}")
            print(f"Using device: {device}")
        else:
            print(f"Specified GPU ID {args.gpu_id} is not available. Available GPUs: {num_gpus}")
            print(f"Defaulting to CPU")
            device = torch.device("cpu")
    else:
        print("CUDA is not available. Using CPU")
        device = torch.device("cpu")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Test prompts
    test_prompts = [
        "What is the capital of Germany?",
        "Explain the concept of quantum computing in simple terms.",
        "Write a short story about a robot who learns to feel emotions."
    ]
    
    print("\n" + "="*50)
    print(f"Testing with {len(test_prompts)} prompts")
    print("="*50)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        # Format as chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Tokenize
        input_ids = tokenizer(formatted_prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        
        # Generate response
        with torch.no_grad():
            output = generate(
                model, 
                input_ids, 
                steps=args.steps, 
                gen_length=args.gen_length, 
                block_length=args.block_length,
                temperature=args.temperature, 
                cfg_scale=args.cfg_scale, 
                remasking=args.remasking
            )
        
        # Decode response
        response = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"Response: {response}")
        print("-"*50)

if __name__ == "__main__":
    main()
