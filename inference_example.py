import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from generate import generate

def inference(args):
    """
    Run inference with a fine-tuned LLaDA model.
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32
    ).to(device).eval()
    
    # Process each prompt
    for prompt in args.prompts:
        print("\n" + "="*50)
        print(f"Prompt: {prompt}")
        print("="*50)
        
        # Format as chat template for Instruct model
        if args.instruct:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            formatted_prompt = prompt
        
        # Tokenize prompt
        input_ids = tokenizer(formatted_prompt, return_tensors="pt")['input_ids'].to(device)
        
        # Generate response
        print("Generating response...")
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
        print("-"*50)
        print(f"Response: {response}")
        print("-"*50)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned LLaDA model")
    
    parser.add_argument("--model_path", type=str, default="output/final",
                        help="Path to the fine-tuned model")
    parser.add_argument("--prompts", type=str, nargs="+", required=True,
                        help="Prompts to generate responses for")
    parser.add_argument("--instruct", action="store_true",
                        help="Use chat template for Instruct model")
    parser.add_argument("--steps", type=int, default=128,
                        help="Number of sampling steps")
    parser.add_argument("--gen_length", type=int, default=128,
                        help="Length of generated response")
    parser.add_argument("--block_length", type=int, default=32,
                        help="Block length for semi-autoregressive sampling")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--cfg_scale", type=float, default=0.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random"],
                        help="Remasking strategy")
    parser.add_argument("--use_bf16", action="store_true",
                        help="Use bfloat16 precision")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inference(args)
