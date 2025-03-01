import torch
from transformers import AutoTokenizer, AutoModel
from generate import generate

def test_fine_tuned_model(model_path, prompt, gen_length=128, steps=128, block_length=32):
    """
    Test a fine-tuned LLaDA model with a prompt.
    
    Args:
        model_path: Path to the fine-tuned model
        prompt: Text prompt to generate a response for
        gen_length: Length of the generated response
        steps: Number of sampling steps
        block_length: Block length for semi-autoregressive sampling
    """
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Format prompt for the Instruct model
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # Tokenize prompt
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
    
    # Generate response
    print("Generating response...")
    output = generate(
        model, 
        input_ids, 
        steps=steps, 
        gen_length=gen_length, 
        block_length=block_length, 
        temperature=0.0, 
        cfg_scale=0.0, 
        remasking='low_confidence'
    )
    
    # Decode response
    response = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    return response

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test a fine-tuned LLaDA model")
    parser.add_argument("--model_path", type=str, default="sft_output/final",
                        help="Path to the fine-tuned model")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?",
                        help="Prompt to generate a response for")
    parser.add_argument("--gen_length", type=int, default=128,
                        help="Length of the generated response")
    parser.add_argument("--steps", type=int, default=128,
                        help="Number of sampling steps")
    parser.add_argument("--block_length", type=int, default=32,
                        help="Block length for semi-autoregressive sampling")
    
    args = parser.parse_args()
    
    response = test_fine_tuned_model(
        args.model_path, 
        args.prompt, 
        args.gen_length, 
        args.steps, 
        args.block_length
    )
    
    print("\nPrompt:")
    print(args.prompt)
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()
