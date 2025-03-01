import argparse
import torch
from transformers import AutoTokenizer, AutoModel

def generate_response(
    model,
    tokenizer,
    prompt,
    steps=128,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    Generate a response using the fine-tuned LLaDA model.
    
    Args:
        model: The LLaDA model
        tokenizer: The tokenizer
        prompt: The input prompt
        steps: Number of sampling steps
        gen_length: Length of the generated response
        block_length: Block length for semi-autoregressive sampling
        temperature: Sampling temperature
        cfg_scale: Classifier-free guidance scale
        remasking: Remasking strategy ("low_confidence" or "random")
        mask_id: ID of the mask token
        
    Returns:
        The generated response
    """
    # Format the prompt for the model
    formatted_prompt = f"<start_id>user<end_id>\n{prompt}<eot_id><start_id>assistant<end_id>\n"
    
    # Tokenize the prompt
    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(model.device)
    
    # Generate the response
    with torch.no_grad():
        # Import the generate function from the generate module
        from generate import generate
        
        # Generate the response
        output = generate(
            model,
            input_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
        )
        
        # Decode the response
        response = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    
    return response

def interactive_chat(model, tokenizer, args):
    """
    Run an interactive chat session with the fine-tuned LLaDA model.
    
    Args:
        model: The LLaDA model
        tokenizer: The tokenizer
        args: Command-line arguments
    """
    print("\n" + "=" * 50)
    print("Interactive Chat with Fine-tuned LLaDA Model")
    print("=" * 50)
    print("Type 'exit' to end the conversation.\n")
    
    conversation_history = []
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if the user wants to exit
        if user_input.lower() == "exit":
            print("\nExiting chat. Goodbye!")
            break
        
        # Add user input to conversation history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Generate response
        response = generate_response(
            model,
            tokenizer,
            user_input,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking,
            mask_id=args.mask_id,
        )
        
        # Add response to conversation history
        conversation_history.append({"role": "assistant", "content": response})
        
        # Print response
        print(f"Assistant: {response}\n")

def evaluate_examples(model, tokenizer, args):
    """
    Evaluate the fine-tuned LLaDA model on a set of example prompts.
    
    Args:
        model: The LLaDA model
        tokenizer: The tokenizer
        args: Command-line arguments
    """
    # Example prompts from different categories
    example_prompts = [
        # Code
        "Write a Python function to find the factorial of a number using recursion.",
        
        # Mathematics
        "Solve the equation: 2x + 5 = 15",
        
        # Instruction-following
        "Explain how to make a simple pasta dish.",
        
        # Structured data
        "Convert this JSON to a Python dictionary: {\"name\": \"Alice\", \"age\": 30, \"city\": \"New York\"}",
        
        # General knowledge
        "Explain the concept of machine learning in simple terms.",
        
        # Reasoning
        "If a shirt costs $25 and is on sale for 20% off, what is the final price?",
    ]
    
    print("\n" + "=" * 50)
    print("Evaluating Fine-tuned LLaDA Model on Example Prompts")
    print("=" * 50 + "\n")
    
    for i, prompt in enumerate(example_prompts):
        print(f"Example {i+1}:")
        print(f"Prompt: {prompt}")
        
        # Generate response
        response = generate_response(
            model,
            tokenizer,
            prompt,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking,
            mask_id=args.mask_id,
        )
        
        print(f"Response: {response}\n")
        print("-" * 50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned LLaDA model")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="sft_output/final_model", help="Path to the fine-tuned model")
    parser.add_argument("--tokenizer", type=str, default="sft_data/preprocessed/tokenizer", help="Path to the tokenizer")
    parser.add_argument("--mask_id", type=int, default=126336, help="ID of the mask token")
    
    # Generation arguments
    parser.add_argument("--steps", type=int, default=128, help="Number of sampling steps")
    parser.add_argument("--gen_length", type=int, default=128, help="Length of the generated response")
    parser.add_argument("--block_length", type=int, default=32, help="Block length for semi-autoregressive sampling")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="Classifier-free guidance scale")
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"], help="Remasking strategy")
    
    # Mode arguments
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "examples"], help="Inference mode")
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 precision")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    # Load model
    print(f"Loading model from {args.model}")
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32,
    )
    model.to(device)
    model.eval()
    
    # Run inference
    if args.mode == "interactive":
        interactive_chat(model, tokenizer, args)
    elif args.mode == "examples":
        evaluate_examples(model, tokenizer, args)

if __name__ == "__main__":
    main()
