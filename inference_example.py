from generate import generate
from transformers import AutoModel, AutoTokenizer

def run_inference():
    device = "cuda"
    model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    model.load_state_dict(torch.load("llada-8b-instruct-sft.pt"))
    model = model.to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    
    # Example prompt
    prompt = "Calculate 3 + 5."
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)
    
    # Generate response
    output = generate(
        model,
        input_ids,
        steps=128,
        gen_length=64,
        block_length=32,
        temperature=0.0,
        remasking="low_confidence"
    )
    
    # Decode and remove special tokens
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Prompt: {prompt}\nGenerated Response: {response}")

if __name__ == "__main__":
    run_inference()