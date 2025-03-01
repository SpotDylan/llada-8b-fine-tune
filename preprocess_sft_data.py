import json
import torch
from transformers import AutoTokenizer

def process_multi_turn(dialogue):
    """Split multi-turn dialogues into single-turn pairs"""
    turns = dialogue.split("\n")
    pairs = []
    context = ""
    for turn in turns:
        if "User:" in turn:
            context = turn  # Reset context for simplicity
        elif "Assistant:" in turn:
            pairs.append((context, turn))
    return pairs

def preprocess(jsonl_path, tokenizer, max_length=4096):
    processed_data = []
    
    with open(jsonl_path) as f:
        for line in f:
            example = json.loads(line)
            
            # Handle multi-turn dialogues
            if example["is_multi_turn"]:
                pairs = process_multi_turn(example["prompt"] + "\n" + example["response"])
                for prompt, response in pairs:
                    encoded = tokenizer(
                        f"<BOS>{prompt}<eot_id><start_id>assistant<end_id>\n{response}<EOS>",
                        truncation=True,
                        max_length=max_length
                    )
                    processed_data.append(encoded)
            else:
                # Single-turn formatting
                text = f"<BOS><start_id>user<end_id>\n{example['prompt']}<eot_id><start_id>assistant<end_id>\n{example['response']}<EOS>"
                encoded = tokenizer(text, truncation=True, max_length=max_length)
                processed_data.append(encoded)
    
    # Pad sequences and create prompt_lengths
    input_ids = [x["input_ids"] for x in processed_data]
    padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(seq) for seq in input_ids],
        batch_first=True,
        padding_value=tokenizer.eos_token_id
    )
    
    return {
        "input_ids": padded,
        "prompt_lengths": torch.tensor([
            len(tokenizer.encode(x.split("<start_id>assistant")[0])) 
            for x in processed_data
        ])
    }

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    processed = preprocess("fake_sft_data.jsonl", tokenizer)
    torch.save(processed, "processed_sft_data.pt")