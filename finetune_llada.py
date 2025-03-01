import torch
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer
from preprocess_sft_data import preprocess

def forward_process(input_ids, eps=1e-3):
    """Modified from guidelines.md - no prompt masking"""
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    return noisy_batch, masked_indices, p_mask

def train():
    # Load data and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True).to(device)
    
    data = torch.load("processed_sft_data.pt")
    dataset = torch.utils.data.TensorDataset(data["input_ids"], data["prompt_lengths"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # Optimizer setup
    optimizer = AdamW(model.parameters(), lr=2.5e-5, weight_decay=0.1)
    
    # Training loop
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch_idx, (input_ids, prompt_lengths) in enumerate(loader):
            input_ids = input_ids.to(device)
            prompt_lengths = prompt_lengths.to(device)

            # SFT masking
            noisy_batch, _, p_mask = forward_process(input_ids)
            token_positions = torch.arange(noisy_batch.shape[1], device=device)
            prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
            noisy_batch[prompt_mask] = input_ids[prompt_mask]

            # Forward pass
            logits = model(noisy_batch).logits
            masked_indices = (noisy_batch == 126336)
            
            # Loss calculation
            token_loss = torch.nn.functional.cross_entropy(
                logits[masked_indices], 
                input_ids[masked_indices], 
                reduction='none'
            ) / p_mask[masked_indices]
            
            loss = token_loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")

        print(f"Epoch {epoch} Avg Loss: {total_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), "llada-8b-instruct-sft.pt")

if __name__ == "__main__":
    train()