import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

os.environ[‘HF_HOME’] = ‘/mount/model-cache'
os.environ[‘HF_HUB_CACHE’] = ‘/mount/model-cache’

class SFTDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # Find max length in the batch
    max_length = max([len(item["input_ids"]) for item in batch])
    
    # Pad input_ids with EOS token (as mentioned in the guidelines)
    input_ids = []
    prompt_lengths = []
    
    for item in batch:
        ids = item["input_ids"]
        prompt_len = item["prompt_length"]
        
        # Pad with EOS token (assuming EOS token is the last token in the vocabulary)
        padding_length = max_length - len(ids)
        padded_ids = ids + [126081] * padding_length  # 126081 is the EOS token ID for LLaDA
        
        input_ids.append(padded_ids)
        prompt_lengths.append(prompt_len)
    
    return {
        "input_ids": torch.tensor(input_ids),
        "prompt_lengths": torch.tensor(prompt_lengths)
    }

def forward_process(input_ids, eps=1e-3):
    """
    Apply the forward process to add noise to the input.
    
    Args:
        input_ids: Input token IDs
        eps: Small epsilon to avoid division by zero
        
    Returns:
        noisy_batch: Input with some tokens masked
        masked_indices: Boolean tensor indicating which tokens are masked
        p_mask: Probability of masking for each token
    """
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # 126336 is used for [MASK] token
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    return noisy_batch, masked_indices, p_mask

def train(args):
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
    print(f"Loading model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(device)
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = SFTDataset(args.data_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    
    num_training_steps = len(dataloader) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            input_ids, prompt_lengths = batch["input_ids"].to(device), batch["prompt_lengths"].to(device)
            
            # Apply forward process to get noisy batch
            noisy_batch, _, p_mask = forward_process(input_ids)
            
            # Do not add noise to the prompt (as mentioned in the guidelines)
            token_positions = torch.arange(noisy_batch.shape[1], device=device).expand(noisy_batch.size(0), noisy_batch.size(1))
            prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
            noisy_batch[prompt_mask] = input_ids[prompt_mask]
            
            # Calculate the answer length (including the padded <EOS> tokens)
            prompt_mask = prompt_mask.to(torch.int64)    
            answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
            answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])    
            
            masked_indices = (noisy_batch == 126336)
            
            # Forward pass
            logits = model(input_ids=noisy_batch).logits
            
            # Calculate loss
            token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
            ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
            
            # Backward pass
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress
            epoch_loss += ce_loss.item()
            global_step += 1
            
            progress_bar.set_postfix({"loss": ce_loss.item()})
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint to {checkpoint_dir}")
        
        # Save model after each epoch
        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        print(f"Saved model after epoch {epoch+1} to {epoch_dir}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    print(f"Saved final model to {final_dir}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaDA with SFT")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base", help="Model name or path")
    parser.add_argument("--data_path", type=str, default="sft_data/processed_data.pt", help="Path to processed data")
    parser.add_argument("--output_dir", type=str, default="sft_output", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (default: 0)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)

if __name__ == "__main__":
    main()
