import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

def forward_process(input_ids, prompt_lengths, eps=1e-3, mask_id=126336):
    """
    Apply forward process (add noise) to the input_ids.
    Only add noise to the response part, not the prompt.
    
    Args:
        input_ids: Tensor of shape (batch_size, seq_len)
        prompt_lengths: Tensor of shape (batch_size,) containing the length of each prompt
        eps: Minimum mask probability
        mask_id: Token ID for the mask token
    
    Returns:
        noisy_batch: Input with noise applied
        masked_indices: Boolean tensor indicating which tokens are masked
        p_mask: Mask probability for each token
    """
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    
    # Create a mask for tokens that can be masked (only in the response, not the prompt)
    token_positions = torch.arange(l, device=input_ids.device).expand(b, l)
    prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
    
    # Only mask tokens in the response (not in the prompt)
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    masked_indices = masked_indices & (~prompt_mask)
    
    # Apply masking
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)
    
    return noisy_batch, masked_indices, p_mask

def train_epoch(model, dataloader, optimizer, scheduler, device, mask_id=126336):
    """
    Train the model for one epoch.
    
    Args:
        model: The LLaDA model
        dataloader: DataLoader containing the training data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        device: Device to train on
        mask_id: Token ID for the mask token
    
    Returns:
        average_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids, prompt_lengths = batch
        input_ids = input_ids.to(device)
        prompt_lengths = prompt_lengths.to(device)
        
        # Apply forward process (add noise)
        noisy_batch, masked_indices, p_mask = forward_process(input_ids, prompt_lengths, mask_id=mask_id)
        
        # Calculate answer lengths (including padded EOS tokens)
        prompt_mask = torch.arange(input_ids.shape[1], device=device).expand(input_ids.size(0), input_ids.size(1)) < prompt_lengths.unsqueeze(1)
        prompt_mask = prompt_mask.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, input_ids.shape[1])
        
        # Forward pass
        logits = model(input_ids=noisy_batch).logits
        
        # Calculate loss only on masked tokens
        token_loss = F.cross_entropy(
            logits[masked_indices].reshape(-1, logits.size(-1)), 
            input_ids[masked_indices].reshape(-1), 
            reduction='none'
        ) / p_mask[masked_indices]
        
        # Normalize by answer length
        ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
        
        # Backward pass
        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += ce_loss.item()
    
    return total_loss / len(dataloader)

def finetune(args):
    """
    Fine-tune the LLaDA model.
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Load model
    print(f"Loading model from {args.model_name}...")
    model = AutoModel.from_pretrained(
        args.model_name, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32
    ).to(device)
    
    # Load preprocessed data
    print(f"Loading preprocessed data from {args.data_path}...")
    data = torch.load(args.data_path)
    input_ids = data['input_ids']
    prompt_lengths = data['prompt_lengths']
    
    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, prompt_lengths)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        avg_loss = train_epoch(
            model, 
            dataloader, 
            optimizer, 
            scheduler, 
            device,
            mask_id=args.mask_id
        )
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            print(f"Saving model checkpoint to {checkpoint_path}...")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
    
    # Save final model
    if args.output_dir:
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        
        print(f"Saving final model to {final_path}...")
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
    
    print("Fine-tuning complete!")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaDA model")
    
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base",
                        help="Name or path of the pre-trained LLaDA model")
    parser.add_argument("--data_path", type=str, default="sft_data/sft_data.pt",
                        help="Path to the preprocessed data file")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of warmup steps")
    parser.add_argument("--use_bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--mask_id", type=int, default=126336,
                        help="Token ID for the mask token")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    finetune(args)
