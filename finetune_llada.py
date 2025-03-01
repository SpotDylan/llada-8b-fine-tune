import os
import sys
import time
import math
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    set_seed,
)
from tqdm import tqdm

os.environ['HF_HOME'] = '/mount/model-cache'

class SFTDataset(Dataset):
    """Dataset for LLaDA supervised fine-tuning."""
    
    def __init__(self, data_path: str):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the preprocessed data file
        """
        self.data = torch.load(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """
    Collate function for the DataLoader.
    
    Args:
        batch: A batch of data
        
    Returns:
        A dictionary containing the batched data
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    prompt_lengths = torch.stack([item["prompt_length"] for item in batch])
    total_lengths = torch.stack([item["total_length"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "prompt_lengths": prompt_lengths,
        "total_lengths": total_lengths,
    }

def forward_process(input_ids, mask_id=126336, eps=1e-3):
    """
    Apply noise to the input sequence.
    
    Args:
        input_ids: Input token IDs
        mask_id: ID of the mask token
        eps: Minimum masking probability
        
    Returns:
        Tuple of (noisy_batch, masked_indices, p_mask)
    """
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)
    
    return noisy_batch, masked_indices, p_mask

def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    mask_id=126336,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
):
    """
    Train the model for one epoch.
    
    Args:
        model: The LLaDA model
        dataloader: DataLoader for the training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        mask_id: ID of the mask token
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for gradient clipping
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        prompt_lengths = batch["prompt_lengths"].to(device)
        total_lengths = batch["total_lengths"].to(device)
        
        # Apply noise to the input sequence
        noisy_batch, masked_indices, p_mask = forward_process(input_ids, mask_id)
        
        # Do not add noise to the prompt
        token_positions = torch.arange(noisy_batch.shape[1], device=device).expand(noisy_batch.size(0), noisy_batch.size(1))
        prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
        noisy_batch[prompt_mask] = input_ids[prompt_mask]
        
        # Calculate the answer length (including the padded EOS tokens)
        prompt_mask = prompt_mask.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
        
        # Forward pass
        outputs = model(noisy_batch)
        logits = outputs.logits
        
        # Calculate loss only on masked tokens
        masked_indices = (noisy_batch == mask_id)
        
        # Cross entropy loss
        token_loss = F.cross_entropy(
            logits[masked_indices].view(-1, logits.size(-1)),
            input_ids[masked_indices].view(-1),
            reduction='none'
        ) / p_mask[masked_indices]
        
        # Normalize by answer length
        loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights if we've accumulated enough gradients
        if (step + 1) % gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update progress bar
        total_loss += loss.item() * gradient_accumulation_steps
        total_samples += 1
        progress_bar.set_postfix({"loss": total_loss / total_samples})
    
    return total_loss / total_samples

def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir):
    """
    Save a checkpoint of the model.
    
    Args:
        model: The model to save
        optimizer: The optimizer
        scheduler: The learning rate scheduler
        epoch: Current epoch
        loss: Current loss
        output_dir: Directory to save the checkpoint
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    
    checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def finetune(args):
    """
    Fine-tune the LLaDA model.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32,
    )
    model.to(device)
    
    # Load dataset
    print(f"Loading dataset from {args.data}")
    dataset = SFTDataset(args.data)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Set up optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    
    # Set up learning rate scheduler
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Training loop
    print(f"Starting training for {args.num_epochs} epochs")
    best_loss = float("inf")
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(
            model,
            dataloader,
            optimizer,
            scheduler,
            device,
            mask_id=args.mask_id,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        
        print(f"Epoch {epoch + 1} completed. Loss: {train_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch + 1,
            train_loss,
            args.output_dir,
        )
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print(f"Saved best model with loss: {best_loss:.4f}")
    
    # Save final model
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    print("Training completed!")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaDA model")
    
    # Data arguments
    parser.add_argument("--data", type=str, default="sft_data/preprocessed/preprocessed_data.pt", help="Path to preprocessed data")
    parser.add_argument("--tokenizer", type=str, default="sft_data/preprocessed/tokenizer", help="Path to tokenizer")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Base", help="Model name or path")
    parser.add_argument("--mask_id", type=int, default=126336, help="ID of the mask token")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="sft_output", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    finetune(args)

if __name__ == "__main__":
    main()
