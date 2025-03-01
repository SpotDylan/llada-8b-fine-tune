#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune LLaDA model using supervised fine-tuning (SFT).
This script implements the SFT process as described in the LLaDA paper.
"""

import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import random
import numpy as np

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Constants
MASK_TOKEN_ID = 126336  # As specified in the guidelines

class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning of LLaDA."""
    
    def __init__(self, data_path):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the preprocessed data file.
        """
        self.data = torch.load(data_path)
        self.input_ids = self.data["input_ids"]
        self.prompt_lengths = self.data["prompt_lengths"]
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "prompt_length": self.prompt_lengths[idx]
        }

def collate_fn(batch, pad_token_id):
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of samples from the dataset.
        pad_token_id: Token ID to use for padding.
        
    Returns:
        Batched tensors.
    """
    # Get max length in the batch
    max_length = max(len(sample["input_ids"]) for sample in batch)
    
    # Pad input_ids
    input_ids = []
    attention_mask = []
    prompt_lengths = []
    
    for sample in batch:
        sample_input_ids = sample["input_ids"]
        padding_length = max_length - len(sample_input_ids)
        
        # Pad with pad_token_id
        padded_input_ids = sample_input_ids + [pad_token_id] * padding_length
        input_ids.append(padded_input_ids)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        mask = [1] * len(sample_input_ids) + [0] * padding_length
        attention_mask.append(mask)
        
        # Store prompt length
        prompt_lengths.append(sample["prompt_length"])
    
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "prompt_lengths": torch.tensor(prompt_lengths)
    }

def forward_process(input_ids, eps=1e-3):
    """
    Apply noise to the input_ids by masking tokens.
    
    Args:
        input_ids: Input token IDs.
        eps: Small constant to ensure minimum masking probability.
        
    Returns:
        Tuple of (noisy_batch, masked_indices, p_mask).
    """
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, MASK_TOKEN_ID, input_ids)
    
    return noisy_batch, masked_indices, p_mask

def train(args, model, tokenizer, train_dataloader, optimizer, scheduler):
    """
    Train the model using supervised fine-tuning.
    
    Args:
        args: Training arguments.
        model: The LLaDA model.
        tokenizer: Tokenizer for the model.
        train_dataloader: DataLoader for training data.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        
    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        prompt_lengths = batch["prompt_lengths"].to(args.device)
        
        # Apply forward process to get noisy batch
        noisy_batch, masked_indices, p_mask = forward_process(input_ids)
        
        # Do not add noise to the prompt
        token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
        prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
        noisy_batch[prompt_mask] = input_ids[prompt_mask]
        
        # Calculate the answer length (including the padded <EOS> tokens)
        prompt_mask = prompt_mask.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
        
        # Forward pass
        outputs = model(input_ids=noisy_batch, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Calculate loss only on masked tokens
        masked_indices = (noisy_batch == MASK_TOKEN_ID)
        
        if masked_indices.sum() == 0:
            # Skip this batch if no tokens are masked
            continue
        
        token_loss = F.cross_entropy(
            logits[masked_indices].view(-1, logits.size(-1)), 
            input_ids[masked_indices].view(-1), 
            reduction='none'
        ) / p_mask[masked_indices]
        
        # Normalize by answer length
        ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
        
        # Backward pass
        optimizer.zero_grad()
        ce_loss.backward()
        
        # Clip gradients
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        
        # Update progress bar
        total_loss += ce_loss.item()
        progress_bar.set_postfix({"loss": ce_loss.item()})
    
    return total_loss / len(train_dataloader)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune LLaDA model using SFT")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="sft_data/processed/sft_data.pt",
                        help="Path to the preprocessed data file")
    parser.add_argument("--output_dir", type=str, default="models/llada-sft",
                        help="Directory to save the fine-tuned model")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="llada-8b",
                        help="Path to the LLaDA model or model name")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5,
                        help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Ratio of warmup steps to total steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save model checkpoint every X steps")
    
    # Other arguments
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    
    Args:
        rank: Rank of the current process.
        world_size: Number of processes.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleanup the distributed environment."""
    dist.destroy_process_group()

def train_model(rank, world_size, args):
    """
    Train the model on a single GPU.
    
    Args:
        rank: Rank of the current process.
        world_size: Number of processes.
        args: Training arguments.
    """
    # Setup the distributed environment
    setup(rank, world_size)
    
    # Set random seed
    set_seed(args.seed + rank)  # Different seed for each process
    
    # Setup device
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)
    
    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer and model
    if rank == 0:
        logger.info(f"Loading tokenizer and model from {args.model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.to(args.device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Load dataset
    if rank == 0:
        logger.info(f"Loading dataset from {args.data_path}")
    
    dataset = SFTDataset(args.data_path)
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed
    )
    
    # Create data loader
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Train the model
    if rank == 0:
        logger.info("Starting training")
    
    for epoch in range(args.num_epochs):
        # Set the epoch for the sampler
        sampler.set_epoch(epoch)
        
        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        avg_loss = train(args, model, tokenizer, train_dataloader, optimizer, scheduler)
        
        if rank == 0:
            logger.info(f"Average loss: {avg_loss:.4f}")
            
            # Save model checkpoint
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save the model (unwrap DDP)
            model_to_save = model.module
            model_to_save.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Saved model checkpoint to {checkpoint_dir}")
    
    # Save final model
    if rank == 0:
        logger.info("Saving final model")
        model_to_save = model.module
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Saved final model to {args.output_dir}")
    
    # Cleanup
    cleanup()

def main():
    """Main function for fine-tuning LLaDA."""
    args = parse_args()
    
    # Add distributed training arguments
    parser = argparse.ArgumentParser(description="Distributed training arguments")
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="Number of GPUs to use for distributed training")
    distributed_args, _ = parser.parse_known_args()
    
    # Launch the distributed training
    world_size = distributed_args.num_gpus
    mp.spawn(
        train_model,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
