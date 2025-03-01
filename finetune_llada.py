import os
import sys
import time
import math
import argparse
import gc
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple, Callable

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as _DDP

# Custom DDP wrapper that preserves model_body attribute
class DDP(_DDP):
    def __init__(self, module, *args, **kwargs):
        super().__init__(module, *args, **kwargs)
        # Preserve model_body attribute for SetFit compatibility
        if hasattr(module, 'model_body'):
            self.model_body = module.model_body
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    set_seed,
    TrainerCallback,
)
from tqdm import tqdm
from setfit import TrainingArguments, Trainer, utils
from datasets import Dataset as HFDataset
import numpy as np

def llada_metric(y_pred, y_test):
    """
    Custom metric function for LLaDA model evaluation.
    
    Args:
        y_pred: Predictions from the model (loss values)
        y_test: Ground truth (not used in this metric)
        
    Returns:
        Dictionary with metric values
    """
    # Use the average loss as the metric
    avg_loss = np.mean(y_pred)
    return {"embedding_loss": avg_loss}

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

def get_gpu_memory_usage(device=None):
    """
    Get the current GPU memory usage in GB.
    
    Args:
        device: CUDA device to check
        
    Returns:
        Memory usage in GB
    """
    if device is None:
        device = torch.cuda.current_device()
    
    # Get memory usage in bytes and convert to GB
    memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    
    return memory_allocated, memory_reserved

class MemoryMonitorCallback(TrainerCallback):
    """
    Callback to monitor GPU memory usage during training and throttle if needed.
    """
    
    def __init__(self, max_memory_gb=70, threshold=0.9):
        """
        Initialize the callback.
        
        Args:
            max_memory_gb: Maximum allowed GPU memory usage in GB
            threshold: Threshold ratio (0.0-1.0) of max_memory_gb to start throttling
        """
        self.max_memory_gb = max_memory_gb
        self.threshold = threshold
        self.throttle_factor = 1.0  # Start with no throttling
        self.last_check_time = time.time()
        self.check_interval = 10  # Check every 10 seconds
    
    def on_step_begin(self, args, state, control, **kwargs):
        """
        Check memory usage before each training step.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            kwargs: Additional arguments
        """
        # Only check memory periodically to avoid overhead
        current_time = time.time()
        if current_time - self.last_check_time < self.check_interval:
            return
        
        self.last_check_time = current_time
        
        # Get current memory usage
        memory_allocated, memory_reserved = get_gpu_memory_usage()
        
        # Calculate memory usage ratio
        memory_ratio = memory_allocated / self.max_memory_gb
        
        # If memory usage is above threshold, throttle by cleaning cache
        if memory_ratio > self.threshold:
            # Try to free some memory
            gc.collect()
            torch.cuda.empty_cache()
            
            # Log memory usage
            if hasattr(args, "local_rank") and args.local_rank == 0:
                print(f"Memory usage high ({memory_allocated:.2f}/{self.max_memory_gb:.2f} GB), cleaning cache")

class LLaDAForwardCallback(TrainerCallback):
    """
    Callback for LLaDA forward process during training.
    """
    
    def __init__(self, mask_id=126336, eps=1e-3):
        """
        Initialize the callback.
        
        Args:
            mask_id: ID of the mask token
            eps: Minimum masking probability
        """
        self.mask_id = mask_id
        self.eps = eps
    
    def on_step_begin(self, args, state, control, **kwargs):
        """
        Apply forward process before each training step.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            kwargs: Additional arguments
        """
        model = kwargs.get("model", None)
        inputs = kwargs.get("inputs", None)
        
        if model is None or inputs is None:
            return
        
        input_ids = inputs["input_ids"]
        prompt_lengths = inputs["prompt_lengths"]
        
        # Apply noise to the input sequence
        noisy_batch, masked_indices, p_mask = forward_process(input_ids, self.mask_id, self.eps)
        
        # Do not add noise to the prompt
        token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
        prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
        noisy_batch[prompt_mask] = input_ids[prompt_mask]
        
        # Update inputs
        inputs["noisy_input_ids"] = noisy_batch
        inputs["masked_indices"] = masked_indices
        inputs["p_mask"] = p_mask

class LLaDAModel(torch.nn.Module):
    """
    Wrapper for LLaDA model to work with SetFit trainer.
    """
    
    def __init__(self, model_name_or_path, mask_id=126336, use_bf16=False):
        """
        Initialize the model.
        
        Args:
            model_name_or_path: Name or path of the model
            mask_id: ID of the mask token
            use_bf16: Whether to use bfloat16 precision
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        )
        self.mask_id = mask_id
        # Add model_body attribute for SetFit compatibility
        self.model_body = self.model
    
    def forward(self, noisy_input_ids, input_ids, masked_indices, p_mask, prompt_lengths, total_lengths):
        """
        Forward pass.
        
        Args:
            noisy_input_ids: Input token IDs with noise applied
            input_ids: Original input token IDs
            masked_indices: Indices of masked tokens
            p_mask: Masking probabilities
            prompt_lengths: Lengths of prompts
            total_lengths: Total lengths of sequences
            
        Returns:
            Loss
        """
        # Forward pass
        outputs = self.model(noisy_input_ids)
        logits = outputs.logits
        
        # Calculate prompt mask
        token_positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand(input_ids.size(0), input_ids.size(1))
        prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
        prompt_mask = prompt_mask.to(torch.int64)
        
        # Calculate answer length (including the padded EOS tokens)
        answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, input_ids.shape[1])
        
        # Calculate loss only on masked tokens
        masked_indices = (noisy_input_ids == self.mask_id)
        
        # Cross entropy loss
        token_loss = F.cross_entropy(
            logits[masked_indices].view(-1, logits.size(-1)),
            input_ids[masked_indices].view(-1),
            reduction='none'
        ) / p_mask[masked_indices]
        
        # Normalize by answer length
        loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
        
        return {"loss": loss}
    
    def save_pretrained(self, output_dir):
        """
        Save the model.
        
        Args:
            output_dir: Directory to save the model
        """
        self.model.save_pretrained(output_dir)

def convert_to_hf_dataset(dataset):
    """
    Convert a PyTorch dataset to a Hugging Face dataset.
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        Hugging Face dataset
    """
    data_dict = {
        "input_ids": [],
        "prompt_lengths": [],
        "total_lengths": [],
        "text": [],  # Add text column for SetFit compatibility
        "label": [],  # Add label column for SetFit compatibility
    }
    
    for i in range(len(dataset)):
        item = dataset[i]
        data_dict["input_ids"].append(item["input_ids"])
        data_dict["prompt_lengths"].append(item["prompt_length"])
        data_dict["total_lengths"].append(item["total_length"])
        data_dict["text"].append(item["input_ids"])  # Use input_ids as text
        data_dict["label"].append(item["input_ids"])  # Use input_ids as label
    
    return HFDataset.from_dict(data_dict)

def setup_distributed(args):
    """
    Set up distributed training environment.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple of (local_rank, world_size)
    """
    if not args.distributed:
        return 0, 1
    
    # Initialize the distributed environment
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        # Environment variables set by torch.distributed.launch or torchrun
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Use all available GPUs or the specified number
        world_size = min(torch.cuda.device_count(), args.num_gpus) if args.num_gpus > 0 else torch.cuda.device_count()
        local_rank = 0
        
        if world_size > 1:
            # Set environment variables for distributed training
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = args.master_port
            
            # Initialize the process group
            dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    
    # Set seed for reproducibility
    set_seed(args.seed + local_rank)  # Different seed per process
    
    return local_rank, world_size

def finetune(args):
    """
    Fine-tune the LLaDA model using SetFit trainer.
    
    Args:
        args: Command-line arguments
    """
    # Set up distributed training
    local_rank, world_size = setup_distributed(args)
    is_main_process = local_rank == 0
    
    # Create output directory (only on main process)
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.distributed:
            print(f"Using {world_size} GPUs for distributed training")
    
    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process:
        print(f"Using device: {device}")
    
    # Load tokenizer (same for all processes)
    if is_main_process:
        print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    # Load dataset
    if is_main_process:
        print(f"Loading dataset from {args.data}")
    dataset = SFTDataset(args.data)
    
    # Convert to Hugging Face dataset and create a small evaluation split
    hf_dataset = convert_to_hf_dataset(dataset)
    
    # Create a small evaluation dataset (10% of the training data)
    dataset_size = len(hf_dataset)
    eval_size = max(1, int(dataset_size * 0.1))  # At least 1 example
    
    # Split the dataset
    indices = list(range(dataset_size))
    if is_main_process:  # Only shuffle on main process to ensure same split across processes
        import random
        random.seed(args.seed)
        random.shuffle(indices)
    
    if args.distributed:
        # Broadcast indices from rank 0 to all other processes
        indices_tensor = torch.tensor(indices, device=device)
        dist.broadcast(indices_tensor, src=0)
        indices = indices_tensor.cpu().tolist()
    
    train_indices = indices[eval_size:]
    eval_indices = indices[:eval_size]
    
    train_dataset = hf_dataset.select(train_indices)
    eval_dataset = hf_dataset.select(eval_indices)
    
    if is_main_process:
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Create model
    if is_main_process:
        print(f"Loading model: {args.model}")
    model = LLaDAModel(args.model, mask_id=args.mask_id, use_bf16=args.use_bf16)
    model.to(device)
    
    # Wrap model with DDP if using distributed training
    if args.distributed and world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        body_learning_rate=args.learning_rate,
        warmup_proportion=args.warmup_ratio,
        l2_weight=args.weight_decay,
        seed=args.seed,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        eval_strategy="steps",  # Match save_strategy
        eval_steps=500,  # Match save_steps
        load_best_model_at_end=True,
        metric_for_best_model="embedding_loss",
        greater_is_better=False,
    )
    
    # Create callbacks
    forward_callback = LLaDAForwardCallback(mask_id=args.mask_id)
    memory_callback = MemoryMonitorCallback(max_memory_gb=args.max_memory_gb)
    
    # Create trainer with column mapping
    column_mapping = {
        "text": "text",  # Map 'text' to 'text'
        "label": "label",  # Map 'label' to 'label'
    }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[forward_callback, memory_callback],
        metric=llada_metric,  # Use custom metric function
        column_mapping=column_mapping,  # Specify column mapping
    )
    
    # Train model
    if is_main_process:
        print(f"Starting training for {args.num_epochs} epochs")
    trainer.train()
    
    # Save final model (only on main process)
    if is_main_process:
        if isinstance(model, DDP):
            model.module.save_pretrained(os.path.join(args.output_dir, "final_model"))
        else:
            model.save_pretrained(os.path.join(args.output_dir, "final_model"))
        print("Training completed!")
    
    # Clean up distributed environment
    if args.distributed and world_size > 1:
        dist.destroy_process_group()

def finetune_distributed(rank, world_size, args):
    """
    Fine-tune the LLaDA model in a distributed setting.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        args: Command-line arguments
    """
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    # Set seed for reproducibility
    set_seed(args.seed + rank)  # Different seed per process
    
    # Create output directory (only on main process)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Using {world_size} GPUs for distributed training")
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    
    # Load tokenizer (same for all processes)
    if rank == 0:
        print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    # Load dataset
    if rank == 0:
        print(f"Loading dataset from {args.data}")
    dataset = SFTDataset(args.data)
    
    # Convert to Hugging Face dataset and create a small evaluation split
    hf_dataset = convert_to_hf_dataset(dataset)
    
    # Create a small evaluation dataset (10% of the training data)
    dataset_size = len(hf_dataset)
    eval_size = max(1, int(dataset_size * 0.1))  # At least 1 example
    
    # Split the dataset
    indices = list(range(dataset_size))
    if rank == 0:  # Only shuffle on rank 0 to ensure same split across processes
        import random
        random.seed(args.seed)
        random.shuffle(indices)
    
    # Broadcast indices from rank 0 to all other processes
    indices_tensor = torch.tensor(indices, device=device)
    dist.broadcast(indices_tensor, src=0)
    indices = indices_tensor.cpu().tolist()
    
    train_indices = indices[eval_size:]
    eval_indices = indices[:eval_size]
    
    train_dataset = hf_dataset.select(train_indices)
    eval_dataset = hf_dataset.select(eval_indices)
    
    if rank == 0:
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Create model
    if rank == 0:
        print(f"Loading model: {args.model}")
    model = LLaDAModel(args.model, mask_id=args.mask_id, use_bf16=args.use_bf16)
    model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        body_learning_rate=args.learning_rate,
        warmup_proportion=args.warmup_ratio,
        l2_weight=args.weight_decay,
        seed=args.seed,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        eval_strategy="steps",  # Match save_strategy
        eval_steps=500,  # Match save_steps
        load_best_model_at_end=True,
        metric_for_best_model="embedding_loss",
        greater_is_better=False,
    )
    
    # Create callbacks
    forward_callback = LLaDAForwardCallback(mask_id=args.mask_id)
    memory_callback = MemoryMonitorCallback(max_memory_gb=args.max_memory_gb)
    
    # Create trainer with column mapping
    column_mapping = {
        "text": "text",  # Map 'text' to 'text'
        "label": "label",  # Map 'label' to 'label'
    }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[forward_callback, memory_callback],
        metric=llada_metric,  # Use custom metric function
        column_mapping=column_mapping,  # Specify column mapping
    )
    
    # Train model
    if rank == 0:
        print(f"Starting training for {args.num_epochs} epochs")
    trainer.train()
    
    # Save final model (only on main process)
    if rank == 0:
        model.module.save_pretrained(os.path.join(args.output_dir, "final_model"))
        print("Training completed!")
    
    # Clean up
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaDA model using SetFit trainer")
    
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
    parser.add_argument("--max_memory_gb", type=float, default=70, help="Maximum GPU memory usage in GB")
    
    # Distributed training arguments
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use (0 for all available)")
    parser.add_argument("--master_port", type=str, default="12355", help="Port for distributed training")
    
    args = parser.parse_args()
    
    if args.distributed:
        # Use all available GPUs or the specified number
        world_size = min(torch.cuda.device_count(), args.num_gpus) if args.num_gpus > 0 else torch.cuda.device_count()
        
        if world_size > 1:
            # Launch distributed processes
            mp.spawn(
                finetune_distributed,
                args=(world_size, args),
                nprocs=world_size,
                join=True
            )
            return
    
    # Fall back to single-GPU or CPU training if distributed is disabled or only one GPU is available
    finetune(args)

if __name__ == "__main__":
    main()
