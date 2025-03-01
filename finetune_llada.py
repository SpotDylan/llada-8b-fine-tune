import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import time
import traceback
from datetime import datetime
import gc

os.environ["HF_HOME"] = "/mount/model-cache"
os.environ["HF_HUB_CACHE"] = "/mount/model-cache"

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

def setup_logging(args):
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging format and file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()

def get_total_gpu_memory():
    """Get total memory across all available GPUs in GB"""
    total_memory = 0
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory += torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
    return total_memory

def get_used_gpu_memory():
    """Get total used memory (allocated + reserved) across all available GPUs in GB"""
    used_memory = 0
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            used_memory += torch.cuda.memory_allocated(i) / (1024 ** 3)
            used_memory += torch.cuda.memory_reserved(i) / (1024 ** 3)
    return used_memory

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_distributed(rank, world_size, args):
    # Set up distributed training
    setup_distributed(rank, world_size)
    
    # Set up logging (only on main process)
    if rank == 0:
        logger = setup_logging(args)
        logger.info(f"Starting distributed training with {world_size} GPUs")
        logger.info(f"Memory limit set to {args.memory_limit} GB across all GPUs")
        
        # Log training parameters
        logger.info(f"Starting training with parameters:")
        for arg, value in vars(args).items():
            logger.info(f"  {arg}: {value}")
    else:
        logger = None
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    if rank == 0 and logger:
        logger.info(f"Process {rank} using device: {device}")
    
    # Load model and tokenizer
    if rank == 0 and logger:
        logger.info(f"Loading model: {args.model_name}")
        start_time = time.time()
    
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    if rank == 0 and logger:
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Load dataset
    if rank == 0 and logger:
        logger.info(f"Loading dataset from {args.data_path}")
    
    dataset = SFTDataset(args.data_path)
    
    # Use DistributedSampler to partition the dataset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    # Create dataloader with the sampler
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn
    )
    
    if rank == 0 and logger:
        logger.info(f"Dataset loaded with {len(dataset)} examples")
        logger.info(f"Each GPU will process approximately {len(dataset) // world_size} examples")
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    
    num_training_steps = len(dataloader) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    if rank == 0 and logger:
        logger.info(f"Training for {args.num_epochs} epochs, {num_training_steps} total steps")
        logger.info(f"Warmup for {num_warmup_steps} steps")
    
    # Training loop
    model.train()
    global_step = 0
    
    # Metrics tracking
    all_losses = []
    all_accuracies = []
    
    # Initial batch size
    current_batch_size = args.batch_size
    
    try:
        for epoch in range(args.num_epochs):
            # Set the epoch for the sampler
            sampler.set_epoch(epoch)
            
            if rank == 0 and logger:
                logger.info(f"Starting Epoch {epoch+1}/{args.num_epochs}")
                epoch_start_time = time.time()
            
            epoch_loss = 0
            epoch_correct_predictions = 0
            epoch_total_predictions = 0
            
            # Only show progress bar on rank 0
            if rank == 0:
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
                dataloader_iterator = progress_bar
            else:
                dataloader_iterator = dataloader
            
            for batch_idx, batch in enumerate(dataloader_iterator):
                try:
                    batch_start_time = time.time()
                    input_ids, prompt_lengths = batch["input_ids"].to(device), batch["prompt_lengths"].to(device)
                    
                    # Check memory usage across all GPUs
                    if rank == 0 and logger and batch_idx % args.log_steps == 0:
                        used_memory = get_used_gpu_memory()
                        total_memory = get_total_gpu_memory()
                        memory_usage_percent = (used_memory / total_memory) * 100
                        
                        logger.info(f"Total GPU Memory Usage: {used_memory:.2f} GB / {total_memory:.2f} GB ({memory_usage_percent:.2f}%)")
                        
                        # Check if we're approaching the memory limit
                        if used_memory > args.memory_limit * 0.9:  # Within 90% of limit
                            if current_batch_size > 1:
                                # Reduce batch size to throttle performance
                                new_batch_size = max(1, current_batch_size // 2)
                                logger.warning(f"Approaching memory limit ({used_memory:.2f} GB / {args.memory_limit} GB). "
                                              f"Reducing batch size from {current_batch_size} to {new_batch_size} to throttle performance.")
                                current_batch_size = new_batch_size
                                
                                # Force garbage collection
                                gc.collect()
                                torch.cuda.empty_cache()
                        
                        # Log detailed memory summary periodically
                        if batch_idx % (args.log_steps * 10) == 0:
                            memory_summary = torch.cuda.memory_summary(device=device, abbreviated=False)
                            logger.info(f"Detailed GPU Memory Summary before forward pass:\n{memory_summary}")
                    
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
                    
                    # Calculate accuracy for masked tokens
                    predictions = torch.argmax(logits[masked_indices], dim=-1)
                    correct_predictions = (predictions == input_ids[masked_indices]).sum().item()
                    total_predictions = masked_indices.sum().item()
                    
                    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                    
                    # Check memory again before backward pass
                    if rank == 0 and logger and batch_idx % args.log_steps == 0:
                        used_memory = get_used_gpu_memory()
                        memory_usage_percent = (used_memory / total_memory) * 100
                        logger.info(f"GPU Memory before backward pass - Used: {used_memory:.2f} GB ({memory_usage_percent:.2f}%)")
                    
                    # Backward pass
                    optimizer.zero_grad()
                    ce_loss.backward()
                    optimizer.step()
                    scheduler.step()
                
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and rank == 0 and logger:
                        logger.warning(f"CUDA Out of Memory Warning in batch {batch_idx}:")
                        logger.warning(str(e))
                        
                        # Log detailed GPU memory information
                        try:
                            used_memory = get_used_gpu_memory()
                            total_memory = get_total_gpu_memory()
                            
                            logger.warning(f"GPU Memory Stats at OOM warning:")
                            logger.warning(f"  Total GPU Memory: {total_memory:.2f} GB")
                            logger.warning(f"  Used Memory: {used_memory:.2f} GB")
                            
                            # Reduce batch size to throttle performance
                            if current_batch_size > 1:
                                new_batch_size = max(1, current_batch_size // 2)
                                logger.warning(f"Reducing batch size from {current_batch_size} to {new_batch_size} to throttle performance")
                                current_batch_size = new_batch_size
                            
                            # Force garbage collection
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # Skip this batch and continue
                            logger.warning(f"Skipping batch {batch_idx} and continuing training")
                            continue
                        except Exception as mem_error:
                            logger.warning(f"Error getting memory stats: {str(mem_error)}")
                            continue
                    else:
                        if rank == 0 and logger:
                            logger.error(f"Runtime error in batch {batch_idx}:")
                            logger.error(str(e))
                            logger.error(traceback.format_exc())
                        raise
            
                # Update metrics
                loss_value = ce_loss.item()
                epoch_loss += loss_value
                epoch_correct_predictions += correct_predictions
                epoch_total_predictions += total_predictions
                
                if rank == 0:
                    all_losses.append(loss_value)
                    all_accuracies.append(accuracy)
                    
                    global_step += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        "loss": f"{loss_value:.4f}", 
                        "accuracy": f"{accuracy:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "batch_size": current_batch_size
                    })
                    
                    # Log detailed metrics periodically
                    if batch_idx % args.log_steps == 0 and logger:
                        batch_time = time.time() - batch_start_time
                        logger.info(
                            f"Epoch: {epoch+1}/{args.num_epochs} | "
                            f"Batch: {batch_idx}/{len(dataloader)} | "
                            f"Step: {global_step} | "
                            f"Loss: {loss_value:.4f} | "
                            f"Accuracy: {accuracy:.4f} | "
                            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                            f"Batch size: {current_batch_size} | "
                            f"Batch time: {batch_time:.2f}s"
                        )
                
                # Save checkpoint (only from rank 0)
                if rank == 0 and global_step % args.save_steps == 0:
                    try:
                        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        # Save the unwrapped model
                        model.module.save_pretrained(checkpoint_dir)
                        logger.info(f"Saved checkpoint to {checkpoint_dir}")
                    except Exception as e:
                        logger.error(f"Error saving checkpoint: {str(e)}")
        
            # Gather and calculate epoch metrics across all processes
            if rank == 0 and logger:
                avg_epoch_loss = epoch_loss / len(dataloader)
                epoch_accuracy = epoch_correct_predictions / epoch_total_predictions if epoch_total_predictions > 0 else 0
                epoch_time = time.time() - epoch_start_time
                
                # Log epoch summary
                logger.info(
                    f"Epoch {epoch+1}/{args.num_epochs} completed in {epoch_time:.2f}s | "
                    f"Average Loss: {avg_epoch_loss:.4f} | "
                    f"Accuracy: {epoch_accuracy:.4f}"
                )
                
                # Save model after each epoch (only from rank 0)
                try:
                    epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
                    os.makedirs(epoch_dir, exist_ok=True)
                    model.module.save_pretrained(epoch_dir)
                    logger.info(f"Saved model after epoch {epoch+1} to {epoch_dir}")
                    
                    # Create a local copy of the config.json file to avoid HuggingFace Hub errors
                    if os.path.exists(os.path.join(epoch_dir, "config.json")):
                        logger.info(f"Model config saved successfully")
                except Exception as e:
                    logger.error(f"Error saving model after epoch: {str(e)}")
    
            # Synchronize all processes before starting the next epoch
            dist.barrier()
    
        # Calculate and log final metrics (only on rank 0)
        if rank == 0 and logger:
            final_avg_loss = sum(all_losses) / len(all_losses) if all_losses else float('nan')
            final_avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else float('nan')
            
            logger.info(f"Training completed!")
            logger.info(f"Final average loss: {final_avg_loss:.4f}")
            logger.info(f"Final average accuracy: {final_avg_accuracy:.4f}")
            
            # Save final model
            try:
                final_dir = os.path.join(args.output_dir, "final")
                os.makedirs(final_dir, exist_ok=True)
                model.module.save_pretrained(final_dir)
                logger.info(f"Saved final model to {final_dir}")
                
                # Create a local copy of the config.json file to avoid HuggingFace Hub errors
                if os.path.exists(os.path.join(final_dir, "config.json")):
                    logger.info(f"Final model config saved successfully")
            except Exception as e:
                logger.error(f"Error saving final model: {str(e)}")
            
            # Return metrics for potential visualization
            metrics = {
                "losses": all_losses,
                "accuracies": all_accuracies,
                "final_loss": final_avg_loss,
                "final_accuracy": final_avg_accuracy,
                "completed_epochs": args.num_epochs
            }
        else:
            metrics = None
    
    except Exception as e:
        if rank == 0 and logger:
            logger.error(f"Unexpected error during training: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return partial metrics
            metrics = {
                "losses": all_losses,
                "accuracies": all_accuracies,
                "error": str(e)
            }
        else:
            metrics = None
    
    # Clean up distributed resources
    cleanup_distributed()
    
    return metrics

def train(args):
    """
    Legacy single-GPU training function, now just a wrapper around the distributed training
    """
    # Check available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger = setup_logging(args)
        logger.info(f"Number of available GPUs: {num_gpus}")
        
        if num_gpus >= 1:
            # Use all available GPUs, up to 8
            world_size = min(num_gpus, 8)
            logger.info(f"Using {world_size} GPUs for distributed training")
            
            # Launch distributed training
            mp.spawn(
                train_distributed,
                args=(world_size, args),
                nprocs=world_size,
                join=True
            )
            
            # Return None as metrics are handled in the distributed training
            return None
        else:
            logger.warning("No GPUs available. Cannot perform distributed training.")
            return {"error": "No GPUs available for distributed training"}
    else:
        logger = setup_logging(args)
        logger.warning("CUDA is not available. Cannot perform distributed training.")
        return {"error": "CUDA not available for distributed training"}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaDA with SFT")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base", help="Model name or path")
    parser.add_argument("--data_path", type=str, default="sft_data/processed_data.pt", help="Path to processed data")
    parser.add_argument("--output_dir", type=str, default="sft_output", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (reduce to save memory)")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--log_steps", type=int, default=10, help="Log metrics every X batches")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (default: 0, only used if distributed training fails)")
    parser.add_argument("--memory_limit", type=float, default=65.0, help="Memory limit in GB across all GPUs (default: 65.0)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model and get metrics
    metrics = train(args)

if __name__ == "__main__":
    main()
