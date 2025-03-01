import os
import torch
import argparse
import torch.nn.functional as F
import logging
import time
import gc
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# Configure logging
def setup_logging(log_file="training_log.log", console_level=logging.INFO, file_level=logging.DEBUG):
    """Set up logging with different levels for file and console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(min(console_level, file_level))  # Set to the more detailed level
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler - captures everything including debug messages
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - only shows info and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging(log_file="alignment_training_log.log", console_level=logging.INFO, file_level=logging.DEBUG)

class AlignmentDataset(Dataset):
    """Dataset for alignment fine-tuning of LLaDA with LLaMA logits."""
    
    def __init__(self, data_path):
        logger.info(f"Loading dataset from {data_path}")
        start_time = time.time()
        self.data = torch.load(data_path)
        logger.info(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
        
        # Log dataset statistics
        logger.info(f"Dataset size: {len(self.data)} examples")
        
        # Only calculate statistics if there are examples
        if len(self.data) > 0:
            lengths = [item["input_ids"].shape[0] for item in self.data]
            
            # Handle both tensor and integer types for prompt_length
            prompt_lengths = []
            for item in self.data:
                if hasattr(item["prompt_length"], "item"):
                    prompt_lengths.append(item["prompt_length"].item())
                else:
                    prompt_lengths.append(item["prompt_length"])
            
            logger.info(f"Sequence length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.2f}")
            logger.info(f"Prompt length stats: min={min(prompt_lengths)}, max={max(prompt_lengths)}, avg={sum(prompt_lengths)/len(prompt_lengths):.2f}")
        else:
            logger.error("Dataset is empty. Please check the preprocessing step.")
            raise ValueError("Dataset is empty. Cannot proceed with training.")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    Pads sequences to the maximum length in the batch.
    """
    max_len = max([item["input_ids"].shape[0] for item in batch])
    vocab_size = batch[0]["llama_logits"].shape[1]
    
    # Initialize tensors
    input_ids = torch.full((len(batch), max_len), 126081, dtype=torch.long)  # EOS token for padding
    prompt_lengths = torch.zeros(len(batch), dtype=torch.long)
    llama_logits = torch.zeros((len(batch), max_len, vocab_size), dtype=torch.float32)
    
    # Fill tensors
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        prompt_lengths[i] = item["prompt_length"]
        llama_logits[i, :seq_len] = item["llama_logits"]
    
    return {
        "input_ids": input_ids,
        "prompt_lengths": prompt_lengths,
        "llama_logits": llama_logits
    }

def forward_process(input_ids, eps=1e-3):
    """
    Apply the forward process to add noise to the input.
    
    Args:
        input_ids: Input token IDs
        eps: Small epsilon to avoid division by zero
        
    Returns:
        noisy_batch: Input with noise applied
        masked_indices: Boolean tensor indicating which tokens are masked
        p_mask: Probability of masking for each token
    """
    start_time = time.time()
    b, l = input_ids.shape
    
    # Log input shape
    logger.debug(f"Forward process input shape: batch_size={b}, seq_length={l}")
    
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    
    # Check for NaN or inf values in p_mask
    if torch.isnan(p_mask).any() or torch.isinf(p_mask).any():
        logger.error(f"NaN or inf values detected in p_mask: {p_mask}")
    
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    
    # Log masking statistics
    mask_percentage = masked_indices.float().mean().item() * 100
    logger.debug(f"Masking {mask_percentage:.2f}% of tokens")
    
    # 126336 is used for [MASK] token
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    
    logger.debug(f"Forward process completed in {time.time() - start_time:.4f} seconds")
    return noisy_batch, masked_indices, p_mask

def log_top_predictions(logits, token_ids, tokenizer, top_k=5, num_examples=2):
    """
    Log the top-k predictions for a few examples to monitor alignment.
    
    Args:
        logits: Logits tensor of shape [batch_size, seq_length, vocab_size]
        token_ids: Token IDs tensor of shape [batch_size, seq_length]
        tokenizer: Tokenizer for decoding token IDs
        top_k: Number of top predictions to log
        num_examples: Number of examples to log
    """
    batch_size, seq_length, vocab_size = logits.shape
    num_examples = min(num_examples, batch_size)
    
    for i in range(num_examples):
        for j in range(seq_length):
            # Skip padding tokens
            if token_ids[i, j] == 126081:  # EOS token used for padding
                continue
                
            # Get the top-k predictions
            top_logits, top_indices = torch.topk(logits[i, j], top_k)
            
            # Log the top-k predictions
            logger.info(f"Example {i+1}, Position {j+1}, Token: '{tokenizer.decode([token_ids[i, j]])}', ID: {token_ids[i, j]}")
            for k, (idx, logit) in enumerate(zip(top_indices, top_logits)):
                token = tokenizer.decode([idx])
                logger.info(f"  {k+1}. Token: '{token}', ID: {idx}, Logit: {logit:.4f}")
            logger.info("")

def train_epoch(model, dataloader, optimizer, scheduler, tokenizer, device, epoch, temperature=1.0, log_interval=100):
    """Train for one epoch using KL divergence loss for alignment."""
    model.train()
    total_loss = 0
    batch_times = []
    
    # Log memory usage at the start of the epoch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)    # GB
        logger.info(f"Epoch {epoch} - Starting memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    epoch_start_time = time.time()
    
    for batch_idx, batch in enumerate(progress_bar):
        batch_start_time = time.time()
        logger.info(f"Epoch {epoch} - Processing batch {batch_idx+1}/{len(dataloader)}")
        
        try:
            # Move data to device
            data_to_device_start = time.time()
            input_ids = batch["input_ids"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)
            llama_logits = batch["llama_logits"].to(device)
            logger.debug(f"Data to device time: {time.time() - data_to_device_start:.4f}s")
            
            # Log batch statistics
            batch_size, seq_length = input_ids.shape
            logger.info(f"Batch shape: batch_size={batch_size}, seq_length={seq_length}")
            
            # Apply forward process
            forward_process_start = time.time()
            noisy_batch, _, _ = forward_process(input_ids)
            logger.debug(f"Forward process time: {time.time() - forward_process_start:.4f}s")
            
            # Do not add noise to the prompt
            prompt_mask_start = time.time()
            token_positions = torch.arange(noisy_batch.shape[1], device=device).expand(noisy_batch.size(0), noisy_batch.size(1))
            prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
            noisy_batch[prompt_mask] = input_ids[prompt_mask]
            logger.debug(f"Prompt masking time: {time.time() - prompt_mask_start:.4f}s")
            
            # Create response mask (inverse of prompt mask)
            response_mask = ~prompt_mask
            
            # Forward pass
            forward_start = time.time()
            outputs = model(input_ids=noisy_batch)
            logits = outputs.logits
            
            # Check for NaN or inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.error(f"NaN or inf values detected in logits")
                nan_count = torch.isnan(logits).sum().item()
                inf_count = torch.isinf(logits).sum().item()
                logger.error(f"NaN count: {nan_count}, Inf count: {inf_count}")
            
            logger.debug(f"Forward pass time: {time.time() - forward_start:.4f}s")
            
            # Calculate KL divergence loss over response tokens
            loss_start = time.time()
            
            # Extract logits for response tokens
            llada_logits = logits[response_mask]  # Shape: [num_response_tokens, vocab_size]
            llama_logits_masked = llama_logits[response_mask]  # Shape: [num_response_tokens, vocab_size]
            
            # Apply temperature scaling
            llada_probs = F.softmax(llada_logits / temperature, dim=-1)
            llama_probs = F.softmax(llama_logits_masked / temperature, dim=-1)

            common_vocab_size = min(llada_logits.size(-1), llama_probs.size(-1))

            # Slice both outputs to the common vocabulary size
            student_logits_common = llada_logits[..., :common_vocab_size]
            teacher_probs_common = llama_probs[..., :common_vocab_size]
            
            # Calculate KL divergence loss
            kl_loss_per_token = F.kl_div(
                F.log_softmax(student_logits_common / temperature, dim=-1),
                teacher_probs_common,
                reduction='none'
            ).sum(-1)
            
            # Normalize the loss by the number of response tokens
            response_lengths = response_mask.sum(dim=1)
            kl_loss = kl_loss_per_token.sum() / response_lengths.sum()
            
            # Check for NaN or inf in final loss
            if torch.isnan(kl_loss).any() or torch.isinf(kl_loss).any():
                logger.error(f"NaN or inf values detected in final loss: {kl_loss.item()}")
            
            logger.debug(f"Loss calculation time: {time.time() - loss_start:.4f}s")
            
            # Backward pass
            backward_start = time.time()
            optimizer.zero_grad()
            kl_loss.backward()
            
            # Check for exploding/vanishing gradients
            grad_norm = 0.0
            max_grad = 0.0
            min_grad = float('inf')
            has_nan_or_inf = False
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        has_nan_or_inf = True
                        logger.error(f"NaN or inf gradient detected in {name}")
                    
                    grad_norm += grad.norm().item() ** 2
                    max_grad = max(max_grad, grad.abs().max().item())
                    min_grad = min(min_grad, grad.abs().min().item() if grad.abs().min().item() > 0 else min_grad)
            
            grad_norm = grad_norm ** 0.5
            logger.info(f"Gradient stats - norm: {grad_norm:.6f}, max: {max_grad:.6f}, min: {min_grad:.6f}")
            
            if has_nan_or_inf:
                logger.error("NaN or inf gradients detected, skipping optimizer step")
                continue
            
            optimizer.step()
            scheduler.step()
            logger.debug(f"Backward pass time: {time.time() - backward_start:.4f}s")
            
            # Log top predictions periodically
            if batch_idx % log_interval == 0:
                log_top_predictions(logits, input_ids, tokenizer)
            
            # Log memory usage
            if torch.cuda.is_available() and batch_idx % 10 == 0:
                torch.cuda.synchronize()
                memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
                memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)    # GB
                logger.info(f"Memory usage: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            
            # Update metrics
            loss_value = kl_loss.item()
            total_loss += loss_value
            
            # Log batch completion
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            logger.info(f"Batch {batch_idx+1} completed in {batch_time:.2f}s with KL loss: {loss_value:.6f}")
            
            # Force garbage collection every few batches
            if batch_idx % 5 == 0:
                gc_start = time.time()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.debug(f"Garbage collection time: {time.time() - gc_start:.4f}s")
            
            progress_bar.set_postfix({"kl_loss": loss_value, "batch_time": f"{batch_time:.2f}s"})
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx+1}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Log epoch statistics
    epoch_time = time.time() - epoch_start_time
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s, average batch time: {avg_batch_time:.2f}s")
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch} average KL loss: {avg_loss:.6f}")
    
    return avg_loss

def log_system_info():
    """Log system information for debugging purposes."""
    logger.info("=== System Information ===")
    
    # PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # CUDA information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: Yes")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {i} name: {torch.cuda.get_device_name(i)}")
            logger.info(f"CUDA device {i} capability: {torch.cuda.get_device_capability(i)}")
            
        # Memory information
        device = torch.device("cuda")
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)    # GB
        logger.info(f"CUDA memory allocated: {memory_allocated:.2f} GB")
        logger.info(f"CUDA memory reserved: {memory_reserved:.2f} GB")
        logger.info(f"CUDA max memory allocated: {torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f} GB")
    else:
        logger.info("CUDA available: No")
    
    # CPU information
    import platform
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"OS: {platform.system()} {platform.release()}")
    
    try:
        import psutil
        logger.info(f"CPU count: {psutil.cpu_count(logical=True)} logical, {psutil.cpu_count(logical=False)} physical")
        memory = psutil.virtual_memory()
        logger.info(f"RAM: {memory.total / (1024 ** 3):.2f} GB total, {memory.available / (1024 ** 3):.2f} GB available")
    except ImportError:
        logger.info("psutil not available for detailed CPU/RAM info")
    
    logger.info("=== End System Information ===")

def main():
    # Start timing the entire training process
    total_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Fine-tune LLaDA model for alignment with LLaMA")
    parser.add_argument("--data_path", type=str, default="sft_data/processed_alignment.pt",
                        help="Path to processed alignment data file")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="alignment_output",
                        help="Directory to save fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Number of warmup steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X steps")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for gradient clipping")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for softmax in KL divergence calculation")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log top predictions every X batches")
    
    args = parser.parse_args()
    
    # Set logging level based on argument
    logger.setLevel(getattr(logging, args.log_level))
    
    # Log all arguments
    logger.info("=== Training Arguments ===")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    # Log system information
    log_system_info()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_name}...")
    model_load_start = time.time()
    
    try:
        model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        
        logger.info(f"Model loaded in {time.time() - model_load_start:.2f} seconds")
        
        # Log model information
        logger.info("=== Model Information ===")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Log model architecture summary
        logger.info("Model architecture:")
        for name, module in model.named_children():
            logger.info(f"  {name}: {type(module).__name__}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}...")
    dataset_load_start = time.time()
    
    try:
        dataset = AlignmentDataset(args.data_path)
        logger.info(f"Dataset loaded in {time.time() - dataset_load_start:.2f} seconds")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0,  # Safer for debugging
            pin_memory=torch.cuda.is_available()  # Speed up data transfer to GPU
        )
        
        logger.info(f"Created DataLoader with {len(dataloader)} batches")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Setup optimizer and scheduler
    logger.info("Setting up optimizer and scheduler...")
    
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8  # For numerical stability
        )
        
        total_steps = len(dataloader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Optimizer: AdamW with lr={args.learning_rate}, weight_decay={args.weight_decay}")
        logger.info(f"Scheduler: Linear warmup ({args.warmup_steps} steps) and decay over {total_steps} total steps")
    except Exception as e:
        logger.error(f"Error setting up optimizer: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Training loop
    logger.info("=== Starting Training ===")
    training_start = time.time()
    
    try:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
            
            # Train for one epoch
            avg_loss = train_epoch(
                model, 
                dataloader, 
                optimizer, 
                scheduler, 
                tokenizer,
                device, 
                epoch + 1,
                temperature=args.temperature,
                log_interval=args.log_interval
            )
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")
            logger.info(f"Epoch {epoch + 1}/{args.epochs} - Average KL Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            save_start = time.time()
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            
            try:
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path} in {time.time() - save_start:.2f} seconds")
            except Exception as e:
                logger.error(f"Error saving checkpoint: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Save final model regardless of how training ended
        try:
            logger.info("Saving final model...")
            final_path = os.path.join(args.output_dir, "final")
            model.save_pretrained(final_path)
            tokenizer.save_pretrained(final_path)
            logger.info(f"Saved final model to {final_path}")
        except Exception as e:
            logger.error(f"Error saving final model: {str(e)}")
    
    # Log total training time
    total_time = time.time() - total_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info("=== Training Complete ===")

if __name__ == "__main__":
    main()
