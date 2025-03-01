import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import time
from datetime import datetime

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

def train(args):
    # Set up logging
    logger = setup_logging(args)
    
    # Log training parameters
    logger.info(f"Starting training with parameters:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Check available GPUs and set device
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of available GPUs: {num_gpus}")
        
        if args.gpu_id >= 0 and args.gpu_id < num_gpus:
            device = torch.device(f"cuda:{args.gpu_id}")
            logger.info(f"Using device: {device}")
        else:
            logger.warning(f"Specified GPU ID {args.gpu_id} is not available. Available GPUs: {num_gpus}")
            logger.warning(f"Defaulting to CPU")
            device = torch.device("cpu")
    else:
        logger.warning("CUDA is not available. Using CPU")
        device = torch.device("cpu")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    start_time = time.time()
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(device)
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    dataset = SFTDataset(args.data_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    logger.info(f"Dataset loaded with {len(dataset)} examples")
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    
    num_training_steps = len(dataloader) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Training for {args.num_epochs} epochs, {num_training_steps} total steps")
    logger.info(f"Warmup for {num_warmup_steps} steps")
    
    # Training loop
    model.train()
    global_step = 0
    
    # Metrics tracking
    all_losses = []
    all_accuracies = []
    
    for epoch in range(args.num_epochs):
        logger.info(f"Starting Epoch {epoch+1}/{args.num_epochs}")
        epoch_start_time = time.time()
        
        epoch_loss = 0
        epoch_correct_predictions = 0
        epoch_total_predictions = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch_start_time = time.time()
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
            
            # Calculate accuracy for masked tokens
            predictions = torch.argmax(logits[masked_indices], dim=-1)
            correct_predictions = (predictions == input_ids[masked_indices]).sum().item()
            total_predictions = masked_indices.sum().item()
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Backward pass
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            loss_value = ce_loss.item()
            epoch_loss += loss_value
            epoch_correct_predictions += correct_predictions
            epoch_total_predictions += total_predictions
            
            all_losses.append(loss_value)
            all_accuracies.append(accuracy)
            
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss_value:.4f}", 
                "accuracy": f"{accuracy:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log detailed metrics periodically
            if batch_idx % args.log_steps == 0:
                batch_time = time.time() - batch_start_time
                logger.info(
                    f"Epoch: {epoch+1}/{args.num_epochs} | "
                    f"Batch: {batch_idx}/{len(dataloader)} | "
                    f"Step: {global_step} | "
                    f"Loss: {loss_value:.4f} | "
                    f"Accuracy: {accuracy:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Batch time: {batch_time:.2f}s"
                )
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Calculate epoch metrics
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_accuracy = epoch_correct_predictions / epoch_total_predictions if epoch_total_predictions > 0 else 0
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch summary
        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs} completed in {epoch_time:.2f}s | "
            f"Average Loss: {avg_epoch_loss:.4f} | "
            f"Accuracy: {epoch_accuracy:.4f}"
        )
        
        # Save model after each epoch
        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        logger.info(f"Saved model after epoch {epoch+1} to {epoch_dir}")
    
    # Calculate and log final metrics
    final_avg_loss = sum(all_losses) / len(all_losses)
    final_avg_accuracy = sum(all_accuracies) / len(all_accuracies)
    
    logger.info(f"Training completed!")
    logger.info(f"Final average loss: {final_avg_loss:.4f}")
    logger.info(f"Final average accuracy: {final_avg_accuracy:.4f}")
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    logger.info(f"Saved final model to {final_dir}")
    
    # Return metrics for potential visualization
    return {
        "losses": all_losses,
        "accuracies": all_accuracies,
        "final_loss": final_avg_loss,
        "final_accuracy": final_avg_accuracy
    }

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaDA with SFT")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base", help="Model name or path")
    parser.add_argument("--data_path", type=str, default="sft_data/processed_data.pt", help="Path to processed data")
    parser.add_argument("--output_dir", type=str, default="sft_output", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--log_steps", type=int, default=10, help="Log metrics every X batches")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (default: 0)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model and get metrics
    metrics = train(args)

if __name__ == "__main__":
    main()
