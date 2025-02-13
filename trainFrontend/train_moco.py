"""
Training script for MOCO v2 frontend model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm

from datamodule.av_dataset import AVDataset
from models.moco_visual_frontend import MoCoVisualFrontend
from utils.general import num_params

def train_moco(config):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        filename='moco_training.log',
        filemode='w'
    )
    logger = logging.getLogger(__name__)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = MoCoVisualFrontend(
        d_model=config["model"]["d_model"],
        num_classes=config["model"].get("num_classes", 500),  # Default for LRW dataset
        frame_length=config["model"]["required_input_length"],
        moco_path=config["data"]["moco_file"],
        feature_dim=config["model"].get("feature_dim", 2048)
    )
    model = model.to(device)
    
    # Load pretrained weights if specified
    if config.get("pretrained_path"):
        logger.info(f"Loading pretrained weights from {config['pretrained_path']}")
        state_dict = torch.load(config["pretrained_path"], map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    # Initialize dataset and dataloader
    train_dataset = AVDataset(
        data_dir=config["data"]["root_dir"],
        split="train",
        modality="video"  # Only load video data for MOCO training
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )
    
    # Setup optimizer with warmup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["max_lr"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Cosine annealing scheduler with warmup
    num_training_steps = len(train_loader) * config["training"]["epochs"]
    num_warmup_steps = int(num_training_steps * config["training"]["warmup_ratio"])
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["training"]["max_lr"],
        total_steps=num_training_steps,
        pct_start=config["training"]["warmup_ratio"],
        anneal_strategy='cos',
        final_div_factor=config["training"]["max_lr"]/config["training"]["min_lr"]
    )
    
    # Loss function (InfoNCE loss for contrastive learning)
    criterion = nn.CrossEntropyLoss()
    
    # Setup tensorboard
    writer = SummaryWriter(os.path.join(config["output"]["log_dir"], "moco"))
    
    # Log model info
    num_total_params, num_trainable_params = num_params(model)
    logger.info(f"Total parameters: {num_total_params}")
    logger.info(f"Trainable parameters: {num_trainable_params}")
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Get video input
                video = batch["video"].to(device)
                
                # Forward pass
                q, k = model(video)  # Get query and key embeddings
                
                # Compute InfoNCE loss
                logits = torch.mm(q, k.t())  # Compute similarity matrix
                labels = torch.arange(logits.shape[0], device=device)  # Diagonal is positive pairs
                loss = criterion(logits / model.T, labels)  # Temperature-scaled cross entropy
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["gradient_clip_val"]
                )
                
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                # Log to tensorboard
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
        
        # Save checkpoint if loss improved
        avg_epoch_loss = total_loss / len(train_loader)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            checkpoint_path = os.path.join(
                config["output"]["checkpoint_dir"],
                f"moco_best_epoch{epoch+1}_loss{avg_epoch_loss:.4f}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best model checkpoint to {checkpoint_path}")
        
        # Log epoch metrics
        logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
    
    writer.close()
    logger.info("Training completed!")

if __name__ == "__main__":
    from config import get_config
    config = get_config()
    train_moco(config)
