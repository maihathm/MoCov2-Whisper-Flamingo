"""
Training script for Whisper frontend model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import WhisperModel, WhisperConfig, WhisperFeatureExtractor
import logging
from tqdm import tqdm

from datamodule.av_dataset import AVDataset
from utils.general import num_params

def train_whisper(config):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        filename='whisper_training.log',
        filemode='w'
    )
    logger = logging.getLogger(__name__)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Whisper model and feature extractor
    whisper_config = WhisperConfig.from_pretrained(config["whisper"]["model_name"])
    model = WhisperModel(whisper_config)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config["whisper"]["model_name"])
    
    # Load pretrained weights
    logger.info(f"Loading pretrained Whisper model: {config['whisper']['model_name']}")
    model = WhisperModel.from_pretrained(config["whisper"]["model_name"])
    model = model.to(device)
    
    # Freeze encoder if specified
    if config["whisper"]["freeze_encoder"]:
        logger.info("Freezing Whisper encoder layers")
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    # Initialize dataset and dataloader
    train_dataset = AVDataset(
        data_dir=config["data"]["root_dir"],
        split="train",
        modality="audio"  # Only load audio data for Whisper training
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
        filter(lambda p: p.requires_grad, model.parameters()),
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
    
    # Loss function (MSE for feature learning)
    criterion = nn.MSELoss()
    
    # Setup tensorboard
    writer = SummaryWriter(os.path.join(config["output"]["log_dir"], "whisper"))
    
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
                # Get audio input
                audio = batch["audio"].to(device)
                
                # Process audio through feature extractor
                with torch.no_grad():
                    features = feature_extractor(
                        audio.squeeze(1).cpu().numpy(),
                        sampling_rate=16000,
                        return_tensors="pt"
                    ).input_features.to(device)
                
                # Forward pass
                outputs = model.encoder(features)
                hidden_states = outputs.last_hidden_state
                
                # Compute reconstruction loss
                reconstructed = model.encoder.proj_out(hidden_states)
                loss = criterion(reconstructed, features)
                
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
                f"whisper_best_epoch{epoch+1}_loss{avg_epoch_loss:.4f}.pt"
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
    train_whisper(config)
