import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from config import get_config
from datamodule.data_module import DataModule
from models.av_net import AVNet
from jiwer import wer
from transformers import WhisperProcessor
import logging

logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class AVSRModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Model configuration
        model_args = (
            config["model"]["d_model"],
            config["model"]["n_heads"],
            config["model"]["n_layers"],
            config["model"]["pe_max_len"],
            config["model"]["fc_hidden_size"],
            config["model"]["dropout"]
        )
        
        # Initialize model
        self.model = AVNet(
            modal=config["data"]["modality"],
            MoCofile=os.path.join(os.getcwd(), config["data"]["moco_file"]),
            reqInpLen=config["model"]["required_input_length"],
            modelargs=model_args,
            vocab_size=WhisperProcessor.from_pretrained("openai/whisper-small").tokenizer.vocab_size,
            enable_logging=config["output"]["enable_logging"]
        )
        
        # Loss functions
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=config["training"]["label_smoothing"])
        
        # Initialize tokenizer
        tokenizer_dir = config["data"].get("updated_tokenizer_dir", None)
        if tokenizer_dir and os.path.exists(tokenizer_dir):
            self.tokenizer_name = tokenizer_dir
        else:
            self.tokenizer_name = "openai/whisper-small"
            if tokenizer_dir:
                os.makedirs(tokenizer_dir, exist_ok=True)
                WhisperProcessor.from_pretrained("openai/whisper-small").save_pretrained(tokenizer_dir)
        
        self.processor = WhisperProcessor.from_pretrained(self.tokenizer_name)
        self.vocab_size = self.processor.tokenizer.vocab_size
    
    def _compute_ctc_loss(self, logits, targets, input_lengths, target_lengths):
        log_probs = F.log_softmax(logits, dim=-1)
        loss = self.ctc_loss(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
        return loss

    def _compute_ce_loss(self, logits, targets):
        return self.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

    def _decode_predictions(self, logits):
        predictions = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predictions, skip_special_tokens=True)

    def training_step(self, batch, batch_idx):
        # Forward pass
        input_data = (
            batch["audio"], batch["audio_mask"],
            batch["video"], batch["video_mask"]
        )
        logits = self.model(input_data)
        
        # Compute losses
        ctc_loss = self._compute_ctc_loss(
            logits, 
            batch["target_ids"],
            batch["audio_lengths"],
            batch["target_lengths"]
        )
        ce_loss = self._compute_ce_loss(logits, batch["target_ids"])
        
        # Combined loss
        loss = ctc_loss + ce_loss
        
        # Logging
        self.log("train/ctc_loss", ctc_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass
        input_data = (
            batch["audio"], batch["audio_mask"],
            batch["video"], batch["video_mask"]
        )
        logits = self.model(input_data)
        
        # Compute losses
        ctc_loss = self._compute_ctc_loss(
            logits,
            batch["target_ids"],
            batch["audio_lengths"],
            batch["target_lengths"]
        )
        ce_loss = self._compute_ce_loss(logits, batch["target_ids"])
        loss = ctc_loss + ce_loss
        
        # Compute WER
        predictions = self._decode_predictions(logits)
        val_wer = wer(batch["target_text"], predictions)
        
        # Logging
        self.log("val/ctc_loss", ctc_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/wer", val_wer, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"val_loss": loss, "val_wer": val_wer}

    def test_step(self, batch, batch_idx):
        # Forward pass
        input_data = (
            batch["audio"], batch["audio_mask"],
            batch["video"], batch["video_mask"]
        )
        logits = self.model(input_data)
        
        # Compute WER
        predictions = self._decode_predictions(logits)
        test_wer = wer(batch["target_text"], predictions)
        
        # Logging
        self.log("test/wer", test_wer, on_step=False, on_epoch=True)
        
        if self.config["output"]["save_predictions"]:
            return {
                "predictions": predictions,
                "targets": batch["target_text"],
                "wer": test_wer
            }
        return {"test_wer": test_wer}

    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["training"]["max_lr"],
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=self.config["training"]["weight_decay"]
        )
        
        # Learning rate scheduler
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(num_training_steps * self.config["training"]["warmup_ratio"])
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config["training"]["max_lr"],
            total_steps=num_training_steps,
            pct_start=self.config["training"]["warmup_ratio"],
            div_factor=25.0,
            final_div_factor=1e4,
            anneal_strategy='linear'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def on_save_checkpoint(self, checkpoint):
        # Save tokenizer state
        self.processor.save_pretrained(self.tokenizer_name)
        checkpoint["tokenizer_dir"] = self.tokenizer_name
        
        # Save model configuration
        checkpoint["config"] = self.config
        
        # Log checkpoint saving
        logger.info(f"Saving checkpoint to {self.tokenizer_name}")

    def on_load_checkpoint(self, checkpoint):
        # Load tokenizer
        if "tokenizer_dir" in checkpoint:
            self.tokenizer_name = checkpoint["tokenizer_dir"]
            self.processor = WhisperProcessor.from_pretrained(self.tokenizer_name)
            logger.info(f"Loaded tokenizer from {self.tokenizer_name}")
        
        # Load configuration
        if "config" in checkpoint:
            self.config.update(checkpoint["config"])
            logger.info("Loaded model configuration from checkpoint")


def main():
    # Load configuration
    config = get_config()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize data module and model
    logger.info("Initializing data module and model...")
    data_module = DataModule(config)
    model = AVSRModule(config)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config["output"]["checkpoint_dir"],
            filename="avsr-{epoch:02d}-{val_loss:.2f}-{val_wer:.3f}",
            save_top_k=config["output"]["save_top_k"],
            monitor=config["output"]["monitor"],
            mode=config["output"]["monitor_mode"],
            save_last=True
        ),
        EarlyStopping(
            monitor=config["output"]["monitor"],
            patience=config["training"]["early_stopping_patience"],
            mode=config["output"]["monitor_mode"],
            verbose=True
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Configure logger
    tb_logger = TensorBoardLogger(
        save_dir=config["output"]["log_dir"],
        name="avsr_logs",
        default_hp_metric=False
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        logger=tb_logger,
        precision="16-mixed",
        accelerator="auto",
        devices="auto",
        strategy=DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else "auto",
        enable_progress_bar=True,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        log_every_n_steps=config["output"]["log_every_n_steps"],
        deterministic=True
    )
    
    # Train and test
    logger.info("Starting training...")
    trainer.fit(model, datamodule=data_module)
    
    logger.info("Starting testing...")
    trainer.test(model, datamodule=data_module)
    
    logger.info("Training and testing completed!")

if __name__ == "__main__":
    main()
