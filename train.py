import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config import get_config
from datamodule.data_module import DataModule
from models.av_net import AVNet


class AVSRModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Model initialization
        model_args = (
            config["model"]["d_model"],
            config["model"]["n_heads"],
            config["model"]["n_layers"],
            config["model"]["pe_max_len"],
            config["model"]["fc_hidden_size"],
            config["model"]["dropout"],
            config["model"]["num_classes"]
        )
        
        self.model = AVNet(
            modal="AV",
            MoCofile=os.path.join(os.getcwd(), config["data"]["moco_file"]),
            reqInpLen=config["model"]["required_input_length"],
            modelargs=model_args
        )
        
        # Loss functions with label smoothing
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=0.1
        )
        
        # Modality dropout probabilities
        self.prob_av = 0.5  # Probability of using both modalities
        self.prob_a = 0.25  # Probability of using only audio
        # Remaining 0.25 probability is for using only video
        
    def _apply_modality_dropout(self, audio_features, video_features):
        """Apply modality dropout during training"""
        if self.training:
            rand_val = torch.rand(1).item()
            if rand_val <= self.prob_av:
                return audio_features, video_features  # Use both
            elif rand_val <= self.prob_av + self.prob_a:
                return audio_features, None  # Use only audio
            else:
                return None, video_features  # Use only video
        return audio_features, video_features
        
    def training_step(self, batch, batch_idx):
        # Prepare input data
        input_data = (
            batch["audio"],
            batch["audio_mask"],
            batch["video"],
            batch["video_lengths"]
        )
        
        # Forward pass with modality dropout
        logits = self.model(
            input_data,
            batch["targets"],
            batch["target_lengths"]
        )
        
        # Calculate loss
        B, T, V = logits.shape
        loss = self.loss_fn(
            logits.view(-1, V),
            batch["targets"].view(-1)
        )
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log gate attention weights
        if hasattr(self.model, 'fusion_module'):
            for i, layer in enumerate(self.model.fusion_module.layers):
                self.log(f'train_attn_gate_{i}', layer.attn_gate.item(), on_step=False, on_epoch=True)
                self.log(f'train_ff_gate_{i}', layer.ff_gate.item(), on_step=False, on_epoch=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        # Forward pass without modality dropout
        input_data = (
            batch["audio"],
            batch["audio_mask"],
            batch["video"],
            batch["video_lengths"]
        )
        
        logits = self.model(
            input_data,
            batch["targets"],
            batch["target_lengths"]
        )
        
        # Calculate loss
        B, T, V = logits.shape
        loss = self.loss_fn(
            logits.view(-1, V),
            batch["targets"].view(-1)
        )
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        correct = (predictions == batch["targets"]).masked_fill(batch["targets"] == -100, 0)
        accuracy = correct.sum() / (batch["targets"] != -100).sum()
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        
        return loss
        
    def configure_optimizers(self):
        # Separate parameters for weight decay
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'gate']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config["training"]["weight_decay"]
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=0.0,  # Will be set by scheduler
            betas=(0.9, 0.98),
            eps=1e-6
        )
        
        # Learning rate scheduler with warmup
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(num_training_steps * 0.1)  # 10% warmup
        
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / 
                float(max(1, num_training_steps - num_warmup_steps))
            )
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
            last_epoch=-1
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


def main():
    # Load configuration
    config = get_config()
    
    # Initialize data module and model
    data_module = DataModule(config)
    model = AVSRModule(config)
    
    # Setup callbacks
    callbacks = [
        # Model checkpointing
        ModelCheckpoint(
            dirpath=config["output"]["checkpoint_dir"],
            filename='avsr-{epoch:02d}-{val_loss:.2f}',
            save_top_k=config["output"]["save_top_k"],
            monitor=config["output"]["monitor"],
            mode=config["output"]["monitor_mode"]
        ),
        # Early stopping
        EarlyStopping(
            monitor=config["output"]["monitor"],
            patience=config["training"]["early_stopping_patience"],
            mode=config["output"]["monitor_mode"]
        ),
        # Learning rate monitor
        pl.callbacks.LearningRateMonitor(logging_interval='step')
    ]
    
    # Setup logger with more detailed metrics
    logger = TensorBoardLogger(
        save_dir=config["output"]["log_dir"],
        name='avsr_logs',
        default_hp_metric=False
    )
    
    # Initialize trainer with improved settings
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 1),
        precision=16,  # Use mixed precision training
        accelerator='auto',
        devices='auto',
        strategy='ddp' if torch.cuda.device_count() > 1 else None,
        deterministic=True,
        benchmark=True
    )
    
    # Train model
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
