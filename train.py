import os
import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
import logging
from config import get_config
from datamodule.data_module import DataModule
from models.av_net import AVNet
from utils.logging_utils import setup_logging, log_tensor_info
from pytorch_lightning.strategies import DDPStrategy
# Setup logging
setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,6'
def is_deepcopyable(obj):
    try:
        copy.deepcopy(obj)
        return True
    except Exception:
        return False


class AVSRModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hparams = {}
        for section, params in config.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    if is_deepcopyable(value):
                        hparams[f"{section}_{key}"] = value
            else:
                if is_deepcopyable(params):
                    hparams[section] = params
        self.save_hyperparameters(hparams)
        
        # Model initialization
        model_args = (
            config["model"]["d_model"],
            config["model"]["n_heads"],
            config["model"]["n_layers"],
            config["model"]["pe_max_len"],
            config["model"]["fc_hidden_size"],
            config["model"]["dropout"]
        )
        
        self.model = AVNet(
            modal="AV",
            MoCofile=os.path.join(os.getcwd(), config["data"]["moco_file"]),
            reqInpLen=config["model"]["required_input_length"],
            modelargs=model_args
        )
        
        # MSE Loss for feature learning
        self.loss_fn = nn.MSELoss()
        
    def training_step(self, batch, batch_idx):
        logger.debug(f"\n{'='*50}\nStarting training step {batch_idx}\n{'='*50}")
        
        
        # Prepare input data
        input_data = (
            batch["audios"],
            batch["audio_attention_mask"],
            batch["videos"],
            batch["video_attention_mask"]
        )
        
        features = self.model(input_data)
        
        # Calculate MSE loss between audio and video features
        features, target_audio = self.model(input_data, return_audio=True)
        loss = self.loss_fn(features, target_audio)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log gate attention weights
        if hasattr(self.model, 'fusion_module'):
            for i, layer in enumerate(self.model.fusion_module.layers):
                self.log(f'train_attn_gate_{i}', layer.attn_gate.item(), on_step=False, on_epoch=True)
                self.log(f'train_ff_gate_{i}', layer.ff_gate.item(), on_step=False, on_epoch=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        logger.debug(f"\n{'='*50}\nStarting validation step {batch_idx}\n{'='*50}")
        
        # Chuẩn bị dữ liệu
        input_data = (
            batch["audios"],
            batch["audio_attention_mask"],
            batch["videos"],
            batch["video_attention_mask"]
        )
        
        # Lấy cả kết quả của fusion module và audio embedding đã được xử lý
        features, target_audio = self.model(input_data, return_audio=True)
        
        # Tính loss giữa các embedding (đều có shape [B, T, dModel])
        loss = self.loss_fn(features, target_audio)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Tính cosine similarity (nếu cần)
        cos_sim = F.cosine_similarity(features.mean(1), target_audio.mean(1))
        self.log('val_cosine_sim', cos_sim.mean(), on_epoch=True, prog_bar=True)
        
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
            lr=0.0,  # Will be set by the scheduler
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



class DebugCallback(pl.Callback):
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        logger.debug(f"\n{'='*50}\nStarting validation batch {batch_idx}\n{'='*50}")
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        logger.debug(f"Completed validation batch {batch_idx}")
        if batch_idx == 0:  # Log detailed info for first batch only
            logger.debug("First batch validation results:")
            log_tensor_info("Model outputs", outputs)

def main():
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = get_config()


    # Initialize data module and model
    data_module = DataModule(config)
    model = AVSRModule(config)

    # Setup callbacks with debug callback
    callbacks = [
        DebugCallback(),
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
    # Setup logger with detailed metrics
    tb_logger = TensorBoardLogger(
        save_dir=config["output"]["log_dir"],
        name='avsr_logs',
        default_hp_metric=False
    )
    # Initialize trainer with improved settings and logging
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        logger=tb_logger,  # Use the TensorBoard logger here
        gradient_clip_val=config["training"]["gradient_clip_val"],
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 1),
        precision="16-mixed", 
        accelerator='auto',
        devices="auto",
        strategy=DDPStrategy(find_unused_parameters=True),
        benchmark=True,
        # sync_batchnorm=True,
        # use_distributed_sampler=False,
        # replace_sampler_ddp=False,
        enable_progress_bar=True,  # Enable progress bar for better monitoring
    )
    
    trainer.fit(model, data_module)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()
