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

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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
            modelargs=model_args,
            vocab_size=WhisperProcessor.from_pretrained("openai/whisper-small").tokenizer.vocab_size,
            enable_logging=config["output"]["enable_logging"]
        )
        
        self.loss_fn = nn.MSELoss()

        tokenizer_dir = config["data"].get("updated_tokenizer_dir", None)
        if tokenizer_dir and os.path.exists(tokenizer_dir):
            self.tokenizer_name = tokenizer_dir
        else:
            self.tokenizer_name = "openai/whisper-small"
            if tokenizer_dir:
                os.makedirs(tokenizer_dir, exist_ok=True)
                WhisperProcessor.from_pretrained("openai/whisper-small").save_pretrained(tokenizer_dir)
        
        self.processor = WhisperProcessor.from_pretrained(self.tokenizer_name)
        vocab_size = self.processor.tokenizer.vocab_size
        self.text_decoder = nn.Linear(config["model"]["d_model"], vocab_size)
    
    def decode_text(self, input_data):
        encoder_output = self.model(input_data, return_audio=False)
        pooled = encoder_output.mean(dim=1)
        logits = self.text_decoder(pooled)
        predicted_ids = logits.argmax(dim=-1)
        return self.processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

    def training_step(self, batch, batch_idx):
        input_data = (
            batch["audios"], batch["audio_attention_mask"],
            batch["videos"], batch["video_attention_mask"]
        )
        features, target_audio = self.model(input_data, return_audio=True)
        loss = self.loss_fn(features, target_audio)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_data = (
            batch["audios"], batch["audio_attention_mask"],
            batch["videos"], batch["video_attention_mask"]
        )
        features, target_audio = self.model(input_data, return_audio=True)
        loss = self.loss_fn(features, target_audio)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        cos_sim = F.cosine_similarity(features.mean(1), target_audio.mean(1)).mean()
        self.log("val_cosine_sim", cos_sim, on_step=False, on_epoch=True, prog_bar=True)
        predicted_text = self.decode_text(input_data)
        val_wer = wer(batch["targets"], predicted_text)
        self.log("val_wer", val_wer, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_data = (
            batch["audios"], batch["audio_attention_mask"],
            batch["videos"], batch["video_attention_mask"]
        )
        predicted_text = self.decode_text(input_data)
        test_wer = wer(batch["targets"], predicted_text)
        self.log("test_wer", test_wer, on_step=False, on_epoch=True, prog_bar=True)
        return {"test_wer": test_wer}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-6)
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(num_training_steps * 0.1)
        
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def on_save_checkpoint(self, checkpoint):
        self.processor.save_pretrained(self.tokenizer_name)
        checkpoint["tokenizer_dir"] = self.tokenizer_name

    def on_load_checkpoint(self, checkpoint):
        if "tokenizer_dir" in checkpoint:
            self.tokenizer_name = checkpoint["tokenizer_dir"]
            self.processor = WhisperProcessor.from_pretrained(self.tokenizer_name)


def main():
    config = get_config()
    data_module = DataModule(config)
    model = AVSRModule(config)
    
    callbacks = [
        ModelCheckpoint(dirpath=config["output"]["checkpoint_dir"], filename="avsr-{epoch:02d}-{val_loss:.2f}", save_top_k=config["output"]["save_top_k"], monitor=config["output"]["monitor"], mode=config["output"]["monitor_mode"]),
        EarlyStopping(monitor=config["output"]["monitor"], patience=config["training"]["early_stopping_patience"], mode=config["output"]["monitor_mode"]),
        LearningRateMonitor(logging_interval="step")
    ]
    
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir=config["output"]["log_dir"],
        name="avsr_logs"),
        precision="16-mixed",
        accelerator="auto",
        devices="auto",
        strategy=DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else "auto",
        enable_progress_bar=True,
        fast_dev_run=True,
        )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()
