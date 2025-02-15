# train.py
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from jiwer import wer
from transformers import WhisperProcessor
import logging

from config import get_config
from datamodule.data_module import DataModule
from models.av_net import AVNet
torch.use_deterministic_algorithms(False)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ""
logger = logging.getLogger(__name__)

# Optional: fix GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def is_deepcopyable(obj):
    try:
        copy.deepcopy(obj)
        return True
    except Exception:
        return False

class AVSRModule(pl.LightningModule):
    """
    Audio-Visual Speech Recognition Module (Lightning).
    Includes:
    - forward pass on AVNet
    - CTC + CE losses
    - decoding + WER for validation/test
    """
    def __init__(self, config, processor, vocab_size=None):
        super().__init__()
        self.config = config
        self.processor = processor

        # Nếu chưa truyền vocab_size -> Lấy len(processor.tokenizer)
        if vocab_size is None:
            self.vocab_size = len(self.processor.tokenizer)
        else:
            self.vocab_size = vocab_size

        # Save hyperparameters for logging
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

        # Model configuration
        model_args = (
            config["model"]["d_model"],
            config["model"]["n_heads"],
            config["model"]["n_layers"],
            config["model"]["pe_max_len"],
            config["model"]["fc_hidden_size"],
            config["model"]["dropout"]
        )

        # Initialize AVNet
        self.model = AVNet(
            modal=config["data"]["modality"],
            MoCofile=os.path.join(os.getcwd(), config["data"]["moco_file"]),
            reqInpLen=config["model"]["required_input_length"],
            modelargs=model_args,
            vocab_size=self.vocab_size,  # QUAN TRỌNG
            enable_logging=config["output"]["enable_logging"]
        )

        # 2 losses: CTC, CE
        self.ctc_loss = nn.CTCLoss(
            blank=0, 
            reduction='mean',
            zero_infinity=True
        )
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=config["training"]["label_smoothing"]
        )

    def _compute_ctc_loss(self, logits, targets, input_lengths, target_lengths):
        """
        logits shape: [B, T, vocab_size]
        => ctc expects [T, B, vocab_size]
        """
        log_probs = F.log_softmax(logits, dim=-1).transpose(0,1)  # => [T, B, vocab_size]
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

    def _compute_ce_loss(self, logits, targets):
        """
        logits: [B, T_pred, vocab_size]
        targets: [B, T_gt]
        -> cắt/pad sao cho T_pred == T_gt (hoặc min)
        """
        B, T_pred, _ = logits.shape
        T_gt = targets.shape[1]

        T_min = min(T_pred, T_gt)
        trimmed_logits = logits[:, :T_min, :]    # [B, T_min, vocab_size]
        trimmed_targets = targets[:, :T_min]     # [B, T_min]

        # Debug check
        t_min_id = trimmed_targets.min().item()
        t_max_id = trimmed_targets.max().item()
        if t_min_id < -100 or t_max_id >= self.vocab_size:
            print(f"[DEBUG] Out-of-range token. min_id={t_min_id}, max_id={t_max_id}, vocab_size={self.vocab_size}")
            # raise ValueError("Found out-of-range token ID in targets")

        # Flatten
        trimmed_logits = trimmed_logits.reshape(-1, self.vocab_size)  # [B*T_min, vocab_size]
        trimmed_targets = trimmed_targets.reshape(-1)                 # [B*T_min]

        ce_loss = self.cross_entropy(trimmed_logits, trimmed_targets)
        return ce_loss

    def _decode_predictions(self, logits):
        """
        Decode using self.processor
        logits: [B, T, vocab_size]
        => predictions: [B, T]
        => text
        """
        predictions = torch.argmax(logits, dim=-1)
        return self.processor.tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True
        )

    def training_step(self, batch, batch_idx):
        """
        - forward
        - ctc loss + ce loss
        - log
        """
        audio, audio_mask = batch["audio"], batch["audio_mask"]
        video, video_mask = batch["video"], batch["video_mask"]
        video_len = batch["video_lengths"]

        input_data = (audio, audio_mask, video, video_mask, video_len)
        logits = self.model(input_data)

        # ctc
        ctc_loss = self._compute_ctc_loss(
            logits,
            batch["target_ids"],
            batch["audio_lengths"],
            batch["target_lengths"]
        )
        # ce
        ce_loss = self._compute_ce_loss(logits, batch["target_ids"])
        loss = ctc_loss + ce_loss

        self.log("train/ctc_loss", ctc_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        - forward
        - ctc + ce
        - decode WER
        """
        audio, audio_mask = batch["audio"], batch["audio_mask"]
        video, video_mask = batch["video"], batch["video_mask"]
        video_len = batch["video_lengths"]

        input_data = (audio, audio_mask, video, video_mask, video_len)
        logits = self.model(input_data)

        ctc_loss = self._compute_ctc_loss(
            logits,
            batch["target_ids"],
            batch["audio_lengths"],
            batch["target_lengths"]
        )
        ce_loss = self._compute_ce_loss(logits, batch["target_ids"])
        loss = ctc_loss + ce_loss

        preds = self._decode_predictions(logits)
        val_wer = wer(batch["target_text"], preds)

        self.log("val/ctc_loss", ctc_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/wer", val_wer, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_wer": val_wer}

    def test_step(self, batch, batch_idx):
        audio, audio_mask = batch["audio"], batch["audio_mask"]
        video, video_mask = batch["video"], batch["video_mask"]
        video_len = batch["video_lengths"]

        input_data = (audio, audio_mask, video, video_mask, video_len)
        logits = self.model(input_data)

        preds = self._decode_predictions(logits)
        test_wer = wer(batch["target_text"], preds)
        self.log("test/wer", test_wer, on_step=False, on_epoch=True)

        return {"test_wer": test_wer}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["training"]["max_lr"],
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=self.config["training"]["weight_decay"]
        )

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


def main():
    """
    Main function for training/testing the AVSR model
    """
    # 1. Load config
    config = get_config()

    # 2. Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 3. (Optionally) init a single WhisperProcessor & add tokens if needed
    logger.info("Initializing data module and model...")

    # Tạo processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    # Ví dụ thêm token
    # new_tokens = ["<my_special_1>", "<my_special_2>"]
    # processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    extended_vocab_size = len(processor.tokenizer)
    logger.info(f"Extended vocab size = {extended_vocab_size}")

    # 4. Initialize DataModule => pass tokenizer_name or pass processor
    data_module = DataModule(config)  # code assume "tokenizer_name" = openai/whisper-small inside

    # 5. Create model => pass same processor + extended_vocab_size
    model = AVSRModule(
        config=config,
        processor=processor,
        vocab_size=extended_vocab_size
    )

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

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=config["output"]["log_dir"],
        name="avsr_logs",
        default_hp_metric=False
    )

    # Trainer
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
        deterministic=False,
        # fast_dev_run=True,  # đặt True để debug nhanh, False để train full
    )

    # 6. Fit
    logger.info("Starting training...")
    trainer.fit(model, datamodule=data_module)

    # 7. Test
    logger.info("Starting testing...")
    trainer.test(model, datamodule=data_module)

    logger.info("Training and testing completed!")


if __name__ == "__main__":
    main()
