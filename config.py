import os
import torch

DATA_ROOT = "/root/maihathm/AVASR/data/avsr_self"
MOCO_PRETRAINED = "moco_v2_800ep_pretrain.pth.tar"

MODEL_CONFIG = {
    "FRONTEND_D_MODEL": 512,
    "WORD_NUM_CLASSES": 500,
    "FRAME_LENGTH": 96,
    "VIDEO_FEATURE_SIZE": 512,
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 6,
    "pe_max_len": 3000,
    "fc_hidden_size": 2048,
    "dropout": 0.1,
    "required_input_length": 96,
    "rate_ratio": 640,
    "fusion_layers": 6,
    "fusion_dropout": 0.1,
    "prob_av": 0.5,
    "prob_a": 0.25,
    "batch_size": 4,
    "val_batch_size": 2,
    "test_batch_size": 2,
    "num_workers": 0,
    "max_frames": 400,
    "max_frames_val": 400,
    "beam_width": 3,
    "lambda": 0.6,
}

TRAIN_CONFIG = {
    "epochs": 30,
    "warmup_ratio": 0.1,
    "max_lr": 1e-3,
    "min_lr": 1e-5,
    "weight_decay": 0.01,
    "gradient_clip_val": 1.0,
    "early_stopping_patience": 10,
    "accumulate_grad_batches": 4,
    "label_smoothing": 0.1,
}

AUGMENTATION = {
    "video": {
        "train": {
            "random_crop": 224,
            "color_jitter": 0.4,
            "grayscale_prob": 0.2,
            "time_mask_window": 10,
            "time_mask_stride": 25
        },
        "val": {
            "center_crop": 224
        }
    },
    "audio": {
        "train": {
            "freq_mask_param": 48,
            "time_mask_param": "length//8",
            "n_freq_masks": 2,
            "n_time_masks": 2
        }
    }
}

WHISPER_CONFIG = {
    "model_name": "openai/whisper-small",
    "freeze_encoder": True,
    "use_flash_attention": True,
    "language": "vietnamese",
    "task": "transcribe"
}

MOCO_CONFIG = {
    "freeze_encoder": True,
    "feature_dim": 512,
}

OUTPUT_CONFIG = {
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs",
    "save_top_k": 3,
    "monitor": "val_loss",
    "monitor_mode": "min",
    "log_every_n_steps": 100,
    "save_predictions": True,
    "tensorboard": {
        "log_graph": True,
        "log_weights": True,
        "log_gates": True
    },
    "enable_logging": True
}

class DotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)
    def __setattr__(self, key, value):
        self[key] = value

def get_config():
    config = DotDict({
        "data": DotDict({
            "root_dir": DATA_ROOT,
            "moco_file": MOCO_PRETRAINED,
            "batch_size": MODEL_CONFIG["batch_size"],
            "val_batch_size": MODEL_CONFIG["val_batch_size"],
            "test_batch_size": MODEL_CONFIG["test_batch_size"],
            "num_workers": MODEL_CONFIG["num_workers"],
            "max_frames": MODEL_CONFIG["max_frames"],
            "max_frames_val": MODEL_CONFIG["max_frames_val"],
            "rate_ratio": MODEL_CONFIG["rate_ratio"],
            "dataset": DotDict({
                "root_dir": DATA_ROOT,
            }),
            "modality": "audiovisual",
            # "updated_tokenizer_dir": "TW_tokenizer"
        }),
        "model": DotDict({
            "d_model": MODEL_CONFIG["d_model"],
            "n_heads": MODEL_CONFIG["n_heads"],
            "n_layers": MODEL_CONFIG["n_layers"],
            "pe_max_len": MODEL_CONFIG["pe_max_len"],
            "fc_hidden_size": MODEL_CONFIG["fc_hidden_size"],
            "dropout": MODEL_CONFIG["dropout"],
            "fusion_layers": MODEL_CONFIG["fusion_layers"],
            "required_input_length": MODEL_CONFIG["required_input_length"],
        }),
        "training": DotDict(TRAIN_CONFIG),
        "augmentation": DotDict(AUGMENTATION),
        "whisper": DotDict(WHISPER_CONFIG),
        "output": DotDict(OUTPUT_CONFIG),
        "trainer": DotDict({
            "num_nodes": 1,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": torch.cuda.device_count() if torch.cuda.is_available() else 1,
        }),
    })

    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output"]["log_dir"], exist_ok=True)

    return config
