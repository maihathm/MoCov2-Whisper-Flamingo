"""
Configuration file for AVSR model using MOCO v2, Whisper, and Flamingo's gate cross attention
"""

import os

# Data paths
DATA_ROOT = "/root/maihathm/AVASR/data/avsr_self"
MOCO_PRETRAINED = "moco_v2_800ep_pretrain.pth.tar"

# Model configuration
MODEL_CONFIG = {
    # Frontend configuration
    "FRONTEND_DMODEL": 512,    # Frontend model dimension
    "WORD_NUM_CLASSES": 500,   # Number of word classes
    "FRAME_LENGTH": 96,        # Frame length for visual frontend
    "VIDEO_FEATURE_SIZE": 2048, # Size of video features
    
    # Model dimensions (aligned with Whisper base)
    "d_model": 512,          # Model dimension
    "n_heads": 8,           # Number of attention heads
    "n_layers": 6,          # Number of transformer layers
    "pe_max_len": 5000,     # Maximum length for positional encoding
    "fc_hidden_size": 2048, # Hidden size for feed-forward layers
    "dropout": 0.1,         # Dropout rate
    
    # Input processing
    "required_input_length": 96,  # Required input sequence length
    "rate_ratio": 640,           # Audio/video sampling rate ratio
    
    # Fusion settings
    "fusion_layers": 3,     # Number of gate cross attention layers
    "fusion_dropout": 0.1,  # Dropout in fusion layers
    
    # Modality dropout
    "prob_av": 0.5,        # Probability of using both modalities
    "prob_a": 0.25,        # Probability of using only audio
    
    # Training settings
    "batch_size": 32,
    "val_batch_size": 32,
    "test_batch_size": 1,
    "num_workers": 4,
    "max_frames": 300,      # Maximum number of frames per batch
    "max_frames_val": 400,  # Maximum number of frames per batch for validation
    
    # Inference settings
    "beam_width": 5,        # Beam width for beam search
    "lambda": 0.6,         # CTC/Attention interpolation weight
}

# Training configuration
TRAIN_CONFIG = {
    "epochs": 100,
    "warmup_ratio": 0.1,    # Percentage of total steps for warmup
    "max_lr": 1e-3,         # Maximum learning rate after warmup
    "min_lr": 1e-5,         # Minimum learning rate
    "weight_decay": 0.01,
    "gradient_clip_val": 1.0,
    "early_stopping_patience": 10,
    "accumulate_grad_batches": 2,  # Gradient accumulation steps
    "label_smoothing": 0.1,  # Label smoothing factor
}

# Data augmentation settings
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

# Model settings
WHISPER_CONFIG = {
    "model_name": "openai/whisper-base",
    "freeze_encoder": True,
    "use_flash_attention": True,  # Use flash attention if available
}

MOCO_CONFIG = {
    "freeze_encoder": True,  # Freeze MOCO v2 parameters
    "feature_dim": 2048,    # MOCO v2 feature dimension
}

# Logging and checkpointing
OUTPUT_CONFIG = {
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs",
    "save_top_k": 3,
    "monitor": "val_loss",
    "monitor_mode": "min",
    "log_every_n_steps": 100,
    "save_predictions": True,  # Save model predictions for analysis
    "tensorboard": {
        "log_graph": True,     # Log model graph
        "log_weights": True,   # Log weight histograms
        "log_gates": True      # Log gate attention weights
    }
}

def get_config():
    """Returns a dictionary containing all configuration settings"""
    config = {
        "data": {
            "root_dir": DATA_ROOT,
            "moco_file": MOCO_PRETRAINED,
            "batch_size": MODEL_CONFIG["batch_size"],
            "val_batch_size": MODEL_CONFIG["val_batch_size"],
            "test_batch_size": MODEL_CONFIG["test_batch_size"],
            "num_workers": MODEL_CONFIG["num_workers"],
            "max_frames": MODEL_CONFIG["max_frames"],
            "max_frames_val": MODEL_CONFIG["max_frames_val"],
            "rate_ratio": MODEL_CONFIG["rate_ratio"],
        },
        "model": {
            "d_model": MODEL_CONFIG["d_model"],
            "n_heads": MODEL_CONFIG["n_heads"],
            "n_layers": MODEL_CONFIG["n_layers"],
            "pe_max_len": MODEL_CONFIG["pe_max_len"],
            "fc_hidden_size": MODEL_CONFIG["fc_hidden_size"],
            "dropout": MODEL_CONFIG["dropout"],
            "fusion_layers": MODEL_CONFIG["fusion_layers"],
            "required_input_length": MODEL_CONFIG["required_input_length"],
        },
        "training": TRAIN_CONFIG,
        "augmentation": AUGMENTATION,
        "whisper": WHISPER_CONFIG,
        "output": OUTPUT_CONFIG,
    }
    
    # Create output directories if they don't exist
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output"]["log_dir"], exist_ok=True)
    
    return config
