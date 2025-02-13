# AVSR System with MOCO v2 and Whisper Frontend

Audio-Visual Speech Recognition (AVSR) system using MOCO v2 for video processing and Whisper for audio processing, combined with Flamingo's gate cross attention.

## Environment Setup

### 1. Conda Environment Setup

```bash
# Create new environment
conda create -n moco_whisper_flamingo python=3.9
conda activate moco_whisper_flamingo

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install FFmpeg for video processing
conda install -c conda-forge ffmpeg
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

## Frontend Training

### 1. MOCO v2 Training (Video Frontend)

Train the MOCO v2 model for video processing:

```bash
python trainFrontend/train_moco.py
```

Key features:
- Loads video data from AVDataset
- Uses contrastive learning with InfoNCE loss
- Implements cosine learning rate scheduling with warmup
- Saves checkpoints and logs training metrics

### 2. Whisper Training (Audio Frontend)

Train the Whisper model for audio processing:

```bash
python trainFrontend/train_whisper.py
```

Key features:
- Loads audio data from AVDataset
- Uses pretrained Whisper model with optional encoder freezing
- Implements feature reconstruction training
- Includes gradient clipping and learning rate scheduling

## Training Monitoring

Monitor training progress:

```bash
# Launch TensorBoard
tensorboard --logdir logs/
```

Training logs are saved in:
- `moco_training.log` for MOCO v2
- `whisper_training.log` for Whisper

## Configuration

The `config.py` file contains all configurable parameters:

- Model architecture settings
- Training hyperparameters
- Data augmentation options
- Logging configurations

## Training Tips

1. **Data Preparation**:
   - Ensure dataset is properly organized in the data directory
   - Run preprocessing if needed: `python preprocess/preprocessing.py`

2. **GPU Requirements**:
   - CUDA-capable GPU recommended
   - Adjust batch size based on available GPU memory
   - Monitor GPU usage during training

3. **Training Flow**:
   - Start with MOCO v2 pretraining
   - Then proceed with Whisper pretraining
   - Monitor loss curves in TensorBoard
   - Save best checkpoints for final model

4. **Troubleshooting**:
   - Check log files for any errors
   - Verify data loading is correct
   - Monitor GPU memory usage
   - Ensure all dependencies are properly installed

## Directory Structure

```
MoCov2-Whisper-Flamingo/
├── config.py                 # Configuration settings
├── models/                   # Model architectures
├── datamodule/              # Data loading and processing
├── trainFrontend/           # Frontend training scripts
│   ├── train_moco.py        # MOCO v2 training
│   └── train_whisper.py     # Whisper training
├── utils/                   # Utility functions
└── requirements.txt         # Project dependencies
