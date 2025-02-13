import os

import torch
from pytorch_lightning import LightningDataModule

from .av_dataset import AVDataset
from .samplers import (
    ByFrameCountSampler,
    DistributedSamplerWrapper,
    RandomSamplerWrapper,
)
from .transforms import AudioTransform, VideoTransform


def pad(samples, pad_val=0.0):
    """Pad a batch of samples to the longest sequence."""
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    """
    Collate function that handles both single modality and audiovisual data
    with support for gate cross attention
    """
    batch_out = {}
    
    for data_type in batch[0].keys():
        # Skip if None
        if all(s[data_type] is None for s in batch):
            continue
            
        # Handle different padding values based on data type
        if data_type == "target":
            pad_val = -1
        elif data_type in ["video_mask", "audio_mask"]:
            pad_val = 0  # Mask padding
        else:
            pad_val = 0.0
            
        # Get data and lengths
        valid_samples = [s[data_type] for s in batch if s[data_type] is not None]
        c_batch, sample_lengths = pad(valid_samples, pad_val)
        
        # Store in output dictionary
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
        
        # Add attention mask for gate cross attention if needed
        if data_type in ["video", "audio"]:
            attention_mask = torch.zeros(len(batch), c_batch.size(1), dtype=torch.bool)
            for i, length in enumerate(sample_lengths):
                attention_mask[i, :length] = True
            batch_out[f"{data_type}_attention_mask"] = attention_mask
            
    return batch_out


class DataModule(LightningDataModule):
    """
    DataModule for handling AVSR data with MOCO v2, Whisper and gate cross attention
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.cfg.gpus = torch.cuda.device_count()
        self.total_gpus = self.cfg.gpus * self.cfg.trainer.num_nodes
        
        # Set default batch sizes if not specified
        self.cfg.data.batch_size = getattr(self.cfg.data, "batch_size", 32)
        self.cfg.data.val_batch_size = getattr(self.cfg.data, "val_batch_size", 32)

    def _dataloader(self, ds, sampler, collate_fn, num_workers=4, pin_memory=True):
        """Create a dataloader with given dataset and sampler"""
        return torch.utils.data.DataLoader(
            ds,
            num_workers=num_workers,
            pin_memory=pin_memory,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        """Create training dataloader with proper transforms and sampling"""
        ds_args = self.cfg.data.dataset
        train_ds = AVDataset(
            root_dir=ds_args.root_dir,
            split="train",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform("train"),
            video_transform=VideoTransform("train"),
            rate_ratio=self.cfg.data.get("rate_ratio", 640)
        )
        
        # Use frame-based sampling for videos
        sampler = ByFrameCountSampler(train_ds, self.cfg.data.max_frames)
        
        # Handle distributed training
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler)
        else:
            sampler = RandomSamplerWrapper(sampler)
            
        return self._dataloader(
            train_ds,
            sampler,
            collate_pad,
            num_workers=self.cfg.data.get("num_workers", 4),
            pin_memory=True
        )

    def val_dataloader(self):
        """Create validation dataloader"""
        ds_args = self.cfg.data.dataset
        val_ds = AVDataset(
            root_dir=ds_args.root_dir,
            split="valid",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform("val"),
            video_transform=VideoTransform("val"),
            rate_ratio=self.cfg.data.get("rate_ratio", 640)
        )
        
        # Use frame-based sampling without shuffling for validation
        sampler = ByFrameCountSampler(
            val_ds,
            self.cfg.data.max_frames_val,
            shuffle=False
        )
        
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(
                sampler,
                shuffle=False,
                drop_last=True
            )
            
        return self._dataloader(
            val_ds,
            sampler,
            collate_pad,
            num_workers=self.cfg.data.get("num_workers", 4),
            pin_memory=True
        )

    def test_dataloader(self):
        """Create test dataloader with noise target if specified"""
        ds_args = self.cfg.data.dataset
        test_ds = AVDataset(
            root_dir=ds_args.root_dir,
            split="test",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform(
                "test",
                snr_target=self.cfg.decode.get("snr_target", None)
            ),
            video_transform=VideoTransform("test"),
            rate_ratio=self.cfg.data.get("rate_ratio", 640)
        )
        
        # For testing, we use a simple dataloader without sampling
        return torch.utils.data.DataLoader(
            test_ds,
            batch_size=self.cfg.data.get("test_batch_size", 1),
            shuffle=False,
            num_workers=self.cfg.data.get("num_workers", 4),
            collate_fn=collate_pad,
            pin_memory=True
        )
