# data_module.py
import os
import logging
import torch
from pytorch_lightning import LightningDataModule
from utils.logging_utils import log_tensor_info

logger = logging.getLogger(__name__)

from .av_dataset import AVDataset
from .samplers import ByFrameCountSampler, DistributedSamplerWrapper, RandomSamplerWrapper
from .transforms import AudioTransform, VideoTransform


def pad(samples, pad_val=0.0):
    """Pad a batch of samples to the longest sequence."""
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = max_size - len(sample)
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    # Với tensor 2D hoặc 4D, giữ nguyên (có thể xử lý sau nếu cần)
    return collated_batch, lengths


def collate_pad(batch):
    """
    Collate function that handles single modality and audiovisual data,
    including creation of attention masks for gate cross attention.
    """
    batch_out = {}
    
    for data_type in batch[0].keys():
        if all(s[data_type] is None for s in batch):
            continue

        if data_type == "target":
            batch_out["targets"] = [s[data_type] for s in batch]
            continue

        pad_val = 0 if data_type in ["video_mask", "audio_mask"] else 0.0

        valid_samples = [s[data_type] for s in batch if s[data_type] is not None]
        if valid_samples:
            log_tensor_info(f"First sample of {data_type}", valid_samples[0])
        
        c_batch, sample_lengths = pad(valid_samples, pad_val)
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
        
        # Tạo attention mask
        if data_type in ["video", "audio"]:
            attention_mask = torch.zeros(len(batch), c_batch.size(1), dtype=torch.bool)
            for i, length in enumerate(sample_lengths):
                attention_mask[i, :length] = True
            batch_out[f"{data_type}_attention_mask"] = attention_mask

    return batch_out


class DataModule(LightningDataModule):
    """
    DataModule cho AVSR sử dụng MOCO v2, Whisper và gate cross attention.
    - Training: Mỗi sample là 1 cặp video–audio, sử dụng DistributedSampler (nếu đa GPU) hoặc RandomSampler.
    - Validation: Sử dụng ByFrameCountSampler để giới hạn tổng số frame của mỗi batch.
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.cfg.gpus = torch.cuda.device_count()
        self.total_gpus = self.cfg.gpus * self.cfg.trainer.num_nodes
        
        # Thiết lập batch size nếu chưa có
        self.cfg.data.batch_size = getattr(self.cfg.data, "batch_size", 32)
        self.cfg.data.val_batch_size = getattr(self.cfg.data, "val_batch_size", 32)
        
        # Thiết lập số frame tối đa
        self.max_frames = self.cfg.data.get("max_frames", 300)
        self.max_frames_val = self.cfg.data.get("max_frames_val", self.max_frames)

    def _dataloader(self, ds, batch_size, sampler, collate_fn, num_workers=4, pin_memory=True):
        """Tạo DataLoader với dataset, batch_size, sampler và collate_fn cho trước."""
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        """Tạo training dataloader với mỗi sample là 1 cặp video–audio."""
        ds_args = self.cfg.data.dataset
        train_ds = AVDataset(
            root_dir=ds_args.root_dir,
            split="train",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform("train"),
            video_transform=VideoTransform("train"),
            rate_ratio=self.cfg.data.get("rate_ratio", 640),
            max_frames=self.max_frames
        )
        
        # Sử dụng DistributedSampler nếu chạy trên nhiều GPU, ngược lại RandomSampler
        if self.total_gpus > 1 and torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(train_ds)
        else:
            sampler = torch.utils.data.RandomSampler(train_ds)
        
        return self._dataloader(
            ds=train_ds,
            batch_size=self.cfg.data.batch_size,
            sampler=sampler,
            collate_fn=collate_pad,
            num_workers=self.cfg.data.get("num_workers", 64),
            pin_memory=True
        )

    def val_dataloader(self):
        """Tạo validation dataloader. Ở đây dùng ByFrameCountSampler để gom batch theo tổng số frame."""
        ds_args = self.cfg.data.dataset
        val_ds = AVDataset(
            root_dir=ds_args.root_dir,
            split="val",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform("val"),
            video_transform=VideoTransform("val"),
            rate_ratio=self.cfg.data.get("rate_ratio", 640),
            max_frames=self.max_frames_val
        )
        
        sampler = ByFrameCountSampler(
            val_ds,
            max_frames_per_gpu=self.max_frames_val,
            shuffle=False,
            max_frames=self.max_frames_val
        )
        
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(
                sampler,
                shuffle=False,
                drop_last=True
            )
            
        # Khi dùng batch_sampler, DataLoader không nhận batch_size
        return torch.utils.data.DataLoader(
            val_ds,
            batch_sampler=sampler,
            num_workers=self.cfg.data.get("num_workers", 0),
            pin_memory=True,
            collate_fn=collate_pad,
        )

    def test_dataloader(self):
        """Tạo test dataloader (không sử dụng sampler đặc biệt)."""
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
            rate_ratio=self.cfg.data.get("rate_ratio", 640),
            max_frames=self.max_frames_val
        )
        
        return torch.utils.data.DataLoader(
            test_ds,
            batch_size=self.cfg.data.get("test_batch_size", 1),
            shuffle=False,
            num_workers=self.cfg.data.get("num_workers", 0),
            collate_fn=collate_pad,
            pin_memory=True
        )
