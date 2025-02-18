# data_module.py
import os
import logging
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from fairseq.data import data_utils
from torch.utils.data import Dataset, DistributedSampler, RandomSampler, DataLoader
from torch.utils.data.sampler import Sampler
from operator import itemgetter

from .av_dataset import AVDataset
from .transforms import AudioTransform, VideoTransform  # => File phụ (tuỳ bạn)

logger = logging.getLogger(__name__)

class ByFrameCountSampler(Sampler):
    """
    Một sampler xếp batch dựa trên số frame video,
    nhằm tránh batch quá dài/khó fitting.
    """
    def __init__(self, dataset, max_frames_per_gpu, shuffle=True, seed=0, max_frames=300):
        self.dataset = dataset
        self.max_frames_per_gpu = max_frames_per_gpu
        self.max_frames = max_frames
        self.sizes = []
        for idx in range(len(dataset)):
            video_path = dataset.samples[idx]['video_path']
            video_info = torchvision.io.read_video_timestamps(video_path, pts_unit='sec')
            # Lấy số frame video thực
            self.sizes.append(min(len(video_info[0]), self.max_frames))
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        # batch_indices = data_utils.batch_by_size(indices, num_tokens_fn, max_tokens=self.max_frames_per_gpu)
        batch_indices = data_utils.batch_by_size(self._get_indices(), lambda i: self.sizes[i], max_tokens=max_frames_per_gpu)
        self.num_batches = len(batch_indices)
        
    def _get_indices(self):
        """
        Sắp xếp index dataset theo rule lexsort, ngược => grouping
        """
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = [torch.randperm(len(self.dataset), generator=g).tolist()]
        else:
            order = [list(range(len(self.dataset)))]
        order.append(self.sizes)
        return np.lexsort(order)[::-1]
        
    def __len__(self):
        return self.num_batches
        
    def __iter__(self):
        batch_indices = data_utils.batch_by_size(
            self._get_indices(),
            lambda i: self.sizes[i],
            max_tokens=self.max_frames_per_gpu
        )
        return iter(batch_indices)
        
    def set_epoch(self, epoch):
        self.epoch = epoch

class DatasetFromSampler(Dataset):
    """
    Biến sampler thành dataset, để wrapper Sampler
    => Cho DistributedSampler, RandomSampler
    """
    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None
        
    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]
        
    def __len__(self) -> int:
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Đóng gói sampler => DistributedSampler
    """
    def __init__(self, sampler, num_replicas=None, rank=None, shuffle=True, drop_last=False):
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler), 
            num_replicas=num_replicas, 
            rank=rank, 
            shuffle=shuffle, 
            drop_last=drop_last
        )
        self.sampler = sampler
        
    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
        
    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.sampler.set_epoch(epoch)

class RandomSamplerWrapper(RandomSampler):
    """
    Đóng gói sampler => random
    """
    def __init__(self, sampler):
        super(RandomSamplerWrapper, self).__init__(DatasetFromSampler(sampler))
        self.sampler = sampler
        
    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


def collate_fn(batch):
    """
    Gộp list sample thành 1 batch. 
    Cần padding target_ids => cùng độ dài, ...
    """
    max_target_len = max(item['target_ids'].size(0) for item in batch)
    
    # Pad target_ids
    padded_targets = []
    for item in batch:
        curr_len = item['target_ids'].size(0)
        if curr_len < max_target_len:
            padding = torch.zeros(max_target_len - curr_len, dtype=item['target_ids'].dtype)
            # ignore_index => -100 ? Tùy code => ta giữ 0
            padded_target = torch.cat([item['target_ids'], padding])
        else:
            padded_target = item['target_ids']
        padded_targets.append(padded_target)
    
    return {
        'video': torch.stack([item['video'] for item in batch]),
        'video_mask': torch.stack([item['video_mask'] for item in batch]),
        'audio': torch.stack([item['audio'] for item in batch]) if batch[0]['audio'] is not None else None,
        'audio_mask': torch.stack([item['audio_mask'] for item in batch]) if batch[0]['audio_mask'] is not None else None,
        'target_ids': torch.stack(padded_targets),
        'target_text': [item['target_text'] for item in batch],
        'target_lengths': torch.stack([item['target_lengths'] for item in batch]),
        'audio_lengths': torch.stack([item['audio_lengths'] for item in batch]) if batch[0]['audio_lengths'] is not None else None,
        'video_lengths': torch.stack([item['video_lengths'] for item in batch])
    }

class DataModule(pl.LightningDataModule):
    """
    LightningDataModule: load AVDataset, sampler, dataloader
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.root_dir = config["data"]["root_dir"]
        self.batch_size = config["data"]["batch_size"]
        self.val_batch_size = config["data"]["val_batch_size"]
        self.test_batch_size = config["data"]["test_batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.max_frames = config["data"]["max_frames"]
        self.max_frames_val = config["data"]["max_frames_val"]
        self.rate_ratio = config["data"]["rate_ratio"]
        self.modality = config["data"]["modality"]
        
        # tokenizer_name
        self.tokenizer_name = config["whisper"]["model_name"]
        if config["data"].get("updated_tokenizer_dir") and os.path.exists(config["data"]["updated_tokenizer_dir"]):
            self.tokenizer_name = config["data"]["updated_tokenizer_dir"]
        
        logger.info(f"DataModule initialized with tokenizer: {self.tokenizer_name}")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            logger.info("Setting up training and validation datasets...")
            # Training dataset
            self.train_dataset = AVDataset(
                root_dir=self.root_dir,
                split='train',
                modality=self.modality,
                audio_transform=AudioTransform('train'),
                video_transform=VideoTransform('train'),
                rate_ratio=self.rate_ratio,
                max_frames=self.max_frames,
                tokenizer_name=self.tokenizer_name
            )
            logger.info(f"Train dataset created with {len(self.train_dataset)} samples")
            
            # Validation dataset
            self.val_dataset = AVDataset(
                root_dir=self.root_dir,
                split='val',
                modality=self.modality,
                audio_transform=AudioTransform('val'),
                video_transform=VideoTransform('val'),
                rate_ratio=self.rate_ratio,
                max_frames=self.max_frames_val,
                tokenizer_name=self.tokenizer_name
            )
            logger.info(f"Validation dataset created with {len(self.val_dataset)} samples")

        if stage == 'test' or stage is None:
            logger.info("Setting up test dataset...")
            self.test_dataset = AVDataset(
                root_dir=self.root_dir,
                split='test',
                modality=self.modality,
                audio_transform=AudioTransform('test'),
                video_transform=VideoTransform('test'),
                rate_ratio=self.rate_ratio,
                max_frames=self.max_frames_val,
                tokenizer_name=self.tokenizer_name
            )
            logger.info(f"Test dataset created with {len(self.test_dataset)} samples")

    def _get_sampler(self, dataset, batch_size, shuffle=True):
        """
        Trả về sampler => ByFrameCountSampler => DistributedSamplerWrapper / RandomSamplerWrapper
        """
        logger.info(f"Creating sampler with batch_size={batch_size}, shuffle={shuffle}")
        base_sampler = ByFrameCountSampler(
            dataset=dataset,
            max_frames_per_gpu=self.max_frames * batch_size,
            shuffle=shuffle,
            max_frames=self.max_frames
        )
        
        if self.trainer and self.trainer.num_devices > 1:
            logger.info(f"Using DistributedSamplerWrapper with {self.trainer.num_devices} devices")
            return DistributedSamplerWrapper(
                sampler=base_sampler,
                num_replicas=self.trainer.num_devices,
                rank=self.trainer.global_rank,
                shuffle=shuffle
            )
        return RandomSamplerWrapper(base_sampler) if shuffle else base_sampler

    def train_dataloader(self):
        logger.info("Creating training dataloader...")
        sampler = self._get_sampler(self.train_dataset, self.batch_size, shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        logger.info("Creating validation dataloader...")
        sampler = self._get_sampler(self.val_dataset, self.val_batch_size, shuffle=False)
        return DataLoader(
            self.val_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        logger.info("Creating test dataloader...")
        sampler = self._get_sampler(self.test_dataset, self.test_batch_size, shuffle=False)
        return DataLoader(
            self.test_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
