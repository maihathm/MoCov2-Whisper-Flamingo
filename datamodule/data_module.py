from operator import itemgetter
import numpy as np
import torch
import torchvision
from fairseq.data import data_utils
from torch.utils.data import Dataset, DistributedSampler, RandomSampler
from torch.utils.data.sampler import Sampler

class ByFrameCountSampler(Sampler):
    def __init__(self, dataset, max_frames_per_gpu, shuffle=True, seed=0, max_frames=300):
        self.dataset = dataset
        self.max_frames_per_gpu = max_frames_per_gpu
        self.max_frames = max_frames
        self.sizes = []
        for idx in range(len(dataset)):
            video_path = dataset.samples[idx]['video_path']
            video_info = torchvision.io.read_video_timestamps(video_path, pts_unit='sec')
            self.sizes.append(min(len(video_info[0]), self.max_frames))
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        batch_indices = data_utils.batch_by_size(self._get_indices(), lambda i: self.sizes[i], max_tokens=max_frames_per_gpu)
        self.num_batches = len(batch_indices)
    def _get_indices(self):
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
        batch_indices = data_utils.batch_by_size(self._get_indices(), lambda i: self.sizes[i], max_tokens=self.max_frames_per_gpu)
        return iter(batch_indices)
    def set_epoch(self, epoch):
        self.epoch = epoch

class DatasetFromSampler(Dataset):
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
    def __init__(self, sampler, num_replicas=None, rank=None, shuffle=True, drop_last=False):
        super(DistributedSamplerWrapper, self).__init__(DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last)
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
    def __init__(self, sampler):
        super(RandomSamplerWrapper, self).__init__(DatasetFromSampler(sampler))
        self.sampler = sampler
    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
