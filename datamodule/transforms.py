# transforms.py

#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import logging
import sentencepiece
import torch
import torchaudio
import torchvision
from utils.logging_utils import log_tensor_info

logger = logging.getLogger(__name__)


NOISE_FILENAME = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "babble_noise.wav"
)

SP_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spm",
    "unigram",
    "unigram3370.model",
)

DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spm",
    "unigram",
    "unigram3370_units.txt",
)


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned


class AddNoise(torch.nn.Module):
    def __init__(
        self,
        noise_filename=NOISE_FILENAME,
        snr_target=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20, 999999]
        self.noise, sample_rate = torchaudio.load(noise_filename)
        assert sample_rate == 16000

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        speech = speech.t()
        start_idx = random.randint(0, self.noise.shape[1] - speech.shape[1])
        noise_segment = self.noise[:, start_idx : start_idx + speech.shape[1]]
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        noisy_speech = torchaudio.functional.add_noise(speech, noise_segment, snr_level)
        return noisy_speech.t()


class VideoTransform:
    """Video transforms for MOCO v2"""
    def __init__(self, subset):
        self.subset = subset
        # MOCO v2 specific transforms
        if subset == "train":
            self.transforms = [
                ("normalize_0_1", FunctionalModule(lambda x: x / 255.0)),  # Normalize to [0,1]
                ("random_flip", torchvision.transforms.RandomHorizontalFlip(p=0.5)),  # Add horizontal flip with 50% probability
                ("color_jitter", torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)),  # MOCO v2 color augmentation
                ("random_gray", torchvision.transforms.RandomGrayscale(p=0.2)),
                ("time_mask", AdaptiveTimeMask(10, 25)),  # Temporal masking
                ("normalize_imagenet", torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet normalization used by MOCO v2
                    std=[0.229, 0.224, 0.225]
                ))
            ]
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            )

    def __call__(self, sample):
        """
        Args:
            sample: T x C x H x W tensor
        Returns:
            T x C x H x W tensor with normalized values
        """
        
        # Ensure input is in correct format
        if sample.dim() != 4:
            raise ValueError(f"Expected 4D tensor (T,C,H,W), got shape {sample.shape}")
        if sample.size(1) != 3:
            raise ValueError(f"Expected 3 channels, got {sample.size(1)} channels")
        
        x = sample
        if self.subset == "train":
            for name, transform in self.transforms:
                x = transform(x)
        else:
            # For val/test, just apply normalization
            x = x / 255.0
            x = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(x)
            
        return x.contiguous()


class AudioTransform:
    """Audio transforms compatible with Whisper"""
    def __init__(self, subset, snr_target=None):
        # Whisper-specific audio transforms
        if subset == "train":
            self.audio_pipeline = torch.nn.Sequential(
                # Whisper uses mel spectrograms, so we focus on frequency masking
                FunctionalModule(lambda x: self._apply_spec_augment(x)),
                AddNoise(),  # Add noise for robustness
                FunctionalModule(
                    lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
                ),
            )
        elif subset == "val" or subset == "test":
            self.audio_pipeline = torch.nn.Sequential(
                AddNoise(snr_target=snr_target)
                if snr_target is not None
                else FunctionalModule(lambda x: x),
                FunctionalModule(
                    lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
                ),
            )
    
    def _apply_spec_augment(self, mel_spectrogram):
        """Apply SpecAugment to mel spectrogram"""
        # Time warping is not used as per Whisper's implementation
        
        # Ensure mel_spectrogram is 2D (time x frequency)
        if len(mel_spectrogram.shape) != 2:
            raise ValueError(f"Expected 2D tensor (time x frequency), got shape {mel_spectrogram.shape}")
        
        # Frequency masking
        freq_mask_param = 48  # Number of mel frequency channels to mask
        num_freq_masks = 2
        
        for _ in range(num_freq_masks):
            freq_start = int(torch.randint(0, mel_spectrogram.size(1) - freq_mask_param, (1,)))
            mel_spectrogram[:, freq_start:freq_start + freq_mask_param] = 0
            
        # Time masking
        time_mask_param = mel_spectrogram.size(0) // 8  # Max time steps to mask
        num_time_masks = 2
        
        for _ in range(num_time_masks):
            time_start = int(torch.randint(0, mel_spectrogram.size(0) - time_mask_param, (1,)))
            mel_spectrogram[time_start:time_start + time_mask_param, :] = 0
            
        return mel_spectrogram

    def __call__(self, sample):
        # sample: T x 1
        # rtype: T x 1
        return self.audio_pipeline(sample)


class TextTransform:
    """Mapping Dictionary Class for SentencePiece tokenization."""

    def __init__(
        self,
        sp_model_path=SP_MODEL_PATH,
        dict_path=DICT_PATH,
    ):

        # Load SentencePiece model
        self.spm = sentencepiece.SentencePieceProcessor(model_file=sp_model_path)

        # Load units and create dictionary
        units = open(dict_path, encoding='utf8').read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}
        # 0 will be used for "blank" in CTC
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def tokenize(self, text):
        tokens = self.spm.EncodeAsPieces(text)
        token_ids = [self.hashmap.get(token, self.hashmap["<unk>"]) for token in tokens]
        return torch.tensor(list(map(int, token_ids)))

    def post_process(self, token_ids):
        token_ids = token_ids[token_ids != -1]
        text = self._ids_to_str(token_ids, self.token_list)
        text = text.replace("\u2581", " ").strip()
        return text

    def _ids_to_str(self, token_ids, char_list):
        token_as_list = [char_list[idx] for idx in token_ids]
        return "".join(token_as_list).replace("<space>", " ")
