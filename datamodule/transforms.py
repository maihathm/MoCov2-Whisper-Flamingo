# transforms.py
import os
import random
import logging
import sentencepiece
import torch
import torchaudio
import torchvision
from utils.logging_utils import log_tensor_info
import torchvision.transforms.functional as F

NOISE_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "babble_noise.wav")
SP_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spm", "unigram", "unigram3370.model")
DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spm", "unigram", "unigram3370_units.txt")
def to_mono(x):
    # If x has more than one channel, average across channels.
    if x.dim() == 2 and x.size(0) > 1:
        return x.mean(dim=0, keepdim=True)
    return x
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
    def __init__(self, noise_filename=NOISE_FILENAME, snr_target=None):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20, 999999]
        self.noise, sample_rate = torchaudio.load(noise_filename)
        assert sample_rate == 16000
    def forward(self, speech):
        speech = speech.t()
        start_idx = random.randint(0, self.noise.shape[1] - speech.shape[1])
        noise_segment = self.noise[:, start_idx : start_idx + speech.shape[1]]
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        noisy_speech = torchaudio.functional.add_noise(speech, noise_segment, snr_level)
        return noisy_speech.t()

class VideoTransform:
    def __init__(self, subset):
        self.subset = subset
        if subset == "train":
            self.transforms = [
                ("resize_64", FunctionalModule(lambda x: F.resize(x, (64,64)))),
                ("to_float", FunctionalModule(lambda x: x / 255.0)),
                ("normalize_0_1", FunctionalModule(lambda x: x / 255.0)),
                ("random_flip", torchvision.transforms.RandomHorizontalFlip(p=0.5)),
                ("color_jitter", torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)),
                ("random_gray", torchvision.transforms.RandomGrayscale(p=0.2)),
                ("time_mask", AdaptiveTimeMask(10, 25)),
                ("normalize_imagenet", torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            ]
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                ("resize_64", FunctionalModule(lambda x: F.resize(x, (64,64)))),
                FunctionalModule(lambda x: x / 255.0),
                FunctionalModule(lambda x: torch.nn.functional.interpolate(
                    x, size=(64,64), mode="bilinear", align_corners=False
                )),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
    def __call__(self, sample):
        if sample.dim() != 4:
            raise ValueError(f"Expected 4D tensor (T,C,H,W), got shape {sample.shape}")
        if sample.size(1) != 3:
            raise ValueError(f"Expected 3 channels, got {sample.size(1)} channels")
        x = sample
        if self.subset == "train":
            for name, transform in self.transforms:
                x = transform(x)
        else:
            x = x / 255.0
            x = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        return x.contiguous()
import torch.nn.functional as F

def pad_or_trim_mel(mel, target_length=3000):
    """
    Pads (or trims) the mel spectrogram along the time dimension to target_length.
    Assumes mel shape is [batch, n_mels, time].
    """
    time_dim = mel.size(-1)
    if time_dim < target_length:
        pad_amount = target_length - time_dim
        mel = F.pad(mel, (0, pad_amount))
    elif time_dim > target_length:
        mel = mel[..., :target_length]  # Trim if too long
    return mel


class AudioTransform:
    def __init__(self, subset, snr_target=None):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80,
        )
        if subset == "train":
            self.audio_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: to_mono(x)),
                FunctionalModule(lambda x: self.mel_transform(x)),
                FunctionalModule(lambda x: x.squeeze(0) if x.dim() == 3 and x.size(0)==1 else x),
                FunctionalModule(lambda x: self._apply_spec_augment(x)),
                FunctionalModule(lambda x: pad_or_trim_mel(x, target_length=3000)),
                FunctionalModule(lambda x: x.transpose(0, 1)),
                AddNoise(),
                FunctionalModule(lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)),
            )
        else:
            self.audio_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: to_mono(x)),
                FunctionalModule(lambda x: self.mel_transform(x)),
                FunctionalModule(lambda x: x.squeeze(0) if x.dim() == 3 and x.size(0)==1 else x),
                FunctionalModule(lambda x: pad_or_trim_mel(x, target_length=3000)),
                FunctionalModule(lambda x: x.transpose(0, 1)),
                AddNoise(snr_target=snr_target) if snr_target is not None else FunctionalModule(lambda x: x),
                FunctionalModule(lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)),
            )

    def _apply_spec_augment(self, mel_spectrogram):
        """
        Thực hiện SpecAugment trên mel_spectrogram (time x freq).
        """
        if len(mel_spectrogram.shape) != 2:
            raise ValueError(f"Expected 2D tensor (time x frequency), got shape {mel_spectrogram.shape}")

        # Số lần mask tần số
        freq_mask_param = 48
        num_freq_masks = 2

        # Thực hiện freq mask
        freq_dim = mel_spectrogram.size(1)
        for _ in range(num_freq_masks):
            # Nếu freq_dim <= freq_mask_param => bỏ qua mask
            max_freq_start = freq_dim - freq_mask_param
            if max_freq_start <= 0:
                # Không mask vì phạm vi âm/0
                continue
            freq_start = int(torch.randint(0, max_freq_start, (1,)))
            mel_spectrogram[:, freq_start:freq_start+freq_mask_param] = 0

        # Tương tự cho time
        time_mask_param = mel_spectrogram.size(0) // 8  # vd
        num_time_masks = 2
        time_dim = mel_spectrogram.size(0)

        for _ in range(num_time_masks):
            max_time_start = time_dim - time_mask_param
            if max_time_start <= 0:
                # bỏ qua mask
                continue
            time_start = int(torch.randint(0, max_time_start, (1,)))
            mel_spectrogram[time_start:time_start+time_mask_param, :] = 0

        return mel_spectrogram


    def __call__(self, sample):
        return self.audio_pipeline(sample)


class TextTransform:
    def __init__(self, sp_model_path=SP_MODEL_PATH, dict_path=DICT_PATH):
        self.spm = sentencepiece.SentencePieceProcessor(model_file=sp_model_path)
        units = open(dict_path, encoding='utf8').read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}
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
