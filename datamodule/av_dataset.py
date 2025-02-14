# av_dataset.py
import os
import numpy as np
import torch
import torchaudio
import torchvision
import logging
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperModel

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, tokenizer_name="SageLiao/whisper-small-zh-TW"):
        logger.info(f"Initializing DataProcessor with tokenizer: {tokenizer_name}")
        
        # Initialize Whisper components
        self.whisper_model = WhisperModel.from_pretrained(tokenizer_name)
        self.whisper_processor = WhisperProcessor.from_pretrained(tokenizer_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_name)
        self.tokenizer = self.whisper_processor.tokenizer
        
        # Freeze Whisper model
        for param in self.whisper_model.parameters():
            param.requires_grad = False
            
        # Constants
        self.SAMPLE_RATE = 16000
        self.N_FRAMES = 30
        self.FRAME_RATE = 30
        
        logger.info("DataProcessor initialized successfully")
    def cut_or_pad(self, data, size, dim=0):
        current_size = data.size(dim)
        if current_size < size:
            pad_amount = size - current_size
            if data.dim() == 2 and dim == 1:
                data = torch.nn.functional.pad(data, (0, pad_amount))
            else:
                pad_tuple = [0] * (2 * data.dim())
                index = 2 * (data.dim() - dim - 1)
                pad_tuple[index + 1] = pad_amount
                data = torch.nn.functional.pad(data, tuple(pad_tuple))
        elif current_size > size:
            if dim == 0:
                data = data[:size]
            elif dim == 1:
                data = data[:, :size]
            else:
                slices = [slice(None)] * data.dim()
                slices[dim] = slice(0, size)
                data = data[tuple(slices)]
        assert data.size(dim) == size
        return data
    def process_audio_whisper(self, waveform):
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
        waveform = waveform / torch.max(torch.abs(waveform))
        features = self.feature_extractor(waveform.numpy(), sampling_rate=self.SAMPLE_RATE, return_tensors="pt").input_features.squeeze(0)
        return features
    def load_video(self, path, max_frames=300):
        vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
        vid = vid.permute((0, 3, 1, 2))
        target_frames = int(vid.size(0) * self.FRAME_RATE / self.N_FRAMES)
        if target_frames != vid.size(0):
            indices = torch.linspace(0, vid.size(0)-1, target_frames).long()
            vid = vid[indices]
        if vid.size(0) > max_frames:
            vid = vid[:max_frames]
        return vid
    def load_audio(self, path):
        waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
        waveform = waveform.squeeze(0)
        if sample_rate != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, self.SAMPLE_RATE)
            waveform = resampler(waveform)
        mel = self.process_audio_whisper(waveform)
        return mel

class AVDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, modality, audio_transform, video_transform, rate_ratio=640, max_frames=300, tokenizer_name="SageLiao/whisper-small-zh-TW"):
        logger.info(f"Initializing AVDataset for split: {split}")
        self.processor = DataProcessor(tokenizer_name)
        self.root_dir = root_dir
        self.split = split
        self.modality = modality
        self.rate_ratio = rate_ratio
        self.max_frames = max_frames
        self.video_dir = os.path.join(root_dir, split, f"{split}_video_seg12s")
        self.text_dir = os.path.join(root_dir, split, f"{split}_text_seg12s")
        self.samples = self._build_dataset()
        self.audio_transform = audio_transform
        self.video_transform = video_transform
    def _build_dataset(self):
        samples = []
        for folder in os.listdir(self.text_dir):
            text_path = os.path.join(self.text_dir, folder)
            video_path = os.path.join(self.video_dir, folder)
            if os.path.isdir(text_path) and os.path.isdir(video_path):
                text_files = [f for f in os.listdir(text_path) if f.endswith('.txt')]
                video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
                for text_file in text_files:
                    base_name = text_file[:-4]
                    video_file = base_name + '.mp4'
                    if video_file in video_files:
                        with open(os.path.join(text_path, text_file), 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        samples.append({'video_path': os.path.join(video_path, video_file), 'text': text})
        return samples
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        text = sample['text']
        
        # Process text target
        encoded = self.processor.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=448,  # Whisper's max sequence length
            return_tensors="pt"
        )
        target_ids = encoded.input_ids.squeeze(0)
        target_length = torch.tensor([len(target_ids)])
        
        return_dict = {
            "video": None,
            "video_mask": None,
            "audio": None, 
            "audio_mask": None,
            "target_ids": target_ids,
            "target_text": text,
            "target_lengths": target_length
        }
        
        # Load and process video if needed
        if self.modality in ["video", "audiovisual"]:
            video = self.processor.load_video(video_path, self.max_frames)
            video = self.video_transform(video)
            video_mask = torch.ones(video.size(0), dtype=torch.bool)
            return_dict["video"] = video
            return_dict["video_mask"] = video_mask
            return_dict["video_lengths"] = torch.tensor([video.size(0)])
        
        # Load and process audio if needed
        if self.modality in ["audio", "audiovisual"]:
            audio = self.processor.load_audio(video_path)
            audio = self.processor.cut_or_pad(audio, size=3000, dim=1)
            audio = self.audio_transform(audio)
            audio_mask = torch.ones(audio.size(0), dtype=torch.bool)
            return_dict["audio"] = audio
            return_dict["audio_mask"] = audio_mask
            return_dict["audio_lengths"] = torch.tensor([audio.size(0)])
        
        return return_dict
    def __len__(self):
        return len(self.samples)
