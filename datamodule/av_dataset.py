import os
import numpy as np
import torch
import torchaudio
import torchvision
import whisper
from transformers import WhisperProcessor


class DataProcessor:
    def __init__(self):
        # Initialize Whisper processor for audio
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.whisper_model = whisper.load_model("base")
        
        # Constants for processing
        self.SAMPLE_RATE = 16000
        self.N_FRAMES = 30  # Number of frames to extract per second
        self.FRAME_RATE = 30  # Video frame rate
        
    def cut_or_pad(self, data, size, dim=0):
        """
        Pads or trims the data along a dimension.
        """
        if data.size(dim) < size:
            padding = size - data.size(dim)
            data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
            size = data.size(dim)
        elif data.size(dim) > size:
            data = data[:size]
        assert data.size(dim) == size
        return data
    
    def process_audio_whisper(self, waveform):
        """
        Process audio using Whisper's preprocessing pipeline
        """
        # Convert to float32 if needed
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
            
        # Normalize audio
        waveform = waveform / torch.max(torch.abs(waveform))
        
        # Convert to mel spectrogram using Whisper's parameters
        mel = self.whisper_model.mel_filters(waveform)
        
        return mel


    def load_video(self, path):
        """
        Load and preprocess video for MOCO v2
        rtype: torch, T x C x H x W
        """
        # Load video
        vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
        
        # Convert to correct format for MOCO v2 (T x C x H x W)
        vid = vid.permute((0, 3, 1, 2))
        
        # Ensure consistent frame rate
        target_frames = int(vid.size(0) * self.FRAME_RATE / self.N_FRAMES)
        if target_frames != vid.size(0):
            indices = torch.linspace(0, vid.size(0)-1, target_frames).long()
            vid = vid[indices]
            
        return vid

    def load_audio(self, path):
        """
        Load and preprocess audio for Whisper
        rtype: torch, T x n_mels
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
        waveform = waveform.squeeze(0)  # Remove channel dimension
        
        # Resample if needed
        if sample_rate != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, self.SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # Process using Whisper's preprocessing
        mel = self.process_audio_whisper(waveform)
        
        return mel


class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        split,  # 'train', 'test', or 'valid'
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
    ):
        # Initialize data processor
        self.processor = DataProcessor()
        self.root_dir = root_dir
        self.split = split
        self.modality = modality
        self.rate_ratio = rate_ratio

        # Xây dựng đường dẫn cho video và text
        self.video_dir = os.path.join(root_dir, split, f"{split}_video_seg12s")
        self.text_dir = os.path.join(root_dir, split, f"{split}_text_seg12s")
        
        # Load danh sách files
        self.samples = self._build_dataset()

        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def _build_dataset(self):
        samples = []
        # Duyệt qua các thư mục con trong text_dir
        for folder in os.listdir(self.text_dir):
            text_path = os.path.join(self.text_dir, folder)
            video_path = os.path.join(self.video_dir, folder)
            
            if os.path.isdir(text_path) and os.path.isdir(video_path):
                # Tìm file text và video tương ứng
                text_files = [f for f in os.listdir(text_path) if f.endswith('.txt')]
                video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
                
                for text_file in text_files:
                    base_name = text_file[:-4]  # bỏ đuôi .txt
                    video_file = base_name + '.mp4'
                    
                    if video_file in video_files:
                        # Đọc và xử lý text
                        with open(os.path.join(text_path, text_file), 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        samples.append({
                            'video_path': os.path.join(video_path, video_file),
                            'text': text
                        })
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        text = sample['text']
        
        if self.modality == "video":
            # Load and process video for MOCO v2
            video = self.processor.load_video(video_path)
            video = self.video_transform(video)
            return {"input": video, "target": text}
            
        elif self.modality == "audio":
            # Load and process audio for Whisper
            audio = self.processor.load_audio(video_path)
            audio = self.audio_transform(audio)
            return {"input": audio, "target": text}
            
        elif self.modality == "audiovisual":
            # Load and process both modalities
            video = self.processor.load_video(video_path)
            audio = self.processor.load_audio(video_path)
            
            # Ensure temporal alignment
            audio = self.processor.cut_or_pad(audio, len(video) * self.rate_ratio)
            
            # Apply transforms
            video = self.video_transform(video)
            audio = self.audio_transform(audio)
            
            # Return both modalities for gate cross attention
            return {
                "video": video,  # For MOCO v2
                "audio": audio,  # For Whisper
                "target": text,
                "video_mask": torch.ones(video.size(0)),  # For gate cross attention
                "audio_mask": torch.ones(audio.size(0))   # For gate cross attention
            }

    def __len__(self):
        return len(self.samples)
