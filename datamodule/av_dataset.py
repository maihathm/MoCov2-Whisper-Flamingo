import os
import numpy as np
import torch
import torchaudio
import torchvision
import whisper
import logging
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperModel
from utils.logging_utils import log_tensor_info

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        # Initialize Whisper model, processor and feature extractor for audio
        self.whisper_model = WhisperModel.from_pretrained("SageLiao/whisper-small-zh-TW")
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("SageLiao/whisper-small-zh-TW")
        
        # Freeze Whisper parameters
        for param in self.whisper_model.parameters():
            param.requires_grad = False
            
        # Constants for processing
        self.SAMPLE_RATE = 16000
        self.N_FRAMES = 30  # Number of frames to extract per second
        self.FRAME_RATE = 30  # Video frame rate
        
    def cut_or_pad(self, data, size, dim=0):
        """
        Pads or trims the data along a specified dimension.
        """
        current_size = data.size(dim)
        if current_size < size:
            pad_amount = size - current_size
            # Nếu data là 2D và padding theo dim=1 (thời gian), chỉ pad chiều cuối.
            if data.dim() == 2 and dim == 1:
                data = torch.nn.functional.pad(data, (0, pad_amount))
            else:
                # Sử dụng pad theo chiều mặc định cho các trường hợp khác
                # Lưu ý: tuple padding cần có độ dài 2*n, n là số chiều cần pad.
                pad_tuple = [0] * (2 * data.dim())
                # Với dim, vị trí trong pad_tuple: padding cho chiều thứ (dim) sẽ nằm ở vị trí 2*(n-dim-1) và 2*(n-dim-1)+1
                index = 2 * (data.dim() - dim - 1)
                pad_tuple[index + 1] = pad_amount
                data = torch.nn.functional.pad(data, tuple(pad_tuple))
        elif current_size > size:
            if dim == 0:
                data = data[:size]
            elif dim == 1:
                data = data[:, :size]
            else:
                # Với các chiều khác, cần xử lý tùy vào dữ liệu
                slices = [slice(None)] * data.dim()
                slices[dim] = slice(0, size)
                data = data[tuple(slices)]
        # Kiểm tra lại
        assert data.size(dim) == size, f"Padding/cutting không thành công ở dim {dim}"
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
        
        # Convert to mel spectrogram using Whisper's feature extractor
        features = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt"
        ).input_features.squeeze(0)  # Remove batch dimension
        
        return features

    def load_video(self, path, max_frames=300):
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
        
        # Truncate to max_frames if needed
        if vid.size(0) > max_frames:
            vid = vid[:max_frames]
            
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
        max_frames=300
    ):
        # Initialize data processor
        self.processor = DataProcessor()
        self.root_dir = root_dir
        self.split = split
        self.modality = modality
        self.rate_ratio = rate_ratio
        self.max_frames = max_frames

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
        
        # Initialize return dictionary with None values
        return_dict = {
            "video": None,
            "audio": None,
            "video_mask": None,
            "audio_mask": None,
            "target": text
        }

        if self.modality in ["video", "audiovisual"]:
            # Load and process video for MOCO v2
            video = self.processor.load_video(video_path, self.max_frames)
            
            video = self.video_transform(video)
            
            return_dict["video"] = video
            return_dict["video_mask"] = torch.ones(video.size(0))
            
        if self.modality in ["audio", "audiovisual"]:
            # Load and process audio for Whisper
            audio = self.processor.load_audio(video_path)
            
            audio = self.processor.cut_or_pad(audio, size=3000, dim=1)
            
            # Apply audio transform for all audio modes
            audio = self.audio_transform(audio)
            
            return_dict["audio"] = audio
            return_dict["audio_mask"] = torch.ones(audio.size(0))

        return return_dict

    def __len__(self):
        return len(self.samples)
