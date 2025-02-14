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
    def __init__(self, tokenizer_name="openai/whisper-small"):
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
        waveform_np = waveform.numpy()
        if waveform_np.ndim == 1:
            waveform_np = waveform_np[None, :]
        features = self.feature_extractor(
            waveform_np,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt"
        ).input_features
        if features.dim() == 3:
            features = features.squeeze(0)
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
    def __init__(self, root_dir, split, modality, audio_transform, video_transform, rate_ratio=640, max_frames=300, tokenizer_name="openai/whisper-small"):
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
    def _get_empty_sample(self):
        """Return an empty sample with correct shapes when loading fails"""
        return {
            "video": torch.zeros(self.max_frames, 3, 96, 96),
            "video_mask": torch.zeros(self.max_frames, dtype=torch.bool),
            "video_lengths": torch.tensor([0]),
            "audio": torch.zeros(3000, 80),
            "audio_mask": torch.zeros(3000, dtype=torch.bool),
            "audio_lengths": torch.tensor([0]),
            "target_ids": torch.zeros(1, dtype=torch.long),
            "target_text": "",
            "target_lengths": torch.tensor([1])
        }
    def __getitem__(self, idx):
        try:
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
                
                # Pad or trim video to max_frames
                current_frames = video.size(0)
                if current_frames > self.max_frames:
                    video = video[:self.max_frames]
                elif current_frames < self.max_frames:
                    padding = torch.zeros(self.max_frames - current_frames, *video.shape[1:])
                    video = torch.cat([video, padding], dim=0)
                
                # Create proper mask for actual frames
                video_mask = torch.zeros(self.max_frames, dtype=torch.bool)
                video_mask[:min(current_frames, self.max_frames)] = True
                
                return_dict["video"] = video
                return_dict["video_mask"] = video_mask
                return_dict["video_lengths"] = torch.tensor([min(current_frames, self.max_frames)])
                
            if self.modality in ["audio", "audiovisual"]:
                audio = self.processor.load_audio(video_path)
                audio = self.processor.cut_or_pad(audio, size=3000, dim=1)
                audio = self.audio_transform(audio)
                
                # Ensure consistent audio length
                audio_length = audio.size(0)
                target_length = 3000  # Or whatever your target length is
                
                if audio_length > target_length:
                    audio = audio[:target_length]
                elif audio_length < target_length:
                    padding = torch.zeros(target_length - audio_length, *audio.shape[1:])
                    audio = torch.cat([audio, padding], dim=0)
                
                # Create proper mask for actual audio frames
                audio_mask = torch.zeros(target_length, dtype=torch.bool)
                audio_mask[:min(audio_length, target_length)] = True
                
                return_dict["audio"] = audio
                return_dict["audio_mask"] = audio_mask
                return_dict["audio_lengths"] = torch.tensor([min(audio_length, target_length)])
            
            return return_dict
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            return self._get_empty_sample()
    def __len__(self):
        return len(self.samples)
