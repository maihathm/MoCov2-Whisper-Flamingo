import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms
import random
import os
from PIL import Image

class LipVideoDataset(Dataset):
    def __init__(self, video_dir, frame_length=29, transform=None):
        """
        Dataset for loading lip videos without labels
        
        Args:
            video_dir: Directory containing video files
            frame_length: Number of frames to sample from each video
            transform: Additional transformations to apply
        """
        self.video_dir = video_dir
        self.frame_length = frame_length
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
        
        # Default augmentation pipeline
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if transform is None else transform

    def __len__(self):
        return len(self.video_files)

    def _load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Read all frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # Convert to numpy array
        frames = np.array(frames)
        
        # Sample frames if needed
        if len(frames) > self.frame_length:
            start_idx = random.randint(0, len(frames) - self.frame_length)
            frames = frames[start_idx:start_idx + self.frame_length]
        elif len(frames) < self.frame_length:
            # Pad with duplicates of last frame
            pad_frames = np.tile(frames[-1:], (self.frame_length - len(frames), 1, 1, 1))
            frames = np.concatenate([frames, pad_frames])
            
        return frames

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        frames = self._load_video(video_path)
        
        # Convert to torch tensor and apply transforms
        frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        
        # Apply same transform to all frames
        transformed_frames = []
        for frame in frames:
            transformed_frames.append(self.transform(frame))
        
        return torch.stack(transformed_frames)  # Shape: [T, C, H, W] 