import os

import torch
import torchaudio
import torchvision


def cut_or_pad(data, size, dim=0):
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


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)


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
            video = load_video(video_path)
            video = self.video_transform(video)
            return {"input": video, "target": text}
            
        elif self.modality == "audio":
            audio = load_audio(video_path)  # audio path được tạo từ video path
            audio = self.audio_transform(audio)
            return {"input": audio, "target": text}
            
        elif self.modality == "audiovisual":
            video = load_video(video_path)
            audio = load_audio(video_path)
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            video = self.video_transform(video)
            audio = self.audio_transform(audio)
            return {"video": video, "audio": audio, "target": text}

    def __len__(self):
        return len(self.samples)
