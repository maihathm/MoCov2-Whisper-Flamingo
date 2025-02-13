dataset = AVDataset(
    root_dir="/root/maihathm/AVASR/data/avsr_self",
    split="train",  # hoặc "test", "valid"
    modality="audiovisual",  # hoặc "video", "audio"
    audio_transform=audio_transform,
    video_transform=video_transform,
    rate_ratio=640  # tỷ lệ giữa audio và video sampling rates
)

# 3. Tạo DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)