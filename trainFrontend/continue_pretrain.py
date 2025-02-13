import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from tqdm import tqdm
import os
import argparse

from models.moco_visual_frontend import MoCoVisualFrontend
from trainFrontend.lip_video_dataset import LipVideoDataset

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features_1, features_2):
        # Normalize features
        features_1 = nn.functional.normalize(features_1, dim=1)
        features_2 = nn.functional.normalize(features_2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features_1, features_2.T) / self.temperature
        
        # Labels are diagonal (positive pairs)
        labels = torch.arange(similarity.size(0), device=similarity.device)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(similarity, labels)
        return loss

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def train_epoch(model, train_loader, optimizer, loss_fn, device, rank):
    model.train()
    total_loss = 0
    
    if rank == 0:
        progress = tqdm(train_loader, desc="Training")
    else:
        progress = train_loader
        
    for batch in progress:
        # Shape: [B, T, C, H, W]
        videos = batch.to(device)
        batch_size = videos.size(0)
        
        # Create two augmented views
        idx1 = torch.randint(0, videos.size(1), (batch_size,))
        idx2 = torch.randint(0, videos.size(1), (batch_size,))
        
        view1 = videos[torch.arange(batch_size), idx1]  # Shape: [B, C, H, W]
        view2 = videos[torch.arange(batch_size), idx2]
        
        # Get features
        features1 = model(view1)
        features2 = model(view2)
        
        # Compute loss
        loss = loss_fn(features1, features2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if rank == 0:
            progress.set_postfix({"Loss": loss.item()})
    
    return total_loss / len(train_loader)

def continue_pretrain(rank, world_size, args):
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Create model
    model = MoCoVisualFrontend(
        dModel=args.frontend_dmodel,
        nClasses=args.num_classes,
        frameLen=args.frame_length,
        vidfeaturedim=args.video_feature_size
    )
    
    # Load checkpoint if exists
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=f"cuda:{rank}")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Create dataset and dataloader
    dataset = LipVideoDataset(
        video_dir=args.video_dir,
        frame_length=args.frame_length
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = ContrastiveLoss(temperature=args.temperature)
    
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        sampler.set_epoch(epoch)
        train_loss = train_epoch(model, loader, optimizer, loss_fn, rank, rank)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {train_loss:.4f}")
            
            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            }
            torch.save(
                checkpoint,
                os.path.join(args.save_dir, f"visual_frontend_checkpoint_epoch_{epoch+1}.pt")
            )
    
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--frontend_dmodel", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=500)
    parser.add_argument("--frame_length", type=int, default=29)
    parser.add_argument("--video_feature_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Launch training processes
    torch.multiprocessing.spawn(
        continue_pretrain,
        args=(args.num_gpus, args),
        nprocs=args.num_gpus,
        join=True
    ) 