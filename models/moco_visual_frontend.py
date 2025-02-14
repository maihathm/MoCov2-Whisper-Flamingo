# moco_visual_frontend.py
import torch
import torch.nn as nn
import torchvision.models as models
import logging
from config import MODEL_CONFIG as args
from utils.logging_utils import log_tensor_info

logger = logging.getLogger(__name__)


class MoCoVisualFrontend(nn.Module):
    def __init__(self, d_model=args["FRONTEND_D_MODEL"], num_classes=args["WORD_NUM_CLASSES"], frame_length=args["FRAME_LENGTH"],
                 vid_feature_dim=args["VIDEO_FEATURE_SIZE"]):
        super(MoCoVisualFrontend, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.frame_length = frame_length
        self.vid_feature_dim = vid_feature_dim
        # Conv3D - input has 3 channels (RGB)
        self.frontend3D = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        # moco
        MoCoModel = models.__dict__['resnet50']()
        MoCoModel.fc = nn.Identity()
        MoCoModel.conv1 = nn.Identity()
        MoCoModel.bn1 = nn.Identity()
        MoCoModel.relu = nn.Identity()
        MoCoModel.maxpool = nn.Identity()
        self.MoCoModel = MoCoModel

    def forward(self, x, x_len):
        # Log input tensor info
        
        if x.dim() == 6 and x.shape[1] == 1:
            x = x.squeeze(1)
        x = x.permute(0, 2, 1, 3, 4)  # Reorder to 1 x C x T x H x W
        
        x = self.frontend3D(x)
        
        B, C, T, H, W = x.shape
        
        # Reshape to process each frame independently while maintaining batch structure
        x = x.transpose(1, 2).contiguous()  # [B, T, C, H, W]
        
        x = x.view(B * T, C, H, W)  # Combine batch and time dimensions
        
        # Create mask for valid frames
        mask = torch.arange(T, device=x.device).expand(B, T) >= x_len.unsqueeze(1)
        
        # Process through MoCo backbone
        x = self.MoCoModel(x)  # [B*T, 2048]
        
        # Reshape back to [B, T, 2048] and apply mask
        x = x.view(B, T, -1)  # Restore batch and temporal dimensions
        
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)  # Apply mask to zero out padding
        return x
