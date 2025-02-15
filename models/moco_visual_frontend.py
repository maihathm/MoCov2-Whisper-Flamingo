import torch
import torch.nn as nn
import torchvision.models as models
import logging
from config import MODEL_CONFIG as args

logger = logging.getLogger(__name__)

class MoCoVisualFrontend(nn.Module):
    def __init__(self, d_model=args["FRONTEND_D_MODEL"], num_classes=args["WORD_NUM_CLASSES"],
                 frame_length=args["FRAME_LENGTH"], vid_feature_dim=args["VIDEO_FEATURE_SIZE"], enable_logging=False):
        super(MoCoVisualFrontend, self).__init__()
        self.enable_logging = enable_logging
        self.frontend3D = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,3,3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        )
        MoCoModel = models.resnet50()
        MoCoModel.fc = nn.Identity()
        MoCoModel.conv1 = nn.Identity()
        MoCoModel.bn1 = nn.Identity()
        MoCoModel.relu = nn.Identity()
        MoCoModel.maxpool = nn.Identity()
        self.MoCoModel = MoCoModel
        if self.enable_logging:
            logger.info("MoCoVisualFrontend initialized")
    def forward(self, x, x_len):
        if self.enable_logging:
            logger.info(f"MoCoVisualFrontend input shape: {x.shape}")
        if x.dim() == 6 and x.shape[1] == 1:
            x = x.squeeze(1)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.frontend3D(x)
        if self.enable_logging:
            logger.info(f"After frontend3D, shape: {x.shape}")
        B, C, T, H, W = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(B * T, C, H, W)

        mask = torch.arange(T, device=x.device).expand(B, T) >= x_len.unsqueeze(1)


        x = self.MoCoModel(x)
        if self.enable_logging:
            logger.info(f"After MoCo backbone, shape: {x.shape}")
        x = x.view(B, T, -1)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        if self.enable_logging:
            logger.info(f"MoCoVisualFrontend output shape: {x.shape}")
        return x
