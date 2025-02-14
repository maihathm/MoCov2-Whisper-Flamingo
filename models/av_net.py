import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from transformers import WhisperModel
import numpy as np

from .moco_visual_frontend import MoCoVisualFrontend
from .gate_cross_attention import GatedCrossModalFusion, LayerNorm

class AVNet(nn.Module):
    """
    Unified AVSR model with MOCO v2, Whisper, and Flamingo's gate cross attention.
    Designed for stable execution and ONNX compatibility.
    """
    def __init__(self, modal, MoCofile, reqInpLen, modelargs):
        super().__init__()
        dModel, nHeads, numLayers, peMaxLen, fcHiddenSize, dropout = modelargs
        self.modal = modal
        self.reqInpLen = reqInpLen
        
        # Positional embeddings and modal switch matrix
        self.register_buffer("positional_embedding", self._get_sinusoids(peMaxLen, dModel), persistent=True)
        modal_switch = torch.tensor([
            [1., 1., 1.],  # AV: use both audio and video
            [0., 1., 0.],  # AO: audio only
            [0., 0., 1.]   # VO: video only
        ])
        self.register_buffer("modal_switch", modal_switch, persistent=True)
        
        # Load Whisper model and freeze all parameters
        self.whisper_model = WhisperModel.from_pretrained("SageLiao/whisper-small-zh-TW")
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        # Unfreeze last 4 layers of the encoder for training
        for layer in self.whisper_model.encoder.layers[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        self.audio_proj = nn.Linear(self.whisper_model.config.d_model, dModel)
        self.audio_ln = LayerNorm(dModel)
        
        # Visual encoder (MoCoVisualFrontend) with projection
        self.visual_model = MoCoVisualFrontend()
        if MoCofile is not None:
            self.visual_model.load_state_dict(torch.load(MoCofile, map_location="cpu"), strict=False)
        self.video_proj = nn.Linear(2048, dModel)
        self.video_ln = LayerNorm(dModel)
        for param in self.visual_model.parameters():
            param.requires_grad = True
        
        # Fusion module
        self.fusion_module = GatedCrossModalFusion(
            d_model=dModel,
            n_heads=nHeads,
            n_layers=numLayers // 2,
            dropout=dropout
        )
        self.fusion_scalar = nn.Parameter(torch.tensor([1.0]))
    
    def _get_sinusoids(self, length, channels, max_timescale=10000):
        if channels % 2 != 0:
            raise ValueError("Number of channels must be even")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    
    def forward(self, inputBatch, return_audio=False):
        """
        Forward pass.
        Nếu return_audio=True, trả về tuple (encoder_output, audio_features)
        để dùng làm target tính loss.
        """
        audioBatch, audioMask, videoBatch, videoLen = inputBatch

        # Process audio features through Whisper encoder
        # (không dùng torch.no_grad() để cho phép gradient trong các lớp mở)
        whisper_out = self.whisper_model.encoder(audioBatch, attention_mask=~audioMask)[0]
        audio_features = self.audio_ln(self.audio_proj(whisper_out))
        audio_features = audio_features + self.positional_embedding[:audio_features.shape[1]]
        
        # Process video features
        video_features = self.visual_model(videoBatch, videoLen)
        video_features = self.video_ln(self.video_proj(video_features))
        video_features = video_features + self.positional_embedding[:video_features.shape[1]]
        
        # Interpolate audio_features nếu số bước thời gian không khớp với video_features
        if audio_features.shape[1] != video_features.shape[1]:
            audio_features = F.interpolate(
                audio_features.transpose(1, 2),
                size=video_features.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        # Nếu video_features có thêm một chiều (ví dụ [B, S, T, dModel]), squeeze nó
        if video_features.dim() == 4:
            video_features = video_features.squeeze(1)  # [B, T, dModel]
        
        modal_idx = self.get_modal_index(self.modal)
        modal_switch = self.modal_switch[modal_idx]
        
        fused_features = self.fusion_module(
            audio_features * modal_switch[1],
            video_features * modal_switch[2],
            audio_mask=audioMask if modal_switch[1] else None,
            video_mask=self.make_padding_mask(videoLen, video_features.shape[1]) if modal_switch[2] else None
        ) * self.fusion_scalar.tanh() * modal_switch[0]
        
        encoder_output = (
            fused_features * modal_switch[0] +
            audio_features * modal_switch[1] +
            video_features * modal_switch[2]
        )
        
        if return_audio:
            return encoder_output, audio_features
        return encoder_output
    
    def get_modal_index(self, modal_str: str) -> int:
        if modal_str == "AV":
            return 0
        elif modal_str == "AO":
            return 1
        elif modal_str == "VO":
            return 2
        return 0
    
    def make_padding_mask(self, lengths, max_length):
        batch_size = lengths.size(0)
        mask = torch.arange(max_length, device=lengths.device).expand(batch_size, max_length) >= lengths.unsqueeze(1)
        return mask
    
    def inference(self, inputBatch):
        return self.forward(inputBatch)
