import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import whisper
from transformers import WhisperModel
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
        
        # Dùng peMaxLen chỉ để định nghĩa phương pháp tính sinusoids, không dùng làm buffer cố định.
        self.pe_max_len = peMaxLen
        
        # Modal switch matrix
        modal_switch = torch.tensor([
            [1., 1., 1.],  # AV: use both audio and video
            [0., 1., 0.],  # AO: audio only
            [0., 0., 1.]   # VO: video only
        ])
        self.register_buffer("modal_switch", modal_switch, persistent=True)
        
        # Load Whisper model và freeze toàn bộ tham số
        self.whisper_model = WhisperModel.from_pretrained("SageLiao/whisper-small-zh-TW")
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        # Unfreeze 4 layer cuối của encoder để training
        for layer in self.whisper_model.encoder.layers[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        self.audio_proj = nn.Linear(self.whisper_model.config.d_model, dModel)
        self.audio_ln = LayerNorm(dModel)
        
        # Visual encoder (MoCoVisualFrontend) với projection
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
        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2, dtype=torch.float))
        positions = torch.arange(length, dtype=torch.float).unsqueeze(1)
        scaled_time = positions * inv_timescales.unsqueeze(0)
        sinusoids = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        return sinusoids

    def forward(self, inputBatch, return_audio=False):
        audioBatch, audioMask, videoBatch, videoLen = inputBatch

        # Process audio features through Whisper encoder
        whisper_out = self.whisper_model.encoder(audioBatch, attention_mask=~audioMask)[0]
        audio_features = self.audio_ln(self.audio_proj(whisper_out))
        # Tính lại positional embeddings cho audio_features
        T_audio = audio_features.shape[1]
        d_audio = audio_features.shape[-1]
        pos_emb_audio = self._get_sinusoids(T_audio, d_audio).to(audio_features.device)
        audio_features = audio_features + pos_emb_audio

        # Process video features
        video_features = self.visual_model(videoBatch, videoLen)
        video_features = self.video_ln(self.video_proj(video_features))
        # Tính lại positional embeddings cho video_features
        T_video = video_features.shape[1]
        d_video = video_features.shape[-1]
        pos_emb_video = self._get_sinusoids(T_video, d_video).to(video_features.device)
        video_features = video_features + pos_emb_video

        # Nếu số bước thời gian của audio và video khác nhau, căn chỉnh audio_features cho khớp với video_features
        if audio_features.shape[1] != video_features.shape[1]:
            audio_features = F.interpolate(
                audio_features.transpose(1, 2),
                size=video_features.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        # Nếu video_features có thêm chiều không cần thiết, squeeze nó
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
