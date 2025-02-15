# av_net.py (rút gọn code freeze, d_model=256,...)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel
import logging
from .moco_visual_frontend import MoCoVisualFrontend
from .gate_cross_attention import GatedCrossModalFusion, LayerNorm
from .utils import PositionalEncoding, Decoder

logger = logging.getLogger(__name__)

class AVNet(nn.Module):
    def __init__(self, modal, MoCofile, reqInpLen, modelargs, vocab_size, enable_logging=False):
        super().__init__()
        self.enable_logging = enable_logging
        self.modal = modal
        self.reqInpLen = reqInpLen

        # Mô hình Whisper
        self.whisper_model = WhisperModel.from_pretrained("openai/whisper-small")

        # **Freeze** toàn bộ encoder (nếu muốn)
        for param in self.whisper_model.encoder.parameters():
            param.requires_grad = False

        # (Tuỳ chọn) unfreeze 1-2 layer cuối:
        # for layer in self.whisper_model.encoder.layers[-2:]:
        #    for param in layer.parameters():
        #        param.requires_grad = True

        d_model = modelargs[0]  # 256
        # Chiều encoder whisper => d_model
        self.audio_proj = nn.Linear(self.whisper_model.config.d_model, d_model)
        self.audio_ln = LayerNorm(d_model)

        # Video MoCo
        self.visual_model = MoCoVisualFrontend(enable_logging=self.enable_logging)
        if MoCofile is not None:
            state = torch.load(MoCofile, map_location="cpu")
            self.visual_model.load_state_dict(state, strict=False)

        # **Freeze** MoCo backbone
        for param in self.visual_model.parameters():
            param.requires_grad = False

        self.video_proj = nn.Linear(2048, d_model)
        self.video_ln = LayerNorm(d_model)

        self.fusion_module = GatedCrossModalFusion(
            d_model=d_model,
            n_heads=modelargs[1],      # 4
            n_layers=modelargs[2]//2,  # 1 => do config n_layers=2
            dropout=modelargs[5],
            enable_logging=self.enable_logging
        )
        self.fusion_scalar = nn.Parameter(torch.tensor([1.0]))

        self.pos_enc_audio = PositionalEncoding(d_model, enable_logging=self.enable_logging)
        self.pos_enc_video = PositionalEncoding(d_model, enable_logging=self.enable_logging)

        self.decoder = Decoder(d_model=d_model, vocab_size=vocab_size, enable_logging=self.enable_logging)
        if self.enable_logging:
            logger.info("AVNet (reduced) initialized")

    def forward(self, inputBatch):
        if self.enable_logging:
            logger.info("AVNet forward start")

        audioBatch, audioMask, videoBatch, videoMask, videoLen = inputBatch

        # AUDIO
        # [B, 3000,80] => [B,80,3000]
        if audioBatch.shape[1] == 3000 and audioBatch.shape[2] == 80:
            audioBatch = audioBatch.transpose(1,2)

        whisper_out = self.whisper_model.encoder(
            audioBatch,
            attention_mask=~audioMask
        )[0]  # [B, T, 768]
        
        audio_features = self.audio_ln(self.audio_proj(whisper_out))
        audio_features = self.pos_enc_audio(audio_features)

        # VIDEO
        video_features = self.visual_model(videoBatch, videoLen)
        video_features = self.video_ln(self.video_proj(video_features))
        video_features = self.pos_enc_video(video_features)

        # align
        min_len = min(audio_features.size(1), video_features.size(1))
        audio_features = audio_features[:, :min_len]
        video_features = video_features[:, :min_len]
        audioMask = audioMask[:, :min_len]
        videoLen.clamp_(max=min_len)

        # fusion
        fused = self.fusion_module(
            audio_features, 
            video_features,
            audio_mask=audioMask,
            video_mask=self.make_padding_mask(videoLen, min_len)
        )
        # combine
        out = fused + audio_features + video_features

        logits = self.decoder(out)
        return logits

    def make_padding_mask(self, lengths, max_length):
        batch_size = lengths.size(0)
        return (torch.arange(max_length, device=lengths.device).expand(batch_size, max_length)
                >= lengths.unsqueeze(1))
