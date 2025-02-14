import os
import math
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
    def __init__(self, modal, MoCofile, reqInpLen, modelargs, vocab_size, enable_logging=True):
        super().__init__()
        self.enable_logging = enable_logging
        self.modal = modal
        self.reqInpLen = reqInpLen
        self.pe_max_len = modelargs[3]
        self.modal_switch = torch.tensor([[1., 1., 1.],
                                            [0., 1., 0.],
                                            [0., 0., 1.]])
        self.whisper_model = WhisperModel.from_pretrained("openai/whisper-small")
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        for layer in self.whisper_model.encoder.layers[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.audio_proj = nn.Linear(self.whisper_model.config.d_model, modelargs[0])
        self.audio_ln = LayerNorm(modelargs[0])
        self.visual_model = MoCoVisualFrontend(enable_logging=self.enable_logging)
        if MoCofile is not None:
            state = torch.load(MoCofile, map_location="cpu")
            self.visual_model.load_state_dict(state, strict=False)
        self.video_proj = nn.Linear(2048, modelargs[0])
        self.video_ln = LayerNorm(modelargs[0])
        self.fusion_module = GatedCrossModalFusion(d_model=modelargs[0],
                                                   n_heads=modelargs[1],
                                                   n_layers=modelargs[2] // 2,
                                                   dropout=modelargs[5],
                                                   enable_logging=self.enable_logging)
        self.fusion_scalar = nn.Parameter(torch.tensor([1.0]))
        self.pos_enc_audio = PositionalEncoding(d_model=modelargs[0], max_len=5000, enable_logging=self.enable_logging)
        self.pos_enc_video = PositionalEncoding(d_model=modelargs[0], max_len=5000, enable_logging=self.enable_logging)
        self.decoder = Decoder(d_model=modelargs[0], vocab_size=vocab_size, enable_logging=self.enable_logging)
        if self.enable_logging:
            logger.info("AVNet initialized")
    def forward(self, inputBatch):
        if self.enable_logging:
            logger.info("AVNet forward start")
            
        # Unpack input batch
        audioBatch, audioMask, videoBatch, videoLen = inputBatch
        batch_size = audioBatch.size(0)
        
        if self.enable_logging:
            logger.info(f"Input audio shape: {audioBatch.shape}, mask shape: {audioMask.shape}")
            logger.info(f"Input video shape: {videoBatch.shape}, lengths shape: {videoLen.shape}")
        
        # Audio branch
        whisper_out = self.whisper_model.encoder(
            audioBatch, 
            attention_mask=~audioMask,
            output_hidden_states=True
        )[0]
        
        if self.enable_logging:
            logger.info(f"Whisper encoder output shape: {whisper_out.shape}")
            
        audio_features = self.audio_ln(self.audio_proj(whisper_out))
        audio_features = self.pos_enc_audio(audio_features)
        
        if self.enable_logging:
            logger.info(f"Audio features final shape: {audio_features.shape}")
        
        # Video branch
        video_features = self.visual_model(videoBatch, videoLen)
        video_features = self.video_ln(self.video_proj(video_features))
        video_features = self.pos_enc_video(video_features)
        
        if self.enable_logging:
            logger.info(f"Video features final shape: {video_features.shape}")
        
        # Align sequence lengths if needed
        if audio_features.shape[1] != video_features.shape[1]:
            target_len = min(audio_features.shape[1], video_features.shape[1])
            if audio_features.shape[1] > target_len:
                audio_features = audio_features[:, :target_len, :]
                audioMask = audioMask[:, :target_len]
            else:
                video_features = video_features[:, :target_len, :]
                videoLen = torch.clamp(videoLen, max=target_len)
            
            if self.enable_logging:
                logger.info(f"Features aligned to length: {target_len}")
        
        # Get modality switch
        modal_idx = self.get_modal_index(self.modal)
        modal_switch = self.modal_switch[modal_idx]
        
        # Create padding masks
        audio_padding_mask = audioMask if modal_switch[1] else None
        video_padding_mask = self.make_padding_mask(videoLen, video_features.shape[1]) if modal_switch[2] else None
        
        # Apply modality switches and fusion
        audio_branch = audio_features * modal_switch[1]
        video_branch = video_features * modal_switch[2]
        
        fused_features = self.fusion_module(
            audio_branch,
            video_branch,
            audio_mask=audio_padding_mask,
            video_mask=video_padding_mask
        ) * torch.tanh(self.fusion_scalar) * modal_switch[0]
        
        if self.enable_logging:
            logger.info(f"Fused features shape: {fused_features.shape}")
        
        # Combine features based on modality
        encoder_output = (
            fused_features * modal_switch[0] + 
            audio_features * modal_switch[1] + 
            video_features * modal_switch[2]
        )
        
        if self.enable_logging:
            logger.info(f"Encoder output shape: {encoder_output.shape}")
        
        # Generate logits
        logits = self.decoder(encoder_output)
        
        if self.enable_logging:
            logger.info(f"Final logits shape: {logits.shape}")
            logger.info("AVNet forward end")
        
        return logits
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
