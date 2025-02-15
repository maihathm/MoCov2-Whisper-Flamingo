import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)

class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, enable_logging=False):
        super().__init__()
        self.enable_logging = enable_logging
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn_ln = LayerNorm(d_model)
        self.ff_ln = LayerNorm(d_model)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))
        self.ff_gate = nn.Parameter(torch.tensor([0.]))
        n_mlp = d_model * 4
        self.ff = nn.Sequential(
            nn.Linear(d_model, n_mlp),
            nn.GELU(),
            nn.Linear(n_mlp, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x, xa=None, mask=None):
        if self.enable_logging:
            logger.info(f"GatedCrossAttentionBlock input x shape: {x.shape}")
            if xa is not None:
                logger.info(f"Cross attention input xa shape: {xa.shape}")
            if mask is not None:
                logger.info(f"Attention mask shape: {mask.shape}")
        if xa is not None and xa.dim() == 4 and xa.size(1) == 1:
            xa = xa.squeeze(1)
        if mask is not None and mask.dim() == 3 and mask.size(1) == 1:
            mask = mask.squeeze(1)
        if xa is not None:
            attn_out = self.attn(self.attn_ln(x), xa, xa, key_padding_mask=mask, need_weights=False)[0]
            if self.enable_logging:
                logger.info(f"Attention output shape: {attn_out.shape}")
            x = x + attn_out * self.attn_gate.tanh()
        ff_out = self.ff(self.ff_ln(x))
        if self.enable_logging:
            logger.info(f"Feed-forward output shape: {ff_out.shape}")
        x = x + ff_out * self.ff_gate.tanh()
        if self.enable_logging:
            logger.info(f"GatedCrossAttentionBlock output shape: {x.shape}")
        return x

class GatedCrossModalFusion(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1, enable_logging=False):
        super().__init__()
        self.enable_logging = enable_logging
        self.audio_proj = nn.Linear(d_model, d_model)
        self.video_proj = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([GatedCrossAttentionBlock(d_model, n_heads, dropout, enable_logging=self.enable_logging) for _ in range(n_layers)])
        self.ln_post = LayerNorm(d_model)
    def forward(self, audio_features, video_features, audio_mask=None, video_mask=None):
        if self.enable_logging:
            logger.info(f"Fusion input audio_features shape: {audio_features.shape}")
            logger.info(f"Fusion input video_features shape: {video_features.shape}")
        x = self.audio_proj(audio_features)
        xa = self.video_proj(video_features)
        if self.enable_logging:
            logger.info(f"After projection, x shape: {x.shape}, xa shape: {xa.shape}")
        for layer in self.layers:
            x = layer(x, xa, video_mask)
        x = self.ln_post(x)
        if self.enable_logging:
            logger.info(f"Fusion output shape: {x.shape}")
        return x
