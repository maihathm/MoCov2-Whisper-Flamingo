import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    """Custom LayerNorm that can handle different dtypes"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)

class GatedCrossAttentionBlock(nn.Module):
    """
    Implements Flamingo's gated cross-attention mechanism following Whisper's implementation style
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalizations
        self.attn_ln = LayerNorm(d_model)
        self.cross_attn_ln = LayerNorm(d_model)
        self.ff_ln = LayerNorm(d_model)
        
        # Gating parameters (following Whisper's implementation)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))
        self.ff_gate = nn.Parameter(torch.tensor([0.]))
        
        # Feed-forward network
        n_mlp = d_model * 4
        self.ff = nn.Sequential(
            nn.Linear(d_model, n_mlp),
            nn.GELU(),
            nn.Linear(n_mlp, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, xa=None, mask=None):
        """
        Forward pass with gated cross-attention and feed-forward layers
        Args:
            x: Main features [B, T, D]
            xa: Cross-attention features [B, T, D]
            mask: Attention mask [B, T]
        """
        # Cross attention with gating
        if xa is not None:
            x = x + self.attn(
                self.attn_ln(x),
                xa,
                xa,
                key_padding_mask=mask,
                need_weights=False
            )[0] * self.attn_gate.tanh()
        
        # Feed-forward with gating
        x = x + self.ff(self.ff_ln(x)) * self.ff_gate.tanh()
        
        return x


class GatedCrossModalFusion(nn.Module):
    """
    Multi-layer gated cross-modal fusion following Whisper's implementation style
    """
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        
        # Projection layers
        self.audio_proj = nn.Linear(d_model, d_model)
        self.video_proj = nn.Linear(d_model, d_model)
        
        # Gated cross-attention layers
        self.layers = nn.ModuleList([
            GatedCrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_post = LayerNorm(d_model)
        
    def forward(self, audio_features, video_features, audio_mask=None, video_mask=None):
        """
        Multi-layer gated cross-modal fusion
        Args:
            audio_features: Audio features from Whisper [B, T_a, D]
            video_features: Video features from MOCO v2 [B, T_v, D]
            audio_mask: Audio attention mask [B, T_a]
            video_mask: Video attention mask [B, T_v]
        """
        # Project features
        x = self.audio_proj(audio_features)
        xa = self.video_proj(video_features)
        
        # Apply gated cross-attention layers
        for layer in self.layers:
            x = layer(x, xa, video_mask)
        
        return self.ln_post(x)
