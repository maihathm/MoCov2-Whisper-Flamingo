import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, enable_logging=True):
        super().__init__()
        self.enable_logging = enable_logging
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        if self.enable_logging:
            logger.info("PositionalEncoding initialized")
    def forward(self, x):
        if self.enable_logging:
            logger.info(f"PositionalEncoding input shape: {x.shape}")
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        if self.enable_logging:
            logger.info(f"PositionalEncoding output shape: {x.shape}")
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, vocab_size, enable_logging=True):
        super().__init__()
        self.enable_logging = enable_logging
        self.linear = nn.Linear(d_model, vocab_size)
        if self.enable_logging:
            logger.info("Decoder initialized")
    def forward(self, x):
        if self.enable_logging:
            logger.info(f"Decoder input shape: {x.shape}")
        pooled = x.mean(dim=1)
        if self.enable_logging:
            logger.info(f"Decoder pooled shape: {pooled.shape}")
        logits = self.linear(pooled)
        if self.enable_logging:
            logger.info(f"Decoder logits shape: {logits.shape}")
        return logits
