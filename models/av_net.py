import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from transformers import WhisperModel, WhisperConfig
from torch.nn.utils.rnn import pad_sequence

from .moco_visual_frontend import MoCoVisualFrontend
from .gate_cross_attention import GatedCrossModalFusion, LayerNorm
from .utils import PositionalEncoding, outputConv, generate_square_subsequent_mask


class AVNet(nn.Module):
    """
    Unified AVSR model with MOCO v2, Whisper, and Flamingo's gate cross attention.
    Designed for stable execution and ONNX compatibility.
    """
    def __init__(self, modal, MoCofile, reqInpLen, modelargs):
        super().__init__()
        dModel, nHeads, numLayers, peMaxLen, fcHiddenSize, dropout, numClasses = modelargs
        self.modal = modal
        self.numClasses = numClasses
        self.reqInpLen = reqInpLen
        
        # Model configuration validation
        self._validate_config(dModel, nHeads, numLayers, peMaxLen)
        
        # Static buffers for efficiency
        self.register_buffer("positional_embedding", self._get_sinusoids(peMaxLen, dModel), persistent=True)
        self.register_buffer("modal_switch", torch.eye(3), persistent=True)  # [AV, AO, VO]
        
        # Audio Encoder (Always initialize for ONNX compatibility)
        self.whisper_model = WhisperModel.from_pretrained("openai/whisper-base")
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        self.audio_proj = nn.Linear(self.whisper_model.config.d_model, dModel)
        self.audio_ln = LayerNorm(dModel)
            
        # Visual Encoder (Always initialize for ONNX compatibility)
        self.visual_model = MoCoVisualFrontend()
        if MoCofile is not None:
            self.visual_model.load_state_dict(torch.load(MoCofile, map_location="cpu"), strict=False)
        self.video_proj = nn.Linear(2048, dModel)
        self.video_ln = LayerNorm(dModel)
        # Freeze MOCO parameters
        for param in self.visual_model.parameters():
            param.requires_grad = False
            
        # Fusion Module (Always initialize for ONNX compatibility)
        self.fusion_module = GatedCrossModalFusion(
            d_model=dModel,
            n_heads=nHeads,
            n_layers=numLayers // 2,
            dropout=dropout
        )
        self.fusion_scalar = nn.Parameter(torch.tensor([1.0]))
            
        # Decoder with improved positional encoding
        self.token_embedding = nn.Embedding(numClasses, dModel)
        self.decoder_pos_embedding = nn.Parameter(torch.empty(peMaxLen, dModel))
        nn.init.normal_(self.decoder_pos_embedding, std=0.02)
        
        # Decoder layers with LayerNorm
        self.decoder_ln = LayerNorm(dModel)
        self.output_ln = LayerNorm(dModel)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dModel,
            nhead=nHeads,
            dim_feedforward=fcHiddenSize,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=numLayers,
            norm=LayerNorm(dModel)
        )
        
        # Output projection
        self.output_projection = nn.Linear(dModel, numClasses, bias=False)

    def _get_sinusoids(self, length, channels, max_timescale=10000):
        """Get sinusoidal positional embeddings"""
        if channels % 2 != 0:
            raise ValueError(f"Number of channels ({channels}) must be even")
            
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    def _validate_config(self, dModel, nHeads, numLayers, peMaxLen):
        """Validate model configuration for stable execution"""
        if dModel % nHeads != 0:
            raise ValueError(f"dModel ({dModel}) must be divisible by nHeads ({nHeads})")
        if peMaxLen > 10000:
            raise ValueError(f"peMaxLen ({peMaxLen}) should not exceed 10000 for stable positional encoding")
        if numLayers < 1:
            raise ValueError(f"numLayers ({numLayers}) must be at least 1")

    @torch.jit.export
    def get_modal_index(self, modal_str: str) -> int:
        """Get modal index for ONNX compatibility"""
        if modal_str == "AV":
            return 0
        elif modal_str == "AO":
            return 1
        elif modal_str == "VO":
            return 2
        else:
            return 0  # Default to AV

    def forward(self, inputBatch, targetinBatch=None, targetLenBatch=None):
        """
        Forward pass with unified handling of all modalities for ONNX compatibility
        """
        audioBatch, audioMask, videoBatch, videoLen = inputBatch
        
        # Process audio features
        with torch.no_grad():
            audio_features = self.whisper_model.encoder(
                audioBatch,
                attention_mask=~audioMask
            )[0]
        audio_features = self.audio_ln(self.audio_proj(audio_features))
        audio_features = audio_features + self.positional_embedding[:audio_features.shape[1]]
        
        # Process video features
        video_features = self.visual_model(videoBatch, videoLen)
        video_features = self.video_ln(self.video_proj(video_features))
        video_features = video_features + self.positional_embedding[:video_features.shape[1]]
        
        # Unified fusion handling
        modal_idx = self.get_modal_index(self.modal)
        modal_switch = self.modal_switch[modal_idx]
        
        # Fuse features based on modality
        fused_features = self.fusion_module(
            audio_features * modal_switch[1],  # Zero if VO
            video_features * modal_switch[2],  # Zero if AO
            audio_mask=audioMask if modal_switch[1] else None,
            video_mask=self.make_padding_mask(videoLen, video_features.shape[1]) if modal_switch[2] else None
        ) * self.fusion_scalar.tanh() * modal_switch[0]  # Apply fusion only for AV
        
        # Select appropriate features based on modality
        encoder_output = (
            fused_features * modal_switch[0] +  # AV mode
            audio_features * modal_switch[1] +  # AO mode
            video_features * modal_switch[2]    # VO mode
        )
            
        # If in training mode, process with decoder
        if targetinBatch is not None:
            # Prepare decoder input
            decoder_input = self.token_embedding(targetinBatch)
            decoder_input = decoder_input + self.decoder_pos_embedding[:decoder_input.shape[1]]
            
            # Generate masks
            tgt_mask = generate_square_subsequent_mask(decoder_input.shape[1], device=decoder_input.device)
            tgt_padding_mask = self.make_padding_mask(targetLenBatch, decoder_input.shape[1])
            
            # Decoder forward pass
            decoder_output = self.decoder(
                self.decoder_ln(decoder_input),
                encoder_output,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask
            )
            
            # Generate outputs
            logits = self.output_projection(self.output_ln(decoder_output))
            return logits
            
        return encoder_output

    def make_padding_mask(self, lengths, max_length):
        """Create padding mask from lengths"""
        batch_size = lengths.size(0)
        mask = torch.arange(max_length, device=lengths.device).expand(batch_size, max_length) >= lengths.unsqueeze(1)
        return mask

    def inference(self, inputBatch, device, Lambda=0.6, beamWidth=5):
        """
        Inference with beam search
        """
        audioBatch, audioMask, videoBatch, videoLen = inputBatch
        
        # Get features
        if not self.modal == "VO":
            with torch.no_grad():
                audio_features = self.whisper_model.encoder(
                    audioBatch,
                    attention_mask=~audioMask
                )[0]
            audio_features = self.audio_proj(audio_features)
            
        if not self.modal == "AO":
            video_features = self.visual_model(videoBatch, videoLen)
            video_features = self.video_proj(video_features)
            
        # Apply fusion
        if self.modal == "AV":
            fused_features = self.fusion_module(
                audio_features,
                video_features,
                audio_mask=audioMask,
                video_mask=self.make_padding_mask(videoLen, video_features.shape[1])
            )
        elif self.modal == "AO":
            fused_features = audio_features
        else:
            fused_features = video_features
            
        # Add positional encoding
        fused_features = self.EncoderPositionalEncoding(fused_features)
        
        # Get CTC probabilities
        ctc_output = self.jointOutputConv(fused_features.transpose(1, 2))
        ctc_output = ctc_output.transpose(1, 2)
        ctc_probs = F.softmax(ctc_output, dim=-1)
        
        # Initialize beam search
        batch_size = fused_features.size(0)
        beam_scores = torch.zeros(batch_size, beamWidth, device=device)
        beam_seqs = torch.full((batch_size, beamWidth, 1), self.numClasses-1, device=device)  # Start with EOS token
        beam_lens = torch.ones(batch_size, beamWidth, device=device)
        
        # Beam search loop
        max_length = min(fused_features.size(1) * 2, 200)  # Limit sequence length
        
        for step in range(max_length):
            num_beams = beam_seqs.size(1)
            
            # Get decoder input
            tgt = self.embed(beam_seqs.view(-1, step+1).transpose(0, 1))
            tgt_mask = generate_square_subsequent_mask(step+1, device)
            
            # Expand encoder output for each beam
            expanded_features = fused_features.unsqueeze(1).expand(-1, num_beams, -1, -1)
            expanded_features = expanded_features.contiguous().view(batch_size * num_beams, -1, fused_features.size(-1))
            
            # Decoder forward pass
            attn_output = self.jointAttentionDecoder(
                tgt,
                expanded_features.transpose(0, 1),
                tgt_mask=tgt_mask
            )
            
            # Get next token probabilities
            next_token_logits = self.jointAttentionOutputConv(attn_output[-1].unsqueeze(-1))
            next_token_logits = next_token_logits.squeeze(-1)
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            
            # Combine CTC and attention scores
            combined_scores = Lambda * next_token_probs + (1 - Lambda) * ctc_probs[:, step] if step < ctc_probs.size(1) else next_token_probs
            
            # Get top k candidates
            topk_probs, topk_ids = torch.topk(combined_scores, beamWidth, dim=-1)
            
            # Update beams
            new_seqs = torch.cat([beam_seqs, topk_ids.unsqueeze(-1)], dim=-1)
            new_scores = beam_scores + torch.log(topk_probs)
            
            # Select top beams
            beam_scores, beam_indices = torch.topk(new_scores.view(batch_size, -1), beamWidth, dim=-1)
            beam_seqs = new_seqs.view(batch_size, -1, step+2).gather(1, beam_indices.unsqueeze(-1).expand(-1, -1, step+2))
            
            # Update sequence lengths
            is_eos = (beam_seqs[:, :, -1] == self.numClasses-1)
            beam_lens = torch.where(is_eos & (beam_lens == step+1), beam_lens, step+2)
            
            # Early stopping if all beams ended
            if (beam_lens == step+1).all():
                break
                
        # Select best sequence from each batch
        best_seqs = beam_seqs[:, 0, 1:]  # Remove initial EOS token
        best_lens = beam_lens[:, 0] - 1   # Adjust length accordingly
        
        return best_seqs, best_lens
