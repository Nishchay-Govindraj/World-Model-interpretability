"""
models/transformer.py

GPT-style decoder-only transformer world model.

Trained via next-token prediction on observation sequences from
MiniGrid and PhysicsSandbox environments. No state supervision —
any world model structure must emerge from the prediction objective alone.
This follows the protocol established by Li et al. (2023) in Othello-GPT.

Architecture:
  - Token embedding + learned positional embedding
  - N decoder blocks: LayerNorm -> CausalSelfAttention -> LayerNorm -> MLP
  - Pre-norm (GPT-2 style) for training stability
  - Final LayerNorm + linear unembedding to vocab logits

Residual stream access:
  - register_residual_hook(layer) returns a hook handle that caches
    the residual stream output after each block
  - Used by the probe pipeline without any overhead during normal training

Two model sizes (from model_config.yaml):
  small: 6 layers, 8 heads, d_model=256  (~5M params)  — RTX 3060
  large: 12 layers, 8 heads, d_model=512 (~25M params) — A100 cluster
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """
    All hyperparameters for the transformer world model.
    Constructed from model_config.yaml entries.
    """
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    dropout: float
    context_length: int
    vocab_size: int

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_config_dict(cls, cfg: dict, scale: str = "small") -> "TransformerConfig":
        """
        Build TransformerConfig from the model_config.yaml dict.

        Args:
            cfg:   full model config dict loaded from YAML
            scale: "small" or "large"
        """
        scale_cfg = cfg[scale]
        return cls(
            n_layers=scale_cfg["n_layers"],
            n_heads=scale_cfg["n_heads"],
            d_model=scale_cfg["d_model"],
            d_ff=scale_cfg["d_ff"],
            dropout=scale_cfg["dropout"],
            context_length=scale_cfg["context_length"],
            vocab_size=scale_cfg["vocab_size"],
        )

    def num_parameters(self) -> int:
        """Estimate total parameter count."""
        embed   = self.vocab_size * self.d_model
        pos     = self.context_length * self.d_model
        per_layer = (
            4 * self.d_model * self.d_model +   # QKV + O projections
            2 * self.d_model * self.d_ff +       # MLP up + down
            4 * self.d_model                     # 4 LayerNorm weight+bias pairs
        )
        unembed = self.d_model * self.vocab_size
        return embed + pos + self.n_layers * per_layer + unembed


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Each token attends only to itself and previous tokens (causal mask).
    Uses scaled dot-product attention with fused kernel when available
    (torch.nn.functional.scaled_dot_product_attention, PyTorch 2.0+).
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head  = config.d_head
        self.d_model = config.d_model

        # Fused QKV projection — more efficient than 3 separate linears
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj  = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Causal mask — registered as buffer so it moves with the model to GPU
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.bool())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        B, T, C = x.shape

        # Project to Q, K, V — split along last dim
        qkv = self.qkv_proj(x)                          # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)       # each (B, T, d_model)

        # Reshape to (B, n_heads, T, d_head)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention with causal mask
        # Use PyTorch 2.0 fused kernel when available
        attn_mask = self.causal_mask[:T, :T]             # (T, T)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        # Merge heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))


class MLP(nn.Module):
    """
    Position-wise feed-forward network.
    Uses GELU activation following GPT-2.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1  = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2  = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Single transformer block: pre-norm -> attention -> pre-norm -> MLP.
    Residual connections around both sub-layers.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.d_model)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class WorldModelTransformer(nn.Module):
    """
    GPT-style decoder-only transformer world model.

    Input:  integer token sequence (B, T)
    Output: logits over vocabulary (B, T, vocab_size)

    Loss: cross-entropy on next-token prediction
          (computed externally in the training script)

    Residual stream access:
        Use get_residual_stream(tokens) to get activations at every layer
        for a given input. Used by the probe pipeline.
        Not called during normal training — zero overhead.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed   = nn.Embedding(config.context_length, config.d_model)
        self.drop        = nn.Dropout(config.dropout)
        self.blocks      = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_final    = nn.LayerNorm(config.d_model)
        self.unembed     = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: token embedding and unembedding share weights
        # Reduces parameters and often improves performance (Press & Wolf, 2017)
        self.unembed.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise weights following GPT-2:
          - Embeddings: N(0, 0.02)
          - Linear layers: N(0, 0.02)
          - Residual projections scaled by 1/sqrt(2*n_layers) for stability
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale residual projections
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=scale * 0.02)
            nn.init.normal_(block.mlp.fc2.weight,       mean=0.0, std=scale * 0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            tokens:  (B, T) int64 token indices
            targets: (B, T) int64 target tokens for loss computation (optional)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss if targets provided, else None
        """
        B, T = tokens.shape
        assert T <= self.config.context_length, (
            f"Sequence length {T} exceeds context_length {self.config.context_length}"
        )

        device = tokens.device
        positions = torch.arange(T, device=device).unsqueeze(0)   # (1, T)

        x = self.drop(self.token_embed(tokens) + self.pos_embed(positions))

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.unembed(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def get_residual_stream(
        self,
        tokens: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Return residual stream activations after each transformer block.

        Used by the linear probe pipeline. NOT called during training.

        Args:
            tokens: (B, T) int64

        Returns:
            List of length n_layers, each tensor of shape (B, T, d_model).
            Index 0 = after block 0, index n_layers-1 = after final block.
        """
        B, T = tokens.shape
        device = tokens.device
        positions = torch.arange(T, device=device).unsqueeze(0)

        x = self.drop(self.token_embed(tokens) + self.pos_embed(positions))

        residual_stream = []
        for block in self.blocks:
            x = block(x)
            residual_stream.append(x.clone())   # clone to avoid in-place issues

        return residual_stream

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def get_attention_weights(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return attention weight matrices for all layers and heads.

        NOTE: The standard forward() uses scaled_dot_product_attention (fused
        kernel, PyTorch 2.0+) which does NOT return attention weights. This
        method uses a manual softmax attention computation for interpretability
        purposes only — it is NOT called during training.

        Args:
            tokens: (B, T) int64, typically B=1 for interpretability

        Returns:
            attn_weights: (n_layers, n_heads, T, T) float32
                          attn_weights[l, h, i, j] = weight from position j
                          to position i (row i sums to 1.0)
        """
        B, T = tokens.shape
        device = tokens.device
        positions = torch.arange(T, device=device).unsqueeze(0)

        x = self.drop(self.token_embed(tokens) + self.pos_embed(positions))

        all_attn_weights = []

        for block in self.blocks:
            # Pre-norm
            x_norm = block.ln1(x)

            # Manual QKV projection (mirrors CausalSelfAttention)
            attn = block.attn
            qkv = attn.qkv_proj(x_norm)
            q, k, v = qkv.split(attn.d_model, dim=-1)

            def reshape(t):
                return t.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)

            q, k, v = reshape(q), reshape(k), reshape(v)

            # Manual scaled dot-product attention — captures weights explicitly
            scale = math.sqrt(attn.d_head)
            raw_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)

            # Apply causal mask
            causal_mask = attn.causal_mask[:T, :T]  # (T, T)
            raw_scores = raw_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            weights = torch.softmax(raw_scores, dim=-1)  # (B, n_heads, T, T)
            all_attn_weights.append(weights[0].cpu())    # (n_heads, T, T), drop batch dim

            # Continue the forward pass normally using the fused kernel
            # to keep residual stream correct for subsequent layers
            attn_out = block.attn(x_norm)
            x = x + attn_out
            x = x + block.mlp(block.ln2(x))

        # Stack: (n_layers, n_heads, T, T)
        return torch.stack(all_attn_weights)


def build_model(config_dict: dict, scale: str = "small") -> WorldModelTransformer:
    """
    Convenience function: build model from YAML config dict.

    Args:
        config_dict: loaded model_config.yaml
        scale:       "small" or "large"

    Returns:
        WorldModelTransformer on CPU (move to device in training script)
    """
    cfg = TransformerConfig.from_config_dict(config_dict, scale=scale)
    model = WorldModelTransformer(cfg)
    n_params = model.num_parameters()
    print(f"Built {scale} transformer: {n_params:,} parameters "
          f"({n_params/1e6:.1f}M)")
    return model


def load_model(
    checkpoint_path: str,
    config_dict: dict,
    scale: str = "small",
    device: str = "cpu",
) -> WorldModelTransformer:
    """
    Load a trained model from a checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint file
        config_dict:     loaded model_config.yaml
        scale:           "small" or "large"
        device:          "cuda", "cpu", etc.

    Returns:
        WorldModelTransformer with loaded weights, in eval mode
    """
    model = build_model(config_dict, scale=scale)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Handle checkpoints saved with/without "model_state_dict" wrapper
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model
