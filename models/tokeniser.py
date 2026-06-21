"""
models/tokeniser.py

VQ-VAE tokeniser for the Physics Sandbox environment.

Converts continuous (64, 64, 3) RGB observations into a discrete
sequence of tokens, analogous to how MiniGrid observations are
naturally discrete integer grids. This lets the same GPT-style
transformer architecture (models/transformer.py) be applied to
both environments using a unified next-token prediction objective.

Architecture (van den Oord et al. 2017 — Neural Discrete Representation
Learning):
  Encoder: conv stack downsamples (64,64,3) -> (8,8,latent_dim) feature map
  Vector Quantisation: each of the 64 spatial positions is snapped to its
                       nearest codebook vector (one of num_embeddings)
  Decoder: conv-transpose stack reconstructs (64,64,3) from quantised codes

Output: 64 discrete tokens per frame (8x8 spatial grid), each in
[0, num_embeddings). This is the token sequence fed to the transformer
world model, exactly mirroring MiniGrid's flattened grid cell sequence.

Loss = reconstruction_loss + codebook_loss + commitment_cost * commitment_loss
  reconstruction_loss: MSE between input and reconstructed frame
  codebook_loss:        moves codebook vectors toward encoder outputs
  commitment_loss:      moves encoder outputs toward codebook vectors
                        (commitment_cost controls how strongly the encoder
                        commits to using existing codes vs. drifting freely)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQVAEConfig:
    """Hyperparameters for the VQ-VAE tokeniser, from physics_config.yaml."""
    obs_dim: int = 64           # input frame size (obs_dim x obs_dim x 3)
    latent_dim: int = 256       # channel dimension of latent feature map
    num_embeddings: int = 512   # codebook size (K)
    commitment_cost: float = 0.25
    learning_rate: float = 3e-4
    batch_size: int = 128
    n_epochs: int = 30

    @property
    def latent_spatial_size(self) -> int:
        """Spatial size of the latent grid after downsampling (64 -> 8 with 3 stride-2 convs)."""
        return self.obs_dim // 8

    @property
    def tokens_per_frame(self) -> int:
        return self.latent_spatial_size ** 2


class VectorQuantizer(nn.Module):
    """
    Vector quantisation layer: snaps continuous encoder outputs to the
    nearest of num_embeddings codebook vectors.

    Implements the straight-through estimator: gradients flow through
    the quantisation step as if it were the identity function, since
    argmin is non-differentiable.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Initialise codebook uniformly — standard VQ-VAE practice
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, C, H, W) encoder output, C = embedding_dim

        Returns:
            z_q:          (B, C, H, W) quantised output (straight-through)
            codebook_loss + commitment_loss: scalar combined VQ loss
            indices:      (B, H, W) codebook indices used (the discrete tokens)
        """
        B, C, H, W = z.shape

        # Flatten to (B*H*W, C) for distance computation
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (B*H*W, C)

        # Compute squared L2 distance to every codebook vector
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z.e
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )  # (B*H*W, num_embeddings)

        indices_flat = distances.argmin(dim=1)  # (B*H*W,)
        indices = indices_flat.view(B, H, W)

        z_q_flat = self.embedding(indices_flat)  # (B*H*W, C)
        z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # VQ losses
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from z_q to z
        z_q_st = z + (z_q - z).detach()

        return z_q_st, vq_loss, indices


class Encoder(nn.Module):
    """
    Conv encoder: (B, 3, 64, 64) -> (B, latent_dim, 8, 8)
    Three stride-2 conv layers, each halving spatial resolution.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),    # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_dim, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),  # refine, keep size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """
    Conv-transpose decoder: (B, latent_dim, 8, 8) -> (B, 3, 64, 64)
    Mirrors the encoder with stride-2 transposed convolutions.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),     # 32 -> 64
            nn.Sigmoid(),  # output in [0, 1], matches normalised input range
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.net(z_q)


class VQVAE(nn.Module):
    """
    Full VQ-VAE: Encoder -> VectorQuantizer -> Decoder.

    Usage:
        vqvae = VQVAE(config)
        x_hat, vq_loss, indices = vqvae(x)
        recon_loss = F.mse_loss(x_hat, x)
        total_loss = recon_loss + vq_loss

        # For tokenising a frame into the transformer's input sequence:
        tokens = vqvae.encode_to_tokens(x)  # (B, 64) flat token sequence
    """

    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.latent_dim)
        self.quantizer = VectorQuantizer(
            config.num_embeddings, config.latent_dim, config.commitment_cost
        )
        self.decoder = Decoder(config.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input frame, values in [0, 1]

        Returns:
            x_hat:   (B, 3, H, W) reconstruction
            vq_loss: scalar VQ loss (codebook + commitment)
            indices: (B, latent_h, latent_w) discrete codebook indices
        """
        z = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss, indices

    @torch.no_grad()
    def encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenise a batch of frames into flat discrete token sequences.

        Args:
            x: (B, 3, H, W) input frames, values in [0, 1]

        Returns:
            tokens: (B, tokens_per_frame) int64 flat token sequence,
                    matching the format expected by data/dataset.py and
                    models/transformer.py (same contract as MiniGrid's
                    flattened grid observations).
        """
        z = self.encoder(x)
        _, _, indices = self.quantizer(z)  # (B, H_latent, W_latent)
        B = indices.shape[0]
        return indices.view(B, -1).long()  # (B, tokens_per_frame)

    @torch.no_grad()
    def decode_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct frames from a flat token sequence (inverse of encode_to_tokens).
        Useful for visualising what the transformer's predicted tokens "look like".

        Args:
            tokens: (B, tokens_per_frame) int64

        Returns:
            x_hat: (B, 3, H, W) reconstructed frames
        """
        B = tokens.shape[0]
        spatial = self.config.latent_spatial_size
        indices = tokens.view(B, spatial, spatial)

        z_q_flat = self.quantizer.embedding(indices.view(-1))  # (B*spatial*spatial, latent_dim)
        z_q = z_q_flat.view(B, spatial, spatial, self.config.latent_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # (B, latent_dim, spatial, spatial)

        return self.decoder(z_q)
