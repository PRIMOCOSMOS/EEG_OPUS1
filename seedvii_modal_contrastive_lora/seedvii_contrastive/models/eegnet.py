from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetEncoder(nn.Module):
    """Compact classic EEGNet encoder for 4s SEED-VII windows.

    Input shape: (B, 1, 62, T), T=800 for 4s@200Hz.
    Output: L2-normalized embedding of shape (B, embed_dim).
    
    FIX: Ensures consistent float32 output regardless of input dtype.
    """

    def __init__(
        self,
        chans: int = 62,
        samples: int = 800,
        embed_dim: int = 128,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning L2-normalized float32 embedding."""
        # FIX: Ensure input is float32 for numerical stability
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
            
        z = self.temporal(x)
        z = self.spatial(z)
        z = self.separable(z)
        z = self.pool(z)
        z = self.proj(z)
        
        # Ensure output is float32
        if z.dtype != torch.float32:
            z = z.to(dtype=torch.float32)
            
        return F.normalize(z, dim=-1)


class EEGNetClassifier(nn.Module):
    """Optional classifier wrapper for probing."""

    def __init__(self, encoder: EEGNetEncoder, embed_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.head(z), z
