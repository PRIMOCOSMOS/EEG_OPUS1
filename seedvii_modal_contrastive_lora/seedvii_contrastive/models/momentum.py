"""
Momentum Encoder wrapper for MoCo-style contrastive learning.

Keeps an exponential-moving-average (EMA) copy of the EEG encoder.
During training the online encoder receives gradient updates normally;
the momentum encoder is updated each step via

    theta_k  <-  m * theta_k  +  (1-m) * theta_q

and its output is fed into the contrastive queue (detached, no grad).

The momentum coefficient `m` should be close to 1 (default 0.999).
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


class MomentumEncoder(nn.Module):
    """EMA-shadow of an encoder used for generating queue negatives.

    Parameters
    ----------
    encoder:     the online encoder to shadow.  We deep-copy its init state.
    momentum:    EMA coefficient  (default 0.999)
    """

    def __init__(self, encoder: nn.Module, momentum: float = 0.999):
        super().__init__()
        self.momentum = momentum

        # Deep copy architecture + weights
        self._encoder = copy.deepcopy(encoder)

        # Freeze all parameters -- no gradient through this encoder EVER.
        for p in self._encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embeddings  (B, embed_dim)."""
        return self._encoder(x)

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        """EMA-update momentum encoder weights from online encoder.

        Called once per training step (or every N steps).
        """
        m = self.momentum
        for p_k, p_q in zip(self._encoder.parameters(), online_encoder.parameters()):
            p_k.data = m * p_k.data + (1.0 - m) * p_q.data

    @torch.no_grad()
    def copy_from(self, online_encoder: nn.Module) -> None:
        """Hard-copy all weights from the online encoder (used at init / resume)."""
        for p_k, p_q in zip(self._encoder.parameters(), online_encoder.parameters()):
            p_k.data.copy_(p_q.data)

    def train(self, mode: bool = True) -> "MomentumEncoder":
        """Always stays in eval mode -- no dropout/BN updates."""
        super().train(False)
        self._encoder.eval()
        return self

    def eval(self) -> "MomentumEncoder":
        return self.train(False)

    def __repr__(self) -> str:
        return f"MomentumEncoder(m={self.momentum}, encoder={self._encoder.__class__.__name__})"
