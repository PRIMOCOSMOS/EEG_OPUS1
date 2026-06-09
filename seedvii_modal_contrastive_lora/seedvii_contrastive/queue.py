"""
MoCo (Momentum Contrast) FIFO Queue for EEG Contrastive Learning.

Key design:
- Fixed-size FIFO queue storing (embedding, label) pairs  (K approx 4096)
- Enqueue happens every training step via the momentum encoder
- Dequeue (oldest entries) + enqueue (new) = O(1) per step
- Queue labels are stored so SupCon can distinguish positive / negative
  entries correctly even for historical embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContrastiveQueue(nn.Module):
    """Fixed-size FIFO queue of L2-normalized embeddings with labels.

    Extremely lightweight: no backward graph is ever attached to queued
    tensors (they come from the momentum encoder under `no_grad`).

    Usage sketch::

        queue = ContrastiveQueue(embed_dim=256, queue_size=4096, n_classes=3)
        # ... per step ...
        with torch.no_grad():
            eeg_k = momentum_encoder(x)               # (B, dim)
        queue.enqueue(eeg_k, labels=y)                # + new entries
        extra_z, extra_lbl = queue.get()              # (K_filled, dim), (K_filled,)

        # use in SupCon:  contrast = [batch_z ; extra_z]  etc.

    Labels are stored so that `extra_lbl` can be compared with anchor
    labels in the supervised contrastive loss.
    """

    def __init__(self, embed_dim: int, queue_size: int = 4096, n_classes: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.queue_size = queue_size
        self.n_classes = n_classes

        # Unified buffer: [queue_size, embed_dim]
        self.register_buffer("embeddings", torch.zeros(queue_size, embed_dim))

        # Labels stored as long; -1 == empty slot
        self.register_buffer("labels", torch.full((queue_size,), -1, dtype=torch.long))

        # Write pointer: next insertion position (cyclical)
        self.register_buffer("_ptr", torch.zeros(1, dtype=torch.long))

        # Number of valid (non-empty) entries.  Cap = queue_size.
        self.register_buffer("_filled", torch.zeros(1, dtype=torch.long))

    # -- state accessors ------------------------------------------------
    @property
    def ptr(self) -> int:
        return int(self._ptr.item())

    @property
    def filled(self) -> int:
        return int(self._filled.item())

    @property
    def is_full(self) -> bool:
        return self.filled >= self.queue_size

    # -- enqueue --------------------------------------------------------
    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Push new embeddings (already L2-normalised) into the queue.

        Parameters
        ----------
        embeddings:  (B, embed_dim)  float32  already L2-normalised
        labels:      (B,)            int64    class labels
        """
        B = embeddings.shape[0]
        assert embeddings.shape[1] == self.embed_dim, f"{embeddings.shape[1]} != {self.embed_dim}"
        assert labels.shape == (B,), f"labels shape {labels.shape} != ({B},)"
        assert embeddings.device == self.embeddings.device

        # Cyclic write
        p = self.ptr
        if p + B <= self.queue_size:
            self.embeddings[p : p + B] = embeddings
            self.labels[p : p + B] = labels
        else:
            split = self.queue_size - p
            self.embeddings[p:] = embeddings[:split]
            self.labels[p:] = labels[:split]
            self.embeddings[: B - split] = embeddings[split:]
            self.labels[: B - split] = labels[split:]

        self._ptr[0] = (p + B) % self.queue_size
        self._filled[0] = min(self.queue_size, int(self._filled.item()) + B)

    # -- get (valid slice) ---------------------------------------------
    @torch.no_grad()
    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (embeddings, labels) of all filled entries.

        Returns
        -------
        z:  (filled, embed_dim)    L2-normalised
        y:  (filled,)              int64 labels (0 ... n_classes-1)
        """
        n = self.filled
        # no-clone: safe because caller (loss fn) only reads under no_grad / torch.cat
        return self.embeddings[:n], self.labels[:n]

    # -- reset ---------------------------------------------------------
    @torch.no_grad()
    def reset(self) -> None:
        self.embeddings.zero_()
        self.labels.fill_(-1)
        self._ptr.zero_()
        self._filled.zero_()

    # -- repr ----------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ContrastiveQueue(embed_dim={self.embed_dim}, "
            f"size={self.queue_size}, filled={self.filled})"
        )
