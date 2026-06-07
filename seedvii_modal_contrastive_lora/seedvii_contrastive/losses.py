from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_dtype(tensor: torch.Tensor, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Ensure tensor has the target dtype for numerical stability."""
    if tensor.dtype != target_dtype:
        return tensor.to(dtype=target_dtype)
    return tensor


def supervised_contrastive_loss(
    anchor: torch.Tensor,
    contrast: torch.Tensor,
    anchor_labels: torch.Tensor,
    contrast_labels: torch.Tensor,
    temperature: float = 0.07,
    exclude_self: bool = False,
) -> torch.Tensor:
    """Label-based Supervised Contrastive Loss (SupCon).
    
    Handles mixed dtype (float32 vs bfloat16) by casting to float32.
    """
    if anchor.numel() == 0 or contrast.numel() == 0:
        return anchor.sum() * 0.0

    # FIX: Cast to float32 for numerical stability in contrastive loss
    target_dtype = torch.float32
    anchor = _ensure_dtype(anchor, target_dtype)
    contrast = _ensure_dtype(contrast, target_dtype)

    logits = anchor @ contrast.t() / temperature

    # Numerical stability
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    valid_den = torch.ones_like(logits, dtype=torch.bool)
    pos_mask = anchor_labels[:, None].eq(contrast_labels[None, :])

    if exclude_self and anchor.shape[0] == contrast.shape[0]:
        eye = torch.eye(anchor.shape[0], dtype=torch.bool, device=anchor.device)
        valid_den = valid_den & ~eye
        pos_mask = pos_mask & ~eye

    neg_inf = torch.finfo(logits.dtype).min
    logits_den = logits.masked_fill(~valid_den, neg_inf)
    log_den = torch.logsumexp(logits_den, dim=1, keepdim=True)
    log_prob = logits - log_den

    pos_count = pos_mask.sum(dim=1)
    valid_anchor = pos_count > 0
    if not torch.any(valid_anchor):
        return anchor.sum() * 0.0

    mean_log_prob_pos = (log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) /
                         pos_count.clamp_min(1))
    loss = -mean_log_prob_pos[valid_anchor].mean()
    return loss


def inter_modal_supcon(eeg_z: torch.Tensor, text_z: torch.Tensor,
                       labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric cross-modal SupCon: EEG->L2Text and L2Text->EEG."""
    # FIX: Ensure consistent dtype before loss computation
    target_dtype = torch.float32
    eeg_z = _ensure_dtype(eeg_z, target_dtype)
    text_z = _ensure_dtype(text_z, target_dtype)
    labels = labels.to(device=eeg_z.device, dtype=torch.int64)
    
    return 0.5 * (
        supervised_contrastive_loss(eeg_z, text_z, labels, labels, temperature, exclude_self=False) +
        supervised_contrastive_loss(text_z, eeg_z, labels, labels, temperature, exclude_self=False)
    )


def intra_modal_supcon(z: torch.Tensor, labels: torch.Tensor,
                       temperature: float = 0.07) -> torch.Tensor:
    """Intra-modal SupCon with self-comparisons removed."""
    target_dtype = torch.float32
    z = _ensure_dtype(z, target_dtype)
    labels = labels.to(device=z.device, dtype=torch.int64)
    return supervised_contrastive_loss(z, z, labels, labels, temperature, exclude_self=True)


# Aliases
inter_modal_loss = inter_modal_supcon
intra_modal_loss = intra_modal_supcon


class TriContrastiveLoss(nn.Module):
    """Total loss based on SupCon."""

    def __init__(self, temperature: float = 0.07, beta_eeg: float = 0.65, beta_llm: float = 0.35):
        super().__init__()
        s = beta_eeg + beta_llm
        if abs(s - 1.0) > 1e-6:
            beta_eeg, beta_llm = beta_eeg / s, beta_llm / s
        self.temperature = temperature
        self.beta_eeg = beta_eeg
        self.beta_llm = beta_llm

    def forward(self, eeg_z: torch.Tensor, text_z: torch.Tensor, labels: torch.Tensor) -> dict:
        # FIX: Ensure consistent dtype before loss computation
        target_dtype = torch.float32
        eeg_z = _ensure_dtype(eeg_z, target_dtype)
        text_z = _ensure_dtype(text_z, target_dtype)
        labels = labels.to(device=eeg_z.device, dtype=torch.int64)

        l_inter = inter_modal_supcon(eeg_z, text_z, labels, self.temperature)
        l_eeg = intra_modal_supcon(eeg_z, labels, self.temperature)
        l_llm = intra_modal_supcon(text_z, labels, self.temperature)
        total = l_inter + self.beta_eeg * l_eeg + self.beta_llm * l_llm
        return {
            "loss": total,
            "inter": l_inter.detach(),
            "eeg_intra": l_eeg.detach(),
            "llm_intra": l_llm.detach(),
        }
