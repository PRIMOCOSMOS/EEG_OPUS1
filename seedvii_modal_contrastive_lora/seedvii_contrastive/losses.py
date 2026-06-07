from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def supervised_contrastive_loss(
    anchor: torch.Tensor,
    contrast: torch.Tensor,
    anchor_labels: torch.Tensor,
    contrast_labels: torch.Tensor,
    temperature: float = 0.07,
    exclude_self: bool = False,
) -> torch.Tensor:
    """Label-based Supervised Contrastive Loss (SupCon).

    This is the Khosla-style supervised contrastive objective generalized to
    cross-modal anchors/contrasts.

    For each anchor i:
      P(i) = {j | y_j == y_i}

    loss_i = - 1/|P(i)| * sum_{p in P(i)} log [ exp(sim(i,p)/tau)
                                                / sum_{a in A(i)} exp(sim(i,a)/tau) ]

    - For cross-modal EEG->Text / Text->EEG, exclude_self=False because anchor
      and contrast belong to different towers; the diagonal pair is a valid
      positive and should remain in the denominator.
    - For intra-modal EEG->EEG / Text->Text, exclude_self=True to remove the
      trivial self-comparison from both positives and denominator.

    Positive/negative definition follows the project requirement:
      same aggregated valence label => positive, different label => negative.
    Subject id, trial id and video id are deliberately ignored.
    """
    if anchor.numel() == 0 or contrast.numel() == 0:
        return anchor.sum() * 0.0

    logits = anchor @ contrast.t() / temperature  # (B_anchor, B_contrast)

    # Numerical stability: subtract row-wise max before logsumexp. This does not
    # change log-softmax probabilities.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    valid_den = torch.ones_like(logits, dtype=torch.bool)
    pos_mask = anchor_labels[:, None].eq(contrast_labels[None, :])

    if exclude_self and anchor.shape[0] == contrast.shape[0]:
        eye = torch.eye(anchor.shape[0], dtype=torch.bool, device=anchor.device)
        valid_den = valid_den & ~eye
        pos_mask = pos_mask & ~eye

    # Denominator: all valid contrasts except self for intra-modal SupCon.
    neg_inf = torch.finfo(logits.dtype).min
    logits_den = logits.masked_fill(~valid_den, neg_inf)
    log_den = torch.logsumexp(logits_den, dim=1, keepdim=True)
    log_prob = logits - log_den

    pos_count = pos_mask.sum(dim=1)  # (B_anchor,)
    valid_anchor = pos_count > 0
    if not torch.any(valid_anchor):
        # This can happen for intra-modal loss if a class appears only once in a
        # batch after removing self. Balanced batches should normally avoid it.
        return anchor.sum() * 0.0

    mean_log_prob_pos = (log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) /
                         pos_count.clamp_min(1))
    loss = -mean_log_prob_pos[valid_anchor].mean()
    return loss


def inter_modal_supcon(eeg_z: torch.Tensor, text_z: torch.Tensor,
                       labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric cross-modal SupCon: EEG->L2Text and L2Text->EEG."""
    return 0.5 * (
        supervised_contrastive_loss(eeg_z, text_z, labels, labels, temperature, exclude_self=False) +
        supervised_contrastive_loss(text_z, eeg_z, labels, labels, temperature, exclude_self=False)
    )


def intra_modal_supcon(z: torch.Tensor, labels: torch.Tensor,
                       temperature: float = 0.07) -> torch.Tensor:
    """Intra-modal SupCon with self-comparisons removed."""
    return supervised_contrastive_loss(z, z, labels, labels, temperature, exclude_self=True)


# Backward-compatible aliases used by older notes/tests.
inter_modal_loss = inter_modal_supcon
intra_modal_loss = intra_modal_supcon


class TriContrastiveLoss(nn.Module):
    """Total loss based on SupCon.

    L_total = L_inter_supcon + beta_eeg * L_eeg_supcon + beta_llm * L_llm_supcon

    beta_eeg + beta_llm is normalized to 1. Based on the user's prior
    experience, beta_eeg defaults to a slightly larger value.
    """
    def __init__(self, temperature: float = 0.07, beta_eeg: float = 0.65, beta_llm: float = 0.35):
        super().__init__()
        s = beta_eeg + beta_llm
        if abs(s - 1.0) > 1e-6:
            beta_eeg, beta_llm = beta_eeg / s, beta_llm / s
        self.temperature = temperature
        self.beta_eeg = beta_eeg
        self.beta_llm = beta_llm

    def forward(self, eeg_z: torch.Tensor, text_z: torch.Tensor, labels: torch.Tensor) -> dict:
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
