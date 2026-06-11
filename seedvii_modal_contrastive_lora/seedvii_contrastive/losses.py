"""
Tri-Modal Contrastive Loss for EEG-LLM Learning  ── v2 MoCo

损失函数:
1. Inter-modal: EEG ↔ LLM 跨模态对比  (+ MoCo queue negatives)
2. Intra-modal EEG: EEG样本间对比      (+ MoCo queue negatives)
3. Intra-modal LLM: LLM样本间对比

数值稳定性优化:
- BF16输入自动转换为FP32进行矩阵乘法
- Log-sum-exp数值稳定化
- 队列负样本用 torch.cat 拼接，单次矩阵乘法完成
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_dtype(tensor: torch.Tensor, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """确保张量为目标dtype以保证数值稳定"""
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
    queue_z: Optional[torch.Tensor] = None,
    queue_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Label-based Supervised Contrastive Loss (SupCon)  ── v2 支持 MoCo 队列.

    数学形式:
    L = -Σ_i log[ Σ_{j∈Pos(i)} exp(sim(z_i, z_j)/τ) / Σ_{k∈Neg(i)} exp(sim(z_i, z_k)/τ) ]

    Parameters
    ----------
    anchor:          (B, dim)   float32
    contrast:        (C, dim)   float32      batch 内的正/负样本
    anchor_labels:   (B,)       int64
    contrast_labels: (C,)       int64
    queue_z:         (K, dim) or None       MoCo 队列额外负样本
    queue_labels:    (K,)      or None      队列条目标签
    exclude_self:    bool                    intra-modal 时排除 anchor==contrast

    When queue_z is given, the contrast set becomes  [contrast ; queue_z]
    and contrast_labels become  [contrast_labels ; queue_labels].
    Labels are carried so positives can be detected across the queue.
    """
    if anchor.numel() == 0 or contrast.numel() == 0:
        return anchor.sum() * 0.0

    anchor = _ensure_dtype(anchor, torch.float32)
    contrast = _ensure_dtype(contrast, torch.float32)

    # ── 拼接队列负样本 ──
    if queue_z is not None and queue_labels is not None and queue_z.shape[0] > 0:
        queue_z = _ensure_dtype(queue_z, torch.float32).to(device=contrast.device)
        queue_labels = queue_labels.to(device=contrast.device, dtype=torch.int64)
        contrast_all = torch.cat([contrast, queue_z], dim=0)         # (C+K, dim)
        labels_all   = torch.cat([contrast_labels, queue_labels], dim=0)  # (C+K,)
    else:
        contrast_all = contrast
        labels_all   = contrast_labels

    # 相似度矩阵  (B, C+K)
    logits = anchor @ contrast_all.t() / temperature

    # 数值稳定化
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # 正负掩码
    pos_mask = anchor_labels[:, None].eq(labels_all[None, :])   # (B, C+K)

    # 排除自对比（intra-modal 时）
    if exclude_self and anchor.shape[0] == contrast.shape[0]:
        B = anchor.shape[0]
        eye = torch.eye(B, dtype=torch.bool, device=anchor.device)  # (B, B)
        # 只在 batch 部分排除自己；queue 部分不需要
        if queue_z is not None and queue_z.shape[0] > 0:
            eye = torch.cat([eye, torch.zeros(B, queue_z.shape[0], dtype=torch.bool, device=anchor.device)], dim=1)
        pos_mask = pos_mask & ~eye

    valid_den = torch.ones_like(logits, dtype=torch.bool)
    if exclude_self and anchor.shape[0] == contrast.shape[0]:
        eye = torch.eye(anchor.shape[0], dtype=torch.bool, device=anchor.device)
        if queue_z is not None and queue_z.shape[0] > 0:
            eye = torch.cat([eye, torch.zeros(anchor.shape[0], queue_z.shape[0], dtype=torch.bool, device=anchor.device)], dim=1)
        valid_den = valid_den & ~eye

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


def inter_modal_supcon(
    eeg_z: torch.Tensor,
    text_z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    queue_z: Optional[torch.Tensor] = None,
    queue_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """跨模态 SupCon: EEG↔Text  双向 + MoCo 队列负样本.

    对称双向对比损失，队列负样本附加在 text→eeg 侧（因为 EEG 队列
    提供额外的 EEG 负样本，让 text anchor 面对更大的负样本池）。
    EEG→text 方向直接使用 batch text embeddings 作为对比。
    """
    eeg_z = _ensure_dtype(eeg_z, torch.float32)
    text_z = _ensure_dtype(text_z, torch.float32)
    labels = labels.to(device=eeg_z.device, dtype=torch.int64)

    # eeg → text  (batch text as negatives — they're diverse enough)
    l_eeg_to_text = supervised_contrastive_loss(
        eeg_z, text_z, labels, labels, temperature, exclude_self=False,
    )

    # text → eeg  (+ queue negatives to dramatically enlarge the pool)
    l_text_to_eeg = supervised_contrastive_loss(
        text_z, eeg_z, labels, labels, temperature, exclude_self=False,
        queue_z=queue_z, queue_labels=queue_labels,
    )

    return 0.5 * (l_eeg_to_text + l_text_to_eeg)


def intra_modal_supcon(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    queue_z: Optional[torch.Tensor] = None,
    queue_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """模态内 SupCon  ── v2 支持队列负样本.

    exclude_self=True: 不与自身计算相似度.
    queue 中的同类项也被视为正样本（提供更多正信号）。
    """
    z = _ensure_dtype(z, torch.float32)
    labels = labels.to(device=z.device, dtype=torch.int64)
    return supervised_contrastive_loss(
        z, z, labels, labels, temperature, exclude_self=True,
        queue_z=queue_z, queue_labels=queue_labels,
    )


# 别名
inter_modal_loss = inter_modal_supcon
intra_modal_loss = intra_modal_supcon


class TriContrastiveLoss(nn.Module):
    """三模态对比损失  ── v2 MoCo

    总损失 = L_inter + β_eeg * L_eeg + β_llm * L_llm

    其中:
    - L_inter: EEG-Text 跨模态对比（text→eeg 侧加入队列负样本）
    - L_eeg:   EEG 模态内对比  （队列负样本加入）
    - L_llm:   LLM 模态内对比  （无队列 — EEG 队列对 LLM 空间无直接意义）
    """

    def __init__(
        self,
        temperature: float = 0.07,
        beta_eeg: float = 0.65,
        beta_llm: float = 0.35,
        intra_weight: float = 1.0,
    ):
        super().__init__()
        s = beta_eeg + beta_llm
        if abs(s - 1.0) > 1e-6:
            beta_eeg, beta_llm = beta_eeg / s, beta_llm / s

        self.temperature = temperature
        self.beta_eeg = beta_eeg
        self.beta_llm = beta_llm
        # intra_weight (lambda): 模态内损失整体相对跨模态(inter)的权重。
        # total = l_inter + intra_weight * (beta_eeg*l_eeg + beta_llm*l_llm)
        # =1.0 时与旧行为完全一致(向后兼容); <1 让 inter(跨模态对齐, eval考的)主导,
        # 抑制模态内塌缩捷径。推荐 0.2~0.5。
        self.intra_weight = float(intra_weight)

    def forward(
        self,
        eeg_z: torch.Tensor,
        text_z: torch.Tensor,
        labels: torch.Tensor,
        queue_z: Optional[torch.Tensor] = None,
        queue_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """计算三模态对比损失.

        Parameters
        ----------
        eeg_z:        (B, dim)  EEG 嵌入
        text_z:       (B, dim)  文本嵌入
        labels:       (B,)      标签
        queue_z:      (K, dim)   MoCo 队列 EEG embedding  或 None
        queue_labels: (K,)      队列标签                 或 None
        """
        eeg_z = _ensure_dtype(eeg_z, torch.float32)
        text_z = _ensure_dtype(text_z, torch.float32)
        labels = labels.to(device=eeg_z.device, dtype=torch.int64)

        l_inter = inter_modal_supcon(
            eeg_z, text_z, labels, self.temperature,
            queue_z=queue_z, queue_labels=queue_labels,
        )

        # If intra_weight is 0, *really* disable the intra-modal branches instead
        # of computing them and multiplying by 0.  This avoids wasted compute and
        # prevents disabled branches from propagating NaN/Inf into the total loss.
        if self.intra_weight > 0:
            l_eeg = intra_modal_supcon(
                eeg_z, labels, self.temperature,
                queue_z=queue_z, queue_labels=queue_labels,
            )
            l_llm = intra_modal_supcon(text_z, labels, self.temperature)
            intra = self.beta_eeg * l_eeg + self.beta_llm * l_llm
        else:
            l_eeg = l_inter.new_zeros(())
            l_llm = l_inter.new_zeros(())
            intra = l_inter.new_zeros(())

        total = l_inter + self.intra_weight * intra

        return {
            "loss": total,
            "inter": l_inter.detach(),
            "eeg_intra": l_eeg.detach(),
            "llm_intra": l_llm.detach(),
        }

    def __repr__(self):
        return (
            f"TriContrastiveLoss(\n"
            f"  temperature={self.temperature},\n"
            f"  beta_eeg={self.beta_eeg:.3f},\n"
            f"  beta_llm={self.beta_llm:.3f},\n"
            f"  intra_weight={self.intra_weight:.3f}\n"
            f")"
        )
