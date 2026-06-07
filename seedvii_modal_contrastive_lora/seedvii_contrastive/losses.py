"""
Tri-Modal Contrastive Loss for EEG-LLM Learning

损失函数:
1. Inter-modal: EEG ↔ LLM 跨模态对比
2. Intra-modal EEG: EEG样本间对比
3. Intra-modal LLM: LLM样本间对比

数值稳定性优化:
- BF16输入自动转换为FP32进行矩阵乘法
- Log-sum-exp数值稳定化
"""
from __future__ import annotations

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
) -> torch.Tensor:
    """Label-based Supervised Contrastive Loss (SupCon)
    
    数学形式:
    L = -Σ_i log[ Σ_{j∈Pos(i)} exp(sim(z_i, z_j)/τ) / Σ_{k∈Neg(i)} exp(sim(z_i, z_k)/τ) ]
    
    其中:
    - Pos(i) = {j | y_j == y_i, j ≠ i}
    - Neg(i) = {k | y_k ≠ y_i}
    - sim(u, v) = u · v / ||u|| ||v|| = cosine similarity
    """
    if anchor.numel() == 0 or contrast.numel() == 0:
        return anchor.sum() * 0.0

    # 转换为float32确保数值稳定
    anchor = _ensure_dtype(anchor, torch.float32)
    contrast = _ensure_dtype(contrast, torch.float32)
    
    # 计算相似度矩阵
    logits = anchor @ contrast.t() / temperature
    
    # 数值稳定化: 减去最大值
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    
    # 创建有效掩码
    valid_den = torch.ones_like(logits, dtype=torch.bool)
    pos_mask = anchor_labels[:, None].eq(contrast_labels[None, :])
    
    # 排除自对比
    if exclude_self and anchor.shape[0] == contrast.shape[0]:
        eye = torch.eye(anchor.shape[0], dtype=torch.bool, device=anchor.device)
        valid_den = valid_den & ~eye
        pos_mask = pos_mask & ~eye
    
    # 计算log-sum-exp（分母）
    neg_inf = torch.finfo(logits.dtype).min
    logits_den = logits.masked_fill(~valid_den, neg_inf)
    log_den = torch.logsumexp(logits_den, dim=1, keepdim=True)
    log_prob = logits - log_den
    
    # 计算正样本对数概率和
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
    temperature: float = 0.07
) -> torch.Tensor:
    """跨模态SupCon: EEG→Text 和 Text→EEG
    
    对称双向对比损失，确保两个模态的表示空间对齐
    """
    eeg_z = _ensure_dtype(eeg_z, torch.float32)
    text_z = _ensure_dtype(text_z, torch.float32)
    labels = labels.to(device=eeg_z.device, dtype=torch.int64)
    
    # 双向对比损失
    l_egg_to_text = supervised_contrastive_loss(
        eeg_z, text_z, labels, labels, temperature, exclude_self=False
    )
    l_text_to_eeg = supervised_contrastive_loss(
        text_z, eeg_z, labels, labels, temperature, exclude_self=False
    )
    
    return 0.5 * (l_egg_to_text + l_text_to_eeg)


def intra_modal_supcon(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """模态内SupCon（同类别样本拉近，不同类别样本推远）
    
    exclude_self=True: 不与自身计算相似度
    """
    z = _ensure_dtype(z, torch.float32)
    labels = labels.to(device=z.device, dtype=torch.int64)
    return supervised_contrastive_loss(
        z, z, labels, labels, temperature, exclude_self=True
    )


# 别名
inter_modal_loss = inter_modal_supcon
intra_modal_loss = intra_modal_supcon


class TriContrastiveLoss(nn.Module):
    """三模态对比损失
    
    总损失 = λ_inter * L_inter + λ_eeg * L_eeg + λ_llm * L_llm
    
    其中:
    - L_inter: EEG-Text跨模态对比损失
    - L_eeg: EEG模态内对比损失
    - L_llm: LLM模态内对比损失
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        beta_eeg: float = 0.65,
        beta_llm: float = 0.35,
    ):
        super().__init__()
        
        # 归一化权重
        s = beta_eeg + beta_llm
        if abs(s - 1.0) > 1e-6:
            beta_eeg, beta_llm = beta_eeg / s, beta_llm / s
            
        self.temperature = temperature
        self.beta_eeg = beta_eeg
        self.beta_llm = beta_llm
        
    def forward(
        self,
        eeg_z: torch.Tensor,
        text_z: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """计算三模态对比损失
        
        Args:
            eeg_z: EEG嵌入 (B, embed_dim)
            text_z: 文本嵌入 (B, embed_dim)
            labels: 标签 (B,)
            
        Returns:
            dict with 'loss', 'inter', 'eeg_intra', 'llm_intra'
        """
        # 确保float32
        eeg_z = _ensure_dtype(eeg_z, torch.float32)
        text_z = _ensure_dtype(text_z, torch.float32)
        labels = labels.to(device=eeg_z.device, dtype=torch.int64)
        
        # 三种损失
        l_inter = inter_modal_supcon(eeg_z, text_z, labels, self.temperature)
        l_eeg = intra_modal_supcon(eeg_z, labels, self.temperature)
        l_llm = intra_modal_supcon(text_z, labels, self.temperature)
        
        # 加权和
        total = l_inter + self.beta_eeg * l_eeg + self.beta_llm * l_llm
        
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
            f"  beta_llm={self.beta_llm:.3f}\n"
            f")"
        )