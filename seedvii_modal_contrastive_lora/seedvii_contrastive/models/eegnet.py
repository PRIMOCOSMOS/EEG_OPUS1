"""
EEGNet Encoder for SEED-VII EEG-LLM Contrastive Learning  (v2 — 时序保留版)

为什么要改 (v1 的问题):
- v1 在 proj 前用 ``AdaptiveAvgPool2d((1,1))`` 把整段时间轴 (4s@200Hz 经卷积栈后
  剩 25 个时间步) **全部平均成 1 个点**, 只留下 F2(=32) 维进入 Linear。
- 于是 ``Linear(32 -> 256)`` 是纯线性升维, 输出 256 维向量的 **秩 <= 32**:
  名义 256 维, 真实信息瓶颈只有 32 维, 且 **时序动态被完全抹掉**。
- 对跨被试 SEED-VII 三分类 (难任务) 来说, EEG 塔表达力远不及 text 塔
  (Qwen+LoRA 输出信息丰富的 256 维), 两塔严重不对等 -> 对比学习塌缩到退化解
  (loss 贴着数学下限, val f1 ~ 0.33 随机)。

v2 的改造 (对齐 SEED-VII 4s 切窗的量级):
1. **保留时间分辨率**: 卷积栈后是 (F2, 1, ~25)。改用 ``AdaptiveAvgPool2d((1, T_keep))``
   把时间压到 ``time_pool`` 段 (默认 4) 而不是 1, 然后 flatten 出 ``F2 * time_pool`` 维。
   这样既保留时序结构, 又对窗长鲁棒 (自适应池化, 窗口/降采样变化都不会改变输出维度)。
2. **加宽容量**: 默认 F1=16, D=2, F2=64 (v1 默认 F2=16/cfg 用 32), 让卷积特征更丰富。
3. **真正满秩的投影头**: ``Linear(F2*time_pool -> hidden) -> GELU -> Dropout -> Linear(hidden -> embed_dim)``,
   进入 proj 的维度 = F2*time_pool (例如 64*4=256), 输出 256 维是"真的" 256 维。

输入:  (B, 1, 62, T), T=800 for 4s@200Hz
输出:  L2-normalized embedding (B, embed_dim)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetEncoder(nn.Module):
    """Compact-but-expressive EEGNet encoder for SEED-VII 4s windows.

    Parameters
    ----------
    chans:         EEG 通道数 (SEED-VII = 62)
    samples:       窗口采样点数 (4s@200Hz = 800); 仅用于文档/校验, 网络对长度自适应
    embed_dim:     输出嵌入维度 (与 text 塔对齐, 默认 256)
    F1:            时间卷积滤波器数
    D:             depthwise 空间卷积的深度倍数 (空间滤波器 = F1*D)
    F2:            separable 卷积输出通道数 (信息瓶颈宽度)
    kernel_length: 时间卷积核长 (64 ≈ 0.32s@200Hz, 覆盖低频节律)
    dropout:       dropout 比例
    time_pool:     **关键** — proj 前在时间轴上保留的段数 (v1 等价于 1; v2 默认 4)
    proj_hidden:   投影头隐藏层维度 (默认 = embed_dim)
    """

    def __init__(
        self,
        chans: int = 62,
        samples: int = 800,
        embed_dim: int = 256,
        F1: int = 16,
        D: int = 2,
        F2: int = 64,
        kernel_length: int = 64,
        dropout: float = 0.25,
        time_pool: int = 4,
        proj_hidden: int | None = None,
        spatial_max_norm: float = 1.0,
        dense_max_norm: float = 0.25,
    ):
        super().__init__()
        self.chans = chans
        self.samples = samples
        self.embed_dim = embed_dim
        self.time_pool = int(time_pool)
        if self.time_pool < 1:
            raise ValueError(f"time_pool must be >= 1, got {time_pool}")
        proj_hidden = proj_hidden or embed_dim
        # EEGNet 论文的 kernel/dense 权重约束 (max-norm). <=0 表示关闭。
        self.spatial_max_norm = float(spatial_max_norm)
        self.dense_max_norm = float(dense_max_norm)

        # ── 时间卷积 (temporal) ──
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )

        # ── 空间深度卷积 (spatial, depthwise over channels) ──
        # 单独持有该 conv 引用, 以便施加 EEGNet 论文的 max_norm=1 约束。
        self.spatial_conv = nn.Conv2d(F1, F1 * D, kernel_size=(chans, 1), groups=F1, bias=False)
        self.spatial = nn.Sequential(
            self.spatial_conv,
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # ── 深度可分离卷积 (separable) ──
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # ── 时序保留池化: 压到 (1, time_pool) 而不是 (1, 1) ──
        # 自适应池化 => 对窗长/降采样率变化鲁棒, 输出维度恒为 F2*time_pool
        self.time_reduce = nn.AdaptiveAvgPool2d((1, self.time_pool))

        # ── 投影头: 真正的非线性升维, 输入维度 = F2*time_pool ──
        # 末端 Linear 单独持有引用, 施加 EEGNet 论文的 dense max_norm=0.25 约束。
        feat_dim = F2 * self.time_pool
        self.dense = nn.Linear(proj_hidden, embed_dim)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, proj_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            self.dense,
        )

        self._init_weights()

    @torch.no_grad()
    def _apply_max_norm(self):
        """EEGNet 标志性正则: 对空间卷积核 / 末端全连接施加 max-norm 约束.

        论文 (Lawhern 2018): depthwise spatial conv 用 max_norm=1, 分类全连接用
        max_norm=0.25. 这里在每次 forward 时投影回约束球内 (训练时生效)。
        """
        if self.spatial_max_norm and self.spatial_max_norm > 0:
            w = self.spatial_conv.weight  # (F1*D, 1, chans, 1)
            norm = w.norm(2, dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
            desired = norm.clamp(max=self.spatial_max_norm)
            w.mul_(desired / norm)
        if self.dense_max_norm and self.dense_max_norm > 0:
            w = self.dense.weight  # (embed_dim, proj_hidden)
            norm = w.norm(2, dim=1, keepdim=True).clamp_min(1e-8)
            desired = norm.clamp(max=self.dense_max_norm)
            w.mul_(desired / norm)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播.

        Args:
            x: EEG 输入 (B, 1, 62, T)
        Returns:
            L2 归一化 embedding (B, embed_dim)
        """
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)

        # EEGNet max-norm 权重约束 (训练时每步生效)
        if self.training:
            self._apply_max_norm()

        z = self.temporal(x)
        z = self.spatial(z)
        z = self.separable(z)
        z = self.time_reduce(z)      # (B, F2, 1, time_pool)  ← 保留时序
        z = self.proj(z)             # (B, embed_dim)

        if z.dtype != torch.float32:
            z = z.to(dtype=torch.float32)

        return F.normalize(z, dim=-1)

    def get_embedding_dim(self) -> int:
        return self.embed_dim


class EEGNetClassifier(nn.Module):
    """EEG 编码器 + 线性分类头 (linear-probe 用)."""

    def __init__(self, encoder: EEGNetEncoder, embed_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.head(z), z

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
