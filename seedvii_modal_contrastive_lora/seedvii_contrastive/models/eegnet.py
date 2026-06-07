"""
EEGNet Encoder for SEED-VII EEG-LLM Contrastive Learning

轻量级EEG编码器，专为4秒EEG窗口优化
支持混合精度训练和优化推理
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetEncoder(nn.Module):
    """Compact EEGNet encoder for SEED-VII windows.
    
    Input: (B, 1, 62, T), T=800 for 4s@200Hz
    Output: L2-normalized embedding (B, embed_dim)
    
    设计:
    - Temporal conv: 时间域特征提取
    - Spatial conv: 通道域注意力 (depthwise)
    - Separable conv: 深度可分离卷积
    - Global pooling: 自适应平均池化
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
        
        # 时间卷积
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), 
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        
        # 空间深度卷积
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        
        # 深度可分离卷积
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), 
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        
        # 全局池化和投影
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2, embed_dim),
        )
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """He初始化"""
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
        """前向传播
        
        Args:
            x: EEG输入 (B, 1, 62, T)
            
        Returns:
            L2归一化embedding (B, embed_dim)
        """
        # 确保float32输入
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
            
        # 时间卷积
        z = self.temporal(x)
        
        # 空间卷积
        z = self.spatial(z)
        
        # 可分离卷积
        z = self.separable(z)
        
        # 全局池化
        z = self.pool(z)
        
        # 投影
        z = self.proj(z)
        
        # 确保float32输出
        if z.dtype != torch.float32:
            z = z.to(dtype=torch.float32)
            
        return F.normalize(z, dim=-1)
    
    def get_embedding_dim(self) -> int:
        """获取embedding维度"""
        return self.proj[-1].out_features


class EEGNetClassifier(nn.Module):
    """EEG编码器 + 分类头的探测模型"""
    
    def __init__(self, encoder: EEGNetEncoder, embed_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.head(z), z
    
    def freeze_encoder(self):
        """冻结编码器参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """解冻编码器参数"""
        for param in self.encoder.parameters():
            param.requires_grad = True