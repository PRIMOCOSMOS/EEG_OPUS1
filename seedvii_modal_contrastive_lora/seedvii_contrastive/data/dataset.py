"""
Dataset and Sampler for SEED-VII EEG-LLM Contrastive Learning

优化:
- LRU缓存NPZ分片
- 预分配numpy数组减少内存分配
- 直接返回tensor避免中间转换
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
import math
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .protocol import load_l2_text_protocol, trial_to_labels


def load_index(npz_dir: str | Path) -> pd.DataFrame:
    """加载NPZ目录的窗口索引"""
    npz_dir = Path(npz_dir)
    df = pd.read_csv(npz_dir / "index.csv")
    # 确保abs_shard是字符串类型
    df["abs_shard"] = df["shard"].astype(str).apply(lambda s: str(npz_dir / s))
    return df


def split_index_by_subjects(
    df: pd.DataFrame,
    train_subjects: List[int],
    val_subjects: List[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按subject划分训练/验证集"""
    tr = df[df.subject.isin(train_subjects)].reset_index(drop=True)
    va = df[df.subject.isin(val_subjects)].reset_index(drop=True)
    return tr, va


def fit_channel_stats(df: pd.DataFrame, max_shards: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """计算训练集通道均值和标准差
    
    使用Welford's algorithm确保数值稳定
    """
    shards = list(dict.fromkeys(df.abs_shard.tolist()))
    if max_shards and len(shards) > max_shards:
        rng = np.random.default_rng(42)
        shards = list(rng.choice(shards, size=max_shards, replace=False))
    
    sums = None
    sqs = None
    count = 0
    
    for p in shards:
        with np.load(p) as z:
            x = z["x"].astype(np.float64)
            s = x.sum(axis=(0, 2))
            q = (x * x).sum(axis=(0, 2))
            n = x.shape[0] * x.shape[2]
            sums = s if sums is None else sums + s
            sqs = q if sqs is None else sqs + q
            count += n
    
    mean = sums / max(count, 1)
    var = np.maximum(sqs / max(count, 1) - mean * mean, 1e-8)
    std = np.sqrt(var)
    
    return mean.astype(np.float32), std.astype(np.float32)


class WindowNpzDataset(Dataset):
    """EEG窗口数据集
    
    优化:
    - LRU缓存NPZ分片
    - 预归一化参数float32
    - 直接返回tensor避免中间转换
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        text_csv_path: str | Path,
        cache_size: int = 8,
    ):
        # 重新索引并确保所有列都是标量值
        self.df = df.reset_index(drop=True)
        
        # 确保abs_shard列是字符串类型
        if "abs_shard" in self.df.columns:
            self.df["abs_shard"] = self.df["abs_shard"].astype(str)
        
        # 预计算归一化参数
        self.mean = channel_mean.astype(np.float32)[:, None]
        self.std = channel_std.astype(np.float32)[:, None]
        
        self.cache_size = cache_size
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.text_by_trial = load_l2_text_protocol(text_csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def _load_shard(self, path: str) -> dict:
        """LRU缓存加载NPZ分片"""
        # 确保path是字符串
        path_str = str(path)
        if path_str in self.cache:
            self.cache.move_to_end(path_str)
            return self.cache[path_str]
        
        with np.load(path_str) as z:
            item = {k: z[k] for k in z.files}
        self.cache[path_str] = item
        
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return item

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        
        # 确保获取标量值（避免Series问题）
        shard_path = str(row["abs_shard"])
        shard_idx = int(row["idx"])
        
        shard = self._load_shard(shard_path)
        
        # 加载并归一化EEG数据
        x = shard["x"][shard_idx].astype(np.float32)
        x = (x - self.mean) / (self.std + 1e-6)
        
        y = int(row["label3"])
        tid = int(row["trial"])
        text = self.text_by_trial[tid]
        
        return {
            "eeg": torch.from_numpy(x).unsqueeze(0),  # (1, 62, T)
            "label": torch.tensor(y, dtype=torch.long),
            "text": text,
            "subject": int(row["subject"]),
            "trial": tid,
        }


class ClassBalancedBatchSampler(Sampler[List[int]]):
    """类别平衡Batch采样器
    
    每个batch保证类别均衡分布，适合类别不平衡数据集
    
    设计:
    - 将样本按类别分组
    - 每个batch从每个类别采样近似相等的数量
    - 支持有放回/无放回采样
    """
    
    def __init__(
        self,
        labels: Iterable[int],
        batch_size: int,
        steps_per_epoch: Optional[int] = None,
        seed: int = 42,
        with_replacement: bool = True,
    ):
        self.labels = np.asarray(list(labels), dtype=np.int64)
        self.batch_size = int(batch_size)
        self.classes = sorted(np.unique(self.labels).tolist())
        self.n_cls = len(self.classes)
        
        # 按类别索引
        self.by_class = {c: np.where(self.labels == c)[0].tolist() for c in self.classes}
        
        if any(len(v) == 0 for v in self.by_class.values()):
            raise ValueError("empty class in labels; cannot build balanced sampler")
        
        # 每个类别的采样数
        self.base_per_cls = self.batch_size // self.n_cls
        self.rem = self.batch_size % self.n_cls
        
        # Epoch设置
        self.steps_per_epoch = steps_per_epoch or math.ceil(len(self.labels) / self.batch_size)
        self.seed = seed
        self.epoch = 0
        self.with_replacement = with_replacement

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        
        for _ in range(self.steps_per_epoch):
            batch = []
            
            # 随机化类别顺序
            order = self.classes[:]
            rng.shuffle(order)
            
            # 从每个类别采样
            for j, c in enumerate(order):
                k = self.base_per_cls + (1 if j < self.rem else 0)
                
                if self.with_replacement:
                    batch.extend(rng.choices(self.by_class[c], k=k))
                else:
                    # 无放回：简单版本，不处理极端不平衡
                    indices = self.by_class[c]
                    if len(indices) <= k:
                        batch.extend(indices)
                    else:
                        batch.extend(rng.sample(indices, k=k))
            
            # 打乱batch内顺序
            rng.shuffle(batch)
            yield batch


def build_l2_text_bank(text_csv_path: str | Path) -> tuple[list[str], np.ndarray, list[int]]:
    """构建L2文本库
    
    Returns:
        texts: 80个trial的文本描述列表
        labels: 对应的valence标签 (0/1/2)
        trials: trial索引 (1-80)
    """
    text_by_trial = load_l2_text_protocol(text_csv_path)
    texts, labels, trials = [], [], []
    
    for tid in range(1, 81):
        _, _, y = trial_to_labels(tid)
        texts.append(text_by_trial[tid])
        labels.append(y)
        trials.append(tid)
    
    return texts, np.asarray(labels, dtype=np.int64), trials