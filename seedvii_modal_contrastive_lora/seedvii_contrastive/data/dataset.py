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
    df = pd.read_csv(Path(npz_dir) / "index.csv")
    df["abs_shard"] = df["shard"].apply(lambda s: str(Path(npz_dir) / s))
    return df


def split_index_by_subjects(df: pd.DataFrame, train_subjects: List[int], val_subjects: List[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = df[df.subject.isin(train_subjects)].reset_index(drop=True)
    va = df[df.subject.isin(val_subjects)].reset_index(drop=True)
    return tr, va


def fit_channel_stats(df: pd.DataFrame, max_shards: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-channel mean/std on training rows only. x shape in shards: (N,62,T)."""
    shards = list(dict.fromkeys(df.abs_shard.tolist()))
    if max_shards and len(shards) > max_shards:
        rng = np.random.default_rng(42)
        shards = list(rng.choice(shards, size=max_shards, replace=False))
    sums = None; sqs = None; count = 0
    for p in shards:
        with np.load(p) as z:
            x = z["x"].astype(np.float64)  # (N,C,T)
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
    def __init__(
        self,
        df: pd.DataFrame,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        text_csv_path: str | Path,
        cache_size: int = 8,
    ):
        self.df = df.reset_index(drop=True)
        self.mean = channel_mean.astype(np.float32)[:, None]
        self.std = channel_std.astype(np.float32)[:, None]
        self.cache_size = cache_size
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.text_by_trial = load_l2_text_protocol(text_csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def _load_shard(self, path: str) -> dict:
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        with np.load(path) as z:
            item = {k: z[k] for k in z.files}
        self.cache[path] = item
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return item

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        shard = self._load_shard(row.abs_shard)
        x = shard["x"][int(row.idx)].astype(np.float32)  # (62,T)
        x = (x - self.mean) / (self.std + 1e-6)
        y = int(row.label3)
        tid = int(row.trial)
        text = self.text_by_trial[tid]  # L2 text only, exactly from protocol CSV.
        return {
            "eeg": torch.from_numpy(x).unsqueeze(0),  # (1,62,T)
            "label": torch.tensor(y, dtype=torch.long),
            "text": text,
            "subject": int(row.subject),
            "trial": tid,
        }


class ClassBalancedBatchSampler(Sampler[List[int]]):
    """Batch sampler that draws approximately equal samples per class with replacement.

    This is useful for supervised contrastive losses because each batch should contain
    positives for all three classes.
    """
    def __init__(self, labels: Iterable[int], batch_size: int, steps_per_epoch: Optional[int] = None, seed: int = 42):
        self.labels = np.asarray(list(labels), dtype=np.int64)
        self.batch_size = int(batch_size)
        self.classes = sorted(np.unique(self.labels).tolist())
        self.by_class = {c: np.where(self.labels == c)[0].tolist() for c in self.classes}
        if any(len(v) == 0 for v in self.by_class.values()):
            raise ValueError("empty class in labels; cannot build balanced sampler")
        self.steps_per_epoch = steps_per_epoch or math.ceil(len(self.labels) / self.batch_size)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        n_cls = len(self.classes)
        base = self.batch_size // n_cls
        rem = self.batch_size % n_cls
        for _ in range(self.steps_per_epoch):
            batch = []
            order = self.classes[:]
            rng.shuffle(order)
            for j, c in enumerate(order):
                k = base + (1 if j < rem else 0)
                batch.extend(rng.choices(self.by_class[c], k=k))
            rng.shuffle(batch)
            yield batch


def build_l2_text_bank(text_csv_path: str | Path) -> tuple[list[str], np.ndarray, list[int]]:
    """Return all 80 L2 texts, their aggregated labels, and trial ids."""
    text_by_trial = load_l2_text_protocol(text_csv_path)
    texts, labels, trials = [], [], []
    for tid in range(1, 81):
        _, _, y = trial_to_labels(tid)
        texts.append(text_by_trial[tid])
        labels.append(y)
        trials.append(tid)
    return texts, np.asarray(labels, dtype=np.int64), trials
