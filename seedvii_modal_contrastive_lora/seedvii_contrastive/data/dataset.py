"""
Dataset and Sampler for SEED-VII EEG-LLM Contrastive Learning.

Key robustness guarantees:
- ``index.csv`` written by preprocessing does not need to contain ``abs_shard``;
  it is derived from ``npz_dir`` at load time.
- Numeric columns stay numeric.  This is important because the YAML config stores
  ``train_subjects`` / ``val_subjects`` as integers.
- Empty/malformed indices fail early with actionable error messages instead of
  surfacing later as pandas ``KeyError`` or DataLoader errors.
- NPZ shards are loaded through a small LRU cache.
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


_REQUIRED_INDEX_COLUMNS = {"shard", "idx", "subject", "trial", "label3"}
_INT_COLUMNS = ["idx", "subject", "trial", "label3", "window_id", "start_sample", "n_samples_clip"]


def _coerce_int_column(df: pd.DataFrame, col: str, *, required: bool = False) -> None:
    """Coerce one dataframe column to integer in-place with a clear error."""
    if col not in df.columns:
        if required:
            raise ValueError(f"index.csv is missing required column: {col!r}")
        return
    values = pd.to_numeric(df[col], errors="coerce")
    bad = values.isna()
    if bad.any():
        examples = df.loc[bad, col].head(5).tolist()
        raise ValueError(f"index.csv column {col!r} contains non-integer values: {examples}")
    df[col] = values.astype(np.int64)


def normalize_index_dataframe(df: pd.DataFrame, npz_dir: str | Path | None = None) -> pd.DataFrame:
    """Return a normalized copy of a SEED-VII window index dataframe.

    Parameters
    ----------
    df:
        Raw dataframe, normally loaded from ``index.csv``.
    npz_dir:
        Directory containing the shard files.  When supplied, ``abs_shard`` is
        always rebuilt from ``shard`` so stale/moved indices remain usable.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = sorted(_REQUIRED_INDEX_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"index.csv is missing required columns {missing}; got columns={df.columns.tolist()}"
        )

    df["shard"] = df["shard"].astype(str).str.strip()
    if (df["shard"] == "").any():
        raise ValueError("index.csv contains empty shard names")

    if npz_dir is not None:
        root = Path(npz_dir).expanduser()
        # Always derive abs_shard from the current npz_dir.  This fixes old
        # indices that did not store abs_shard and indices moved between machines.
        df["abs_shard"] = df["shard"].map(lambda s: str((root / s).resolve()))
    elif "abs_shard" in df.columns:
        df["abs_shard"] = df["abs_shard"].astype(str)
    else:
        raise ValueError("index dataframe has no 'abs_shard'; load it with load_index(npz_dir)")

    for col in _INT_COLUMNS:
        _coerce_int_column(df, col, required=col in _REQUIRED_INDEX_COLUMNS)

    if "fine_emotion" in df.columns:
        df["fine_emotion"] = df["fine_emotion"].astype(str)
    if "valence" in df.columns:
        df["valence"] = df["valence"].astype(str)

    return df.reset_index(drop=True)


def load_index(npz_dir: str | Path) -> pd.DataFrame:
    """Load and validate the NPZ window index.

    The preprocessing writer stores relative shard names in ``index.csv``.  This
    loader adds/refreshes ``abs_shard`` and keeps ``subject/trial/label3`` as
    integers so subject splitting works with integer lists from YAML.
    """
    npz_dir = Path(npz_dir).expanduser()
    index_path = npz_dir / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"NPZ index not found: {index_path}. Run seedvii_contrastive.scripts.preprocess_npz first."
        )

    df = pd.read_csv(index_path)
    df = normalize_index_dataframe(df, npz_dir=npz_dir)

    if len(df) == 0:
        raise ValueError(
            f"NPZ index is empty: {index_path}. Preprocessing produced zero windows; "
            "check the .mat files, fs/window-sec/center-ratio settings, and rerun preprocessing."
        )

    missing_shards = [p for p in dict.fromkeys(df["abs_shard"].tolist()) if not Path(p).exists()]
    if missing_shards:
        shown = missing_shards[:5]
        raise FileNotFoundError(
            f"index.csv references missing NPZ shard files under {npz_dir}: {shown}"
            f"{' ...' if len(missing_shards) > 5 else ''}. Rerun preprocessing or fix npz_dir."
        )

    return df


def split_index_by_subjects(
    df: pd.DataFrame,
    train_subjects: List[int],
    val_subjects: List[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a normalized index by subject IDs.

    ``train_subjects`` and ``val_subjects`` may contain strings or integers; both
    are coerced to integers.  The dataframe ``subject`` column is also coerced by
    ``load_index``.
    """
    if "subject" not in df.columns:
        raise ValueError("index dataframe is missing 'subject' column")
    train_set = {int(s) for s in train_subjects}
    val_set = {int(s) for s in val_subjects}
    overlap = sorted(train_set & val_set)
    if overlap:
        raise ValueError(f"train_subjects and val_subjects overlap: {overlap}")

    subjects = pd.to_numeric(df["subject"], errors="raise").astype(np.int64)
    tr = df[subjects.isin(train_set)].reset_index(drop=True)
    va = df[subjects.isin(val_set)].reset_index(drop=True)
    return tr, va


def fit_channel_stats(df: pd.DataFrame, max_shards: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std on the training shards."""
    if len(df) == 0:
        raise ValueError("cannot fit channel stats: training dataframe is empty")
    if "abs_shard" not in df.columns:
        raise ValueError("cannot fit channel stats: dataframe is missing 'abs_shard'")

    shards = list(dict.fromkeys(df.abs_shard.tolist()))
    if not shards:
        raise ValueError("cannot fit channel stats: no shards found")
    if max_shards and len(shards) > max_shards:
        rng = np.random.default_rng(42)
        shards = list(rng.choice(shards, size=max_shards, replace=False))

    sums = None
    sqs = None
    count = 0

    for p in shards:
        with np.load(p) as z:
            if "x" not in z.files:
                raise KeyError(f"NPZ shard {p} does not contain array 'x'; keys={z.files}")
            x = z["x"].astype(np.float64)
            if x.ndim != 3:
                raise ValueError(f"NPZ shard {p} expected x shape (N,C,T), got {x.shape}")
            s = x.sum(axis=(0, 2))
            q = (x * x).sum(axis=(0, 2))
            n = x.shape[0] * x.shape[2]
            sums = s if sums is None else sums + s
            sqs = q if sqs is None else sqs + q
            count += n

    if sums is None or sqs is None or count <= 0:
        raise ValueError("cannot fit channel stats: selected shards contain no EEG samples")

    mean = sums / count
    var = np.maximum(sqs / count - mean * mean, 1e-8)
    std = np.sqrt(var)

    return mean.astype(np.float32), std.astype(np.float32)


class WindowNpzDataset(Dataset):
    """EEG window dataset backed by compressed NPZ shards."""

    def __init__(
        self,
        df: pd.DataFrame,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        text_csv_path: str | Path,
        cache_size: int = 8,
    ):
        self.df = normalize_index_dataframe(df, npz_dir=None)

        channel_mean = np.asarray(channel_mean, dtype=np.float32)
        channel_std = np.asarray(channel_std, dtype=np.float32)
        if channel_mean.ndim != 1 or channel_std.ndim != 1 or channel_mean.shape != channel_std.shape:
            raise ValueError(
                f"channel_mean/channel_std must be 1-D arrays with the same shape; "
                f"got {channel_mean.shape} and {channel_std.shape}"
            )
        if np.any(~np.isfinite(channel_mean)) or np.any(~np.isfinite(channel_std)):
            raise ValueError("channel_mean/channel_std contain NaN or Inf")

        self.mean = channel_mean[:, None]
        self.std = channel_std[:, None]

        self.cache_size = int(cache_size)
        if self.cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {cache_size}")
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.text_by_trial = load_l2_text_protocol(text_csv_path)

        self._abs_shard_arr = self.df["abs_shard"].values
        self._idx_arr = self.df["idx"].values
        self._label3_arr = self.df["label3"].values
        self._trial_arr = self.df["trial"].values
        self._subject_arr = self.df["subject"].values

    def __len__(self) -> int:
        return len(self.df)

    def _load_shard(self, path: str) -> dict:
        """Load one NPZ shard through an LRU cache."""
        path_str = str(path)
        if path_str in self.cache:
            self.cache.move_to_end(path_str)
            return self.cache[path_str]

        with np.load(path_str) as z:
            item = {k: z[k] for k in z.files}
        if "x" not in item:
            raise KeyError(f"NPZ shard {path_str} does not contain array 'x'; keys={list(item)}")
        self.cache[path_str] = item

        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return item

    def __getitem__(self, i: int) -> dict:
        shard_path = str(self._abs_shard_arr[i])
        shard_idx = int(self._idx_arr[i])
        y = int(self._label3_arr[i])
        tid = int(self._trial_arr[i])
        subject = int(self._subject_arr[i])

        shard = self._load_shard(shard_path)

        x = shard["x"][shard_idx].astype(np.float32)
        if x.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"channel count mismatch for {shard_path}[{shard_idx}]: "
                f"x has {x.shape[0]} channels but stats have {self.mean.shape[0]}"
            )
        x = (x - self.mean) / (self.std + 1e-6)

        text = self.text_by_trial[tid]

        return {
            "eeg": torch.from_numpy(x).unsqueeze(0),  # (1, 62, T)
            "label": torch.tensor(y, dtype=torch.long),
            "text": text,
            "subject": subject,
            "trial": tid,
        }


class ClassBalancedBatchSampler(Sampler[List[int]]):
    """Class-balanced batch sampler.

    This is a *batch sampler*: pass it to DataLoader as ``batch_sampler=...``
    rather than as ``sampler=...``.
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
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.labels.size == 0:
            raise ValueError("empty labels; cannot build balanced sampler")

        self.classes = sorted(np.unique(self.labels).tolist())
        self.n_cls = len(self.classes)

        self.by_class = {c: np.where(self.labels == c)[0].tolist() for c in self.classes}

        if any(len(v) == 0 for v in self.by_class.values()):
            raise ValueError("empty class in labels; cannot build balanced sampler")

        self.base_per_cls = self.batch_size // self.n_cls
        self.rem = self.batch_size % self.n_cls

        default_steps = math.ceil(len(self.labels) / self.batch_size)
        self.steps_per_epoch = int(steps_per_epoch or default_steps)
        if self.steps_per_epoch <= 0:
            raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
        self.seed = int(seed)
        self.epoch = 0
        self.with_replacement = bool(with_replacement)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)

        for _ in range(self.steps_per_epoch):
            batch = []

            order = self.classes[:]
            rng.shuffle(order)

            for j, c in enumerate(order):
                k = self.base_per_cls + (1 if j < self.rem else 0)
                if k <= 0:
                    continue

                if self.with_replacement:
                    batch.extend(rng.choices(self.by_class[c], k=k))
                else:
                    indices = self.by_class[c]
                    if len(indices) <= k:
                        batch.extend(indices)
                    else:
                        batch.extend(rng.sample(indices, k=k))

            rng.shuffle(batch)
            yield batch


def build_l2_text_bank(text_csv_path: str | Path) -> tuple[list[str], np.ndarray, list[int]]:
    """Build the 80-trial L2 text bank and 3-class labels."""
    text_by_trial = load_l2_text_protocol(text_csv_path)
    texts, labels, trials = [], [], []

    for tid in range(1, 81):
        _, _, y = trial_to_labels(tid)
        texts.append(text_by_trial[tid])
        labels.append(y)
        trials.append(tid)

    return texts, np.asarray(labels, dtype=np.int64), trials
