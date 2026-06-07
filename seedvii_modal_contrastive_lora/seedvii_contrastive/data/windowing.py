from __future__ import annotations

from typing import Tuple
import numpy as np


def center_crop_signal(x: np.ndarray, center_ratio: float = 0.60) -> Tuple[np.ndarray, int]:
    """Crop the middle center_ratio portion of a (C,T) signal.

    Returns cropped signal and its start index in the original clip.
    """
    assert x.ndim == 2, x.shape
    n = x.shape[1]
    crop_n = int(round(n * center_ratio))
    crop_n = max(1, min(n, crop_n))
    start = (n - crop_n) // 2
    end = start + crop_n
    return x[:, start:end], start


def make_windows(
    x: np.ndarray,
    fs: int = 200,
    window_sec: float = 4.0,
    stride_sec: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Non-overlap by default: (62,T) -> (N,62,window_samples), start_samples."""
    win = int(round(fs * window_sec))
    stride = int(round(fs * stride_sec))
    if x.shape[1] < win:
        return np.empty((0, x.shape[0], win), dtype=np.float32), np.empty((0,), dtype=np.int64)
    starts = np.arange(0, x.shape[1] - win + 1, stride, dtype=np.int64)
    out = np.stack([x[:, s:s + win] for s in starts], axis=0).astype(np.float32)
    return out, starts


def choose_fixed_windows(windows: np.ndarray, starts: np.ndarray, max_windows: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample at most max_windows windows per clip to prevent long clips dominating."""
    if max_windows <= 0 or len(windows) <= max_windows:
        return windows, starts
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(windows), size=max_windows, replace=False))
    return windows[idx], starts[idx]
