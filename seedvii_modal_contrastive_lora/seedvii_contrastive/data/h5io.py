from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import re
import h5py
import numpy as np


def subject_mat_path(root: str | Path, subject_id: int) -> Path:
    root = Path(root)
    candidates = [root / f"{subject_id}.mat", root / f"{subject_id:02d}.mat"]
    for p in candidates:
        if p.exists():
            return p
    hits = sorted(root.rglob(f"{subject_id}.mat")) + sorted(root.rglob(f"{subject_id:02d}.mat"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"cannot find subject {subject_id}.mat under {root}")


def list_trial_keys(mat_path: str | Path) -> List[str]:
    with h5py.File(mat_path, "r") as f:
        keys = [k for k in f.keys() if re.fullmatch(r"\d+", str(k))]
    return sorted(keys, key=lambda x: int(x))


def read_trial_array(mat_path: str | Path, trial_id: int, n_channels: int = 62) -> np.ndarray:
    """Read one HDF5 MATLAB dataset and return float32 array shaped (62, n_samples).

    The provided EEG_preprocessed files store keys '1'...'80', each usually 62×N.
    Some HDF5 MATLAB exports may appear transposed, so this function repairs it.
    """
    key = str(int(trial_id))
    with h5py.File(mat_path, "r") as f:
        if key not in f:
            raise KeyError(f"{mat_path} has no trial key {key}; available keys example={list(f.keys())[:10]}")
        arr = np.asarray(f[key], dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"trial {key} in {mat_path} must be 2-D, got {arr.shape}")
    if arr.shape[0] == n_channels:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.shape[1] == n_channels:
        return np.ascontiguousarray(arr.T, dtype=np.float32)
    raise ValueError(f"trial {key} in {mat_path} expected one dimension={n_channels}, got {arr.shape}")
