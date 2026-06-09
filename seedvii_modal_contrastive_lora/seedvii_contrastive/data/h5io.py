from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any
import re
import os

import numpy as np

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
MATLAB_V5_PREFIX = b"MATLAB 5.0 MAT-file"


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


def sniff_mat_file(path: str | Path, n: int = 512) -> dict:
    """Return lightweight file-signature diagnostics for .mat files."""
    p = Path(path)
    info = {"path": str(p), "exists": p.exists(), "size": None, "kind": "missing", "head": b""}
    if not p.exists():
        return info
    info["size"] = p.stat().st_size
    with open(p, "rb") as f:
        head = f.read(n)
    info["head"] = head
    if head.startswith(HDF5_SIGNATURE):
        info["kind"] = "matlab_v7.3_hdf5"
    elif head.startswith(MATLAB_V5_PREFIX):
        info["kind"] = "matlab_v5_v7_classic"
    elif head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        info["kind"] = "git_lfs_pointer"
    elif head.startswith(b"PK\x03\x04"):
        info["kind"] = "zip_archive"
    elif head.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        info["kind"] = "html_or_error_page"
    else:
        info["kind"] = "unknown"

    # size sanity check for classic .mat files
    if info["size"] is not None and info["size"] < 1024 * 10:
        # Under 10KB — likely Git LFS pointer or truncated download
        info["_tiny"] = True
    else:
        info["_tiny"] = False
    return info


def _format_head(head: bytes, max_len: int = 64) -> str:
    if not head:
        return ""
    shown = head[:max_len]
    try:
        txt = shown.decode("utf-8", errors="replace")
        if any(c.isprintable() for c in txt):
            return repr(txt)
    except Exception:
        pass
    return repr(shown)


def explain_bad_mat_file(path: str | Path) -> str:
    info = sniff_mat_file(path)
    msg = [
        f"Unsupported or corrupted .mat file: {info['path']}",
        f"exists={info['exists']} size={info['size']} kind={info['kind']}",
        f"first_bytes={_format_head(info.get('head', b''))}",
    ]
    kind = info["kind"]
    if kind == "git_lfs_pointer":
        msg.append("This is a Git-LFS pointer, not the real EEG .mat payload. Download with ModelScope dataset/LFS support.")
    elif kind == "html_or_error_page":
        msg.append("This looks like an HTML/error page, often caused by an authentication or wrong URL download.")
    elif kind == "zip_archive":
        msg.append("This is a ZIP archive. Extract it first.")
    elif kind == "unknown":
        msg.append("Expected either HDF5 MATLAB v7.3 signature or MATLAB 5/v7 classic header.")
    return "\n".join(msg)


def _normalize_trial_array(arr: np.ndarray, trial_id: int, mat_path: str | Path, n_channels: int = 62) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"trial {trial_id} in {mat_path} must be 2-D, got {arr.shape}")
    if arr.shape[0] == n_channels:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.shape[1] == n_channels:
        return np.ascontiguousarray(arr.T, dtype=np.float32)
    raise ValueError(f"trial {trial_id} in {mat_path} expected one dimension={n_channels}, got {arr.shape}")


class SubjectMatReader:
    """Read one SEED-VII subject .mat robustly."""

    def __init__(self, mat_path: str | Path, n_channels: int = 62):
        self.mat_path = Path(mat_path)
        self.n_channels = n_channels
        self.kind = sniff_mat_file(self.mat_path)["kind"]
        self._h5 = None
        self._mat: Optional[Dict[str, Any]] = None

    def __enter__(self) -> "SubjectMatReader":
        if self.kind == "matlab_v7.3_hdf5":
            try:
                print(f"  [h5py] {self.mat_path.name} opening (HDF5 format, lazy read)...", flush=True)
                import h5py
                self._h5 = h5py.File(self.mat_path, "r")
            except Exception as e:
                raise OSError(f"Failed to open HDF5 MAT file {self.mat_path}: {e}\n{explain_bad_mat_file(self.mat_path)}") from e
        elif self.kind == "matlab_v5_v7_classic":
            size_bytes = self.mat_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            if size_bytes < 1024 * 20:
                info = sniff_mat_file(self.mat_path)
                msg = (
                    f"\n{'='*60}\n"
                    f"  FATAL: {self.mat_path.name} is only {size_mb:.1f} MB\n"
                    f"  This .mat file is too small to contain real EEG data.\n"
                    f"  Likely cause: Git LFS pointer or truncated download.\n"
                    f"  Fix: re-run ModelScope download with LFS support.\n"
                    f"  First bytes: {_format_head(info.get('head', b''), 200)}\n"
                    f"{'='*60}"
                )
                raise OSError(msg)
            try:
                from scipy.io import loadmat
                print(f"  [loadmat] {self.mat_path.name} ({size_mb:.0f} MB classic format)"
                      f" — loading entire file into RAM, please wait...", flush=True)
                self._mat = loadmat(self.mat_path)
                print(f"  [loadmat] {self.mat_path.name} loaded.", flush=True)
            except Exception as e:
                raise OSError(f"Failed to open classic MATLAB MAT file {self.mat_path}: {e}\n{explain_bad_mat_file(self.mat_path)}") from e
        else:
            raise OSError(explain_bad_mat_file(self.mat_path))
        return self
    def __exit__(self, exc_type, exc, tb):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        self._mat = None

    def trial_keys(self) -> List[str]:
        if self._h5 is not None:
            keys = [k for k in self._h5.keys() if re.fullmatch(r"\d+", str(k))]
            return sorted(keys, key=lambda x: int(x))
        if self._mat is not None:
            keys = [k for k in self._mat.keys() if re.fullmatch(r"\d+", str(k))]
            return sorted(keys, key=lambda x: int(x))
        raise RuntimeError("SubjectMatReader is not opened")

    def read_trial(self, trial_id: int) -> np.ndarray:
        key = str(int(trial_id))
        if self._h5 is not None:
            if key not in self._h5:
                raise KeyError(f"{self.mat_path} has no trial key {key}; available keys example={list(self._h5.keys())[:10]}")
            arr = np.asarray(self._h5[key], dtype=np.float32)
            return _normalize_trial_array(arr, trial_id, self.mat_path, self.n_channels)
        if self._mat is not None:
            if key not in self._mat:
                public = [k for k in self._mat.keys() if not k.startswith("__")]
                raise KeyError(f"{self.mat_path} has no trial key {key}; available keys example={public[:10]}")
            return _normalize_trial_array(self._mat[key], trial_id, self.mat_path, self.n_channels)
        raise RuntimeError("SubjectMatReader is not opened")


def list_trial_keys(mat_path: str | Path) -> List[str]:
    with SubjectMatReader(mat_path) as r:
        return r.trial_keys()


def read_trial_array(mat_path: str | Path, trial_id: int, n_channels: int = 62) -> np.ndarray:
    """Read one trial and return float32 array shaped (62, n_samples)."""
    with SubjectMatReader(mat_path, n_channels=n_channels) as r:
        return r.read_trial(trial_id)
