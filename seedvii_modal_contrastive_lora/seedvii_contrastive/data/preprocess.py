from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import json
import os
import shutil
import tempfile
import zipfile

import numpy as np
from tqdm import tqdm

from .h5io import subject_mat_path, read_trial_array
from .protocol import trial_to_labels, write_label_protocol_csvs
from .windowing import center_crop_signal, make_windows, choose_fixed_windows


def _find_member_for_subject(zf: zipfile.ZipFile, sid: int) -> str:
    suffixes = [f"/{sid}.mat", f"/{sid:02d}.mat", f"{sid}.mat", f"{sid:02d}.mat"]
    names = zf.namelist()
    for name in names:
        clean = name.replace("\\", "/")
        if any(clean.endswith(s) for s in suffixes):
            return name
    raise FileNotFoundError(f"cannot find {sid}.mat in zip; first names={names[:10]}")


def _extract_subject_from_zip(zip_path: str | Path, sid: int, tmp_dir: str | Path) -> Path:
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        member = _find_member_for_subject(zf, sid)
        out_path = tmp_dir / f"{sid}.mat"
        with zf.open(member, "r") as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024 * 16)
    return out_path


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int = 512):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.buf_x: List[np.ndarray] = []
        self.buf_meta: List[dict] = []
        self.shard_id = 0
        self.rows: List[dict] = []

    def add_many(self, x: np.ndarray, metas: List[dict]):
        for i in range(len(x)):
            self.buf_x.append(x[i])
            self.buf_meta.append(metas[i])
            if len(self.buf_x) >= self.shard_size:
                self.flush()

    def flush(self):
        if not self.buf_x:
            return
        shard_name = f"shard_{self.shard_id:06d}.npz"
        shard_path = self.out_dir / shard_name
        x = np.stack(self.buf_x, axis=0).astype(np.float32)
        y = np.asarray([m["label3"] for m in self.buf_meta], dtype=np.int64)
        subjects = np.asarray([m["subject"] for m in self.buf_meta], dtype=np.int16)
        trials = np.asarray([m["trial"] for m in self.buf_meta], dtype=np.int16)
        starts = np.asarray([m["start_sample"] for m in self.buf_meta], dtype=np.int64)
        np.savez_compressed(shard_path, x=x, y=y, subject=subjects, trial=trials, start_sample=starts)
        for idx, m in enumerate(self.buf_meta):
            row = dict(m)
            row.update({"shard": shard_name, "idx": idx})
            self.rows.append(row)
        self.shard_id += 1
        self.buf_x.clear(); self.buf_meta.clear()

    def write_index(self):
        self.flush()
        index_path = self.out_dir / "index.csv"
        fields = ["shard", "idx", "subject", "trial", "fine_emotion", "valence", "label3", "window_id", "start_sample", "n_samples_clip"]
        with open(index_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(self.rows)
        return index_path


def preprocess_to_npz(
    output_dir: str | Path,
    input_root: Optional[str | Path] = None,
    zip_path: Optional[str | Path] = None,
    subjects: List[int] | None = None,
    fs: int = 200,
    window_sec: float = 4.0,
    stride_sec: float = 4.0,
    center_ratio: float = 0.60,
    max_windows_per_clip: int = 12,
    shard_size: int = 512,
    tmp_dir: Optional[str | Path] = None,
) -> Path:
    """Preprocess SEED-VII EEG_preprocessed H5 .mat files to window-level NPZ shards.

    This extracts/opens one subject at a time, crops the middle 60%, cuts 4s windows,
    optionally caps each clip to a fixed number of windows, and writes small .npz shards.
    No train/test statistics are fitted here; normalization is fitted later on training
    subjects only to avoid preprocessing leakage.
    """
    if not input_root and not zip_path:
        raise ValueError("provide either input_root or zip_path")
    subjects = subjects or list(range(1, 21))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_label_protocol_csvs(out_dir / "protocol")
    writer = ShardWriter(out_dir, shard_size=shard_size)
    tmp_base = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="seedvii_extract_"))
    tmp_base.mkdir(parents=True, exist_ok=True)

    summary = {"subjects": subjects, "fs": fs, "window_sec": window_sec, "stride_sec": stride_sec,
               "center_ratio": center_ratio, "max_windows_per_clip": max_windows_per_clip}
    try:
        for sid in tqdm(subjects, desc="subjects"):
            extracted = False
            if zip_path:
                mat_path = _extract_subject_from_zip(zip_path, sid, tmp_base)
                extracted = True
            else:
                mat_path = subject_mat_path(input_root, sid)
            for tid in tqdm(range(1, 81), desc=f"S{sid:02d} trials", leave=False):
                arr = read_trial_array(mat_path, tid)  # (62, N)
                cropped, crop_start = center_crop_signal(arr, center_ratio=center_ratio)
                windows, starts = make_windows(cropped, fs=fs, window_sec=window_sec, stride_sec=stride_sec)
                windows, starts = choose_fixed_windows(windows, starts, max_windows_per_clip, seed=sid * 1000 + tid)
                fine, val, y3 = trial_to_labels(tid)
                metas = []
                for wi, st in enumerate(starts):
                    metas.append({
                        "subject": sid, "trial": tid, "fine_emotion": fine, "valence": val,
                        "label3": y3, "window_id": wi, "start_sample": int(crop_start + st),
                        "n_samples_clip": int(arr.shape[1]),
                    })
                if len(windows):
                    writer.add_many(windows, metas)
            if extracted:
                try:
                    os.remove(mat_path)
                except OSError:
                    pass
        index_path = writer.write_index()
        summary["n_windows"] = len(writer.rows)
        with open(out_dir / "preprocess_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return index_path
    finally:
        # Only remove temp dir if we created it implicitly.
        if tmp_dir is None:
            shutil.rmtree(tmp_base, ignore_errors=True)
