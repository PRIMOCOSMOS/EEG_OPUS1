#!/usr/bin/env python
"""
One-time converter: Classic MATLAB v5/v7 .mat → MATLAB v7.3 HDF5 format.

This allows the downstream preprocess_npz pipeline to use lazy (memory-efficient)
reading via h5py instead of loading the entire file into RAM with loadmat.

Usage:
    python -m seedvii_contrastive.scripts.convert_classic_mat_to_hdf5 \
        --input-dir /path/to/classic_mats \
        --output-dir /path/to/hdf5_mats \
        --subjects 1-20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from tqdm import tqdm


def parse_subjects(s: str):
    if not s:
        return list(range(1, 21))
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = map(int, part.split("-"))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def convert_subject(src_path: Path, dst_path: Path):
    """Convert one classic .mat to v7.3 HDF5 format."""
    print(f"[Convert] Loading {src_path.name} ...", flush=True)
    try:
        data = loadmat(str(src_path))
    except Exception as e:
        print(f"[ERROR] Failed to load {src_path}: {e}", file=sys.stderr)
        return False

    # Keep only the actual trial variables (numeric keys)
    trial_data = {k: v for k, v in data.items() if k.isdigit() or (k.startswith("-") and k[1:].isdigit())}

    if not trial_data:
        print(f"[WARN] No numeric trial variables found in {src_path.name}", file=sys.stderr)
        return False

    print(f"[Convert] Writing {len(trial_data)} trials to {dst_path.name} (v7.3 HDF5)...", flush=True)
    try:
        savemat(str(dst_path), trial_data, format="7.3", do_compression=True)
        print(f"[OK] Converted {src_path.name} → {dst_path.name}", flush=True)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save {dst_path}: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description="Convert classic SEED-VII .mat files to memory-efficient v7.3 HDF5")
    ap.add_argument("--input-dir", required=True, help="Directory containing classic 1.mat, 2.mat, ...")
    ap.add_argument("--output-dir", required=True, help="Directory to write converted HDF5 .mat files")
    ap.add_argument("--subjects", default="1-20", help="Subjects to convert, e.g. 1-20 or 1,3,5")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = parse_subjects(args.subjects)

    success = 0
    for sid in tqdm(subjects, desc="Converting subjects"):
        src = input_dir / f"{sid}.mat"
        if not src.exists():
            src = input_dir / f"{sid:02d}.mat"
        if not src.exists():
            print(f"[SKIP] {sid}.mat not found in {input_dir}", file=sys.stderr)
            continue

        dst = output_dir / f"{sid}.mat"
        if convert_subject(src, dst):
            success += 1

    print(f"\n[Done] Successfully converted {success}/{len(subjects)} subjects.")
    print(f"Converted files are in: {output_dir}")
    print("You can now point --input-root to the output directory for preprocessing.")


if __name__ == "__main__":
    main()