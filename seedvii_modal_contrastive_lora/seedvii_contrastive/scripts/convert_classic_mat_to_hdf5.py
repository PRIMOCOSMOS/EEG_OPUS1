#!/usr/bin/env python
"""
One-time converter: Classic MATLAB v5/v7 .mat -> MATLAB v7.3 (HDF5) format.

This allows the downstream preprocess_npz pipeline to use lazy (memory-efficient)
reading via h5py instead of loading the entire file into RAM with loadmat.

------------------------------------------------------------------------------
FIX HISTORY
------------------------------------------------------------------------------
The original version used ``scipy.io.savemat(path, data, format="7.3")``.
scipy's ``savemat`` ONLY supports ``format="4"`` or ``format="5"`` and raises
``ValueError: Format should be '4' or '5'`` for "7.3".  Worse, ``savemat`` had
already *created/truncated* the destination file before failing, leaving a
ZERO-byte ``N.mat`` behind.  The exception was swallowed (``return False``), so:

  * no real HDF5 files were ever produced;
  * the 0-byte files made the notebook's ``if not (DIR/'1.mat').exists()`` guard
    *skip* conversion on the next run;
  * the pipeline then pointed ``EEG_ROOT`` at a folder full of empty files and
    died on subject 1 (the traceback hidden by ``!python`` subprocess buffering),
    showing up as "the NPZ cell hangs after the progress bar".

This rewrite uses ``h5py`` to write a genuine HDF5 (v7.3-compatible) file that the
repo's own ``SubjectMatReader`` reads via its h5py branch.  Writes are atomic
(temp file + rename) so an interrupted/failed run never leaves a 0-byte .mat.

Usage:
    python -m seedvii_contrastive.scripts.convert_classic_mat_to_hdf5 \
        --input-dir /path/to/classic_mats \
        --output-dir /path/to/hdf5_mats \
        --subjects 1-20 \
        --key-mode numeric        # numeric | prefix | order

key-mode:
    numeric : variable names are already '1'..'80'  (SEED-VII default; your case)
    prefix  : names like 'xx_eeg1','djc_eeg2' -> sort by trailing int -> '1'..'N'
    order   : rename strictly by appearance order -> '1'..'N' (last-resort)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

try:
    import h5py
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "h5py is required for HDF5 conversion. Install with: pip install h5py\n"
        f"(import error: {e})"
    )


def parse_subjects(s: str):
    if not s:
        return list(range(1, 21))
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-"))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def _trailing_int(name: str):
    m = re.search(r"(\d+)\s*$", str(name))
    return int(m.group(1)) if m else None


def _collect_trials(data: dict, key_mode: str, n_channels: int = 62) -> dict:
    """Return {'1': ndarray(62,T), ...} with clean sequential numeric keys."""
    candidates = []
    for k, v in data.items():
        if k.startswith("__"):
            continue
        a = np.squeeze(np.asarray(v))
        if a.ndim == 2 and a.size > 1000:
            candidates.append((k, a))

    if not candidates:
        raise ValueError("no 2-D numeric trial arrays found in this .mat")

    if key_mode == "numeric":
        pairs = [(int(k), a) for k, a in candidates if re.fullmatch(r"\d+", str(k))]
        if not pairs:
            raise ValueError(
                "key-mode=numeric but no purely-numeric variable names found; "
                "try --key-mode prefix or --key-mode order"
            )
        pairs.sort(key=lambda x: x[0])
    elif key_mode == "prefix":
        pairs = [(_trailing_int(k), a) for k, a in candidates if _trailing_int(k) is not None]
        if not pairs:
            raise ValueError(
                "key-mode=prefix but no variable names ending in a number; "
                "try --key-mode order"
            )
        pairs.sort(key=lambda x: x[0])
    elif key_mode == "order":
        pairs = [(i + 1, a) for i, (_, a) in enumerate(candidates)]
    else:
        raise ValueError(f"unknown key-mode={key_mode}")

    out = {}
    for new_id, (_, a) in enumerate(pairs, start=1):
        # orient to (62, T) like the reader expects
        if a.shape[0] != n_channels and a.shape[1] == n_channels:
            a = a.T
        out[str(new_id)] = np.ascontiguousarray(a, dtype=np.float32)
    return out


def convert_subject(src_path: Path, dst_path: Path, key_mode: str, n_channels: int = 62) -> bool:
    print(f"[Convert] Loading {src_path.name} (loadmat -> RAM, ~1-2 GB/subject)...", flush=True)
    try:
        data = loadmat(str(src_path))
    except Exception as e:
        print(f"[ERROR] Failed to load {src_path}: {e}", file=sys.stderr)
        return False

    try:
        trials = _collect_trials(data, key_mode, n_channels=n_channels)
    except Exception as e:
        print(f"[ERROR] {src_path.name}: {e}", file=sys.stderr)
        return False

    n = len(trials)
    print(f"[Convert] Writing {n} trials to {dst_path.name} (real HDF5, gzip-4)...", flush=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    try:
        with h5py.File(str(tmp_path), "w") as f:
            for k, a in trials.items():
                f.create_dataset(k, data=a, compression="gzip", compression_opts=4)
        # atomic publish: only a fully-written file ever appears at dst_path
        os.replace(tmp_path, dst_path)
    except Exception as e:
        print(f"[ERROR] Failed to write {dst_path}: {e}", file=sys.stderr)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return False

    ok = open(dst_path, "rb").read(8) == b"\x89HDF\r\n\x1a\n"
    print(f"[OK] {src_path.name} -> {dst_path.name}  trials={n}  hdf5_header={ok}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Convert classic SEED-VII .mat -> memory-efficient v7.3 HDF5")
    ap.add_argument("--input-dir", required=True, help="Directory containing classic 1.mat, 2.mat, ...")
    ap.add_argument("--output-dir", required=True, help="Directory to write converted HDF5 .mat files")
    ap.add_argument("--subjects", default="1-20", help="Subjects to convert, e.g. 1-20 or 1,3,5")
    ap.add_argument("--key-mode", default="numeric", choices=["numeric", "prefix", "order"],
                    help="How to derive trial keys from variable names")
    ap.add_argument("--n-channels", type=int, default=62)
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
        # treat a 0-byte / non-HDF5 leftover as "needs conversion"
        if dst.exists():
            try:
                good = dst.stat().st_size > 1024 and open(dst, "rb").read(8) == b"\x89HDF\r\n\x1a\n"
            except OSError:
                good = False
            if good:
                print(f"[SKIP] {dst.name} already a valid HDF5", flush=True)
                success += 1
                continue
            else:
                dst.unlink(missing_ok=True)  # remove broken 0-byte leftover

        if convert_subject(src, dst, args.key_mode, n_channels=args.n_channels):
            success += 1

    print(f"\n[Done] Successfully converted {success}/{len(subjects)} subjects.")
    print(f"Converted files are in: {output_dir}")
    print("You can now point --input-root to the output directory for preprocessing.")


if __name__ == "__main__":
    main()
