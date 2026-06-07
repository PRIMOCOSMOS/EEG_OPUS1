from __future__ import annotations

import argparse
from pathlib import Path

from seedvii_contrastive.data.preprocess import preprocess_to_npz


def parse_subjects(s: str):
    if not s:
        return list(range(1, 21))
    out = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = map(int, part.split('-'))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description="Preprocess SEED-VII EEG_preprocessed H5 .mat to NPZ shards")
    ap.add_argument("--input-root", default=None, help="Directory containing 1-20.mat, if already available")
    ap.add_argument("--zip-path", default=None, help="Combined zip path; extracts one subject .mat at a time")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--subjects", default="1-20")
    ap.add_argument("--fs", type=int, default=200)
    ap.add_argument("--window-sec", type=float, default=4.0)
    ap.add_argument("--stride-sec", type=float, default=4.0)
    ap.add_argument("--center-ratio", type=float, default=0.60)
    ap.add_argument("--max-windows-per-clip", type=int, default=12)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--tmp-dir", default=None)
    args = ap.parse_args()
    index = preprocess_to_npz(
        output_dir=args.output_dir,
        input_root=args.input_root,
        zip_path=args.zip_path,
        subjects=parse_subjects(args.subjects),
        fs=args.fs,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        center_ratio=args.center_ratio,
        max_windows_per_clip=args.max_windows_per_clip,
        shard_size=args.shard_size,
        tmp_dir=args.tmp_dir,
    )
    print(f"[OK] wrote index: {index}")


if __name__ == "__main__":
    main()
