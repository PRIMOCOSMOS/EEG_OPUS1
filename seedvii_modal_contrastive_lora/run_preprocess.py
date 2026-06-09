#!/usr/bin/env python
"""
一键 EEG 预处理（独立脚本，进程内运行，不依赖 torch）。

为什么用它：算力平台 Notebook 里 `!python -m ...` 子进程的 stdout 是块缓冲，
进度条出现后看起来"假死"，报错 traceback 也会被吞掉。本脚本直接调用预处理
函数，进度实时刷新，出错立刻抛出完整 traceback。

用法（你的真实情况：经典 v5 .mat，字段 '1'..'80'，50GB 内存够用）:

    cd seedvii_modal_contrastive_lora
    python run_preprocess.py \
        --input-root /path/to/dir_with_1-20.mat \
        --output-dir /path/to/seedvii_npz

可选：先转成 HDF5 再预处理（内存很小时才需要）:

    python run_preprocess.py \
        --input-root /path/to/classic_mats \
        --output-dir /path/to/seedvii_npz \
        --convert-hdf5 /path/to/seedvii_hdf5 \
        --key-mode numeric
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让本仓库可导入（无论是否 pip install -e）
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


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


def main():
    ap = argparse.ArgumentParser(description="一键 EEG -> NPZ 预处理（进程内、无 torch 依赖）")
    ap.add_argument("--input-root", required=True, help="包含 1-20.mat 的目录")
    ap.add_argument("--output-dir", required=True, help="NPZ 输出目录")
    ap.add_argument("--subjects", default="1-20")
    ap.add_argument("--fs", type=int, default=200)
    ap.add_argument("--window-sec", type=float, default=4.0)
    ap.add_argument("--stride-sec", type=float, default=4.0)
    ap.add_argument("--center-ratio", type=float, default=0.60)
    ap.add_argument("--max-windows-per-clip", type=int, default=12)
    ap.add_argument("--shard-size", type=int, default=512)
    # 可选转换
    ap.add_argument("--convert-hdf5", default=None,
                    help="若给定目录，则先把经典 .mat 转成 HDF5 到该目录，再用它预处理")
    ap.add_argument("--key-mode", default="numeric", choices=["numeric", "prefix", "order"])
    args = ap.parse_args()

    subjects = parse_subjects(args.subjects)
    input_root = Path(args.input_root)

    if args.convert_hdf5:
        from seedvii_contrastive.scripts.convert_classic_mat_to_hdf5 import convert_subject

        conv_dir = Path(args.convert_hdf5)
        conv_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Convert] -> {conv_dir} (key-mode={args.key_mode})", flush=True)
        for sid in subjects:
            src = input_root / f"{sid}.mat"
            if not src.exists():
                src = input_root / f"{sid:02d}.mat"
            if not src.exists():
                print(f"[SKIP] {sid}.mat not found", file=sys.stderr)
                continue
            dst = conv_dir / f"{sid}.mat"
            good = dst.exists() and dst.stat().st_size > 1024 and open(dst, "rb").read(8) == b"\x89HDF\r\n\x1a\n"
            if good:
                print(f"[SKIP] {dst.name} already valid HDF5")
                continue
            convert_subject(src, dst, args.key_mode)
        input_root = conv_dir

    from seedvii_contrastive.data.preprocess import preprocess_to_npz

    print(f"[NPZ] input_root={input_root}", flush=True)
    print(f"[NPZ] output_dir={args.output_dir}", flush=True)
    index = preprocess_to_npz(
        output_dir=args.output_dir,
        input_root=str(input_root),
        subjects=subjects,
        fs=args.fs,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        center_ratio=args.center_ratio,
        max_windows_per_clip=args.max_windows_per_clip,
        shard_size=args.shard_size,
    )
    out = Path(args.output_dir)
    n = len(list(out.glob("shard_*.npz")))
    print(f"[OK] index={index}  shards={n}")


if __name__ == "__main__":
    main()
