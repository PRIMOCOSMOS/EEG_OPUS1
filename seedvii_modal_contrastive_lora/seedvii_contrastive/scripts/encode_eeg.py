from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seedvii_contrastive.utils import load_yaml, resolve_device
from seedvii_contrastive.data.dataset import load_index, split_index_by_subjects, WindowNpzDataset, build_l2_text_bank
from seedvii_contrastive.scripts.train_contrastive import (
    build_models, load_ckpt, validate_training_split,
    _dataloader_worker_kwargs, _is_cuda_device,
)


def collate_fn(batch):
    return {
        "eeg": torch.stack([b["eeg"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "text": [b["text"] for b in batch],
        "subject": [b["subject"] for b in batch],
        "trial": [b["trial"] for b in batch],
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Run trained EEG encoder and save embeddings/predictions")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["train", "val", "all"], default="val")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    
    cfg = load_yaml(args.config)
    device = resolve_device(cfg["runtime"].get("device", "auto"))
    
    # Build models.  Encoding/inference should not put the LLM tower in LoRA
    # training mode; the checkpoint still restores the trained LoRA weights.
    eeg, text = build_models(cfg, device, train_llm=False)
    load_ckpt(args.checkpoint, eeg, text, device=device)
    eeg.eval()
    text.eval()

    # Load data
    df = load_index(cfg["data"]["npz_dir"])
    tr_df, va_df = split_index_by_subjects(df, cfg["data"]["train_subjects"], cfg["data"]["val_subjects"])
    validate_training_split(df, tr_df, va_df, cfg["data"]["train_subjects"], cfg["data"]["val_subjects"])
    use_df = {"train": tr_df, "val": va_df, "all": df}[args.split].reset_index(drop=True)
    if len(use_df) == 0:
        raise ValueError(f"selected split {args.split!r} has zero windows")
    
    stats_path = Path(cfg["runtime"]["output_dir"]) / "norm_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"normalization stats not found: {stats_path}; run training first")
    stats = np.load(stats_path)
    ds = WindowNpzDataset(
        use_df, stats["mean"], stats["std"],
        text_csv_path=cfg["data"]["text_csv_path"],
        cache_size=cfg["data"].get("cache_size", 8)
    )
    num_workers = cfg["train"].get("num_workers", 2)
    dl = DataLoader(
        ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=_is_cuda_device(device),
        **_dataloader_worker_kwargs(num_workers, cfg["train"].get("prefetch_factor", 2)),
    )
    
    # Build text embeddings
    bank_texts, bank_labels, _ = build_l2_text_bank(cfg["data"]["text_csv_path"])
    bank_z = text(bank_texts).float()  # Ensure float32
    protos = []
    for c in range(3):
        idx = torch.tensor(bank_labels == c, dtype=torch.bool, device=bank_z.device)
        proto = bank_z[idx].mean(dim=0)
        protos.append(torch.nn.functional.normalize(proto, dim=0))
    class_z = torch.stack(protos, dim=0)
    
    # Encode
    embs, preds, labels, subjects, trials = [], [], [], [], []
    for b in tqdm(dl, desc="encoding"):
        x = b["eeg"].to(device)
        z = eeg(x).float()  # Ensure float32
        pred = (z @ class_z.t()).argmax(dim=1)
        
        embs.append(z.cpu().numpy())
        preds.extend(pred.cpu().numpy().tolist())
        labels.extend(b["label"].numpy().tolist())
        subjects.extend(b["subject"])
        trials.extend(b["trial"])
    
    if not embs:
        raise RuntimeError(f"no batches were encoded for split {args.split!r}")

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        embedding=np.concatenate(embs, axis=0),
        pred=np.asarray(preds),
        label=np.asarray(labels),
        subject=np.asarray(subjects),
        trial=np.asarray(trials)
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
