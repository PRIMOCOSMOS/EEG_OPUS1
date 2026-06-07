from __future__ import annotations

# Allow running this file directly, e.g.
# python seedvii_contrastive/scripts/xxx.py
# without requiring `pip install -e .` or setting PYTHONPATH.
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path
import time
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seedvii_contrastive.data.dataset import (
    load_index, split_index_by_subjects, fit_channel_stats,
    WindowNpzDataset, ClassBalancedBatchSampler, build_l2_text_bank,
)
from seedvii_contrastive.data.protocol import VALENCE_NAMES
from seedvii_contrastive.models.eegnet import EEGNetEncoder
from seedvii_contrastive.models.llm_tower import LoRATextTower
from seedvii_contrastive.losses import TriContrastiveLoss
from seedvii_contrastive.metrics import accuracy_macro_f1
from seedvii_contrastive.utils import load_yaml, save_json, set_seed, resolve_device, now
from seedvii_contrastive.data.discovery import discover_seedvii_paths


def collate(batch):
    return {
        "eeg": torch.stack([b["eeg"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "text": [b["text"] for b in batch],
        "subject": [b["subject"] for b in batch],
        "trial": [b["trial"] for b in batch],
    }


def _looks_like_transformers_model_dir(path: Path) -> bool:
    return path.exists() and (path / "config.json").exists()


def _find_local_llm_dir(base: Path, preferred_name: str = "") -> Path | None:
    """Find an actual Transformers model directory under a ModelScope cache/local dir."""
    if not base.exists():
        return None
    candidates = [p.parent for p in base.rglob("config.json") if p.is_file()]
    if not candidates:
        return None
    preferred_name = preferred_name.lower()
    if preferred_name:
        hits = [c for c in candidates if preferred_name in str(c).lower()]
        if hits:
            return sorted(hits, key=lambda x: len(str(x)))[0]
    return sorted(candidates, key=lambda x: len(str(x)))[0]


def resolve_llm_model_path(llm_cfg: dict) -> str:
    """Resolve local/ModelScope LLM path robustly."""
    raw = str(llm_cfg["model_name_or_path"])
    p = Path(raw).expanduser()
    if _looks_like_transformers_model_dir(p):
        return str(p)

    if p.is_absolute() or raw.startswith(".") or raw.startswith("~"):
        search_roots = []
        if p.exists():
            search_roots.append(p)
        search_roots.append(p.parent)
        for root in search_roots:
            found = _find_local_llm_dir(root, preferred_name=p.name)
            if found is not None:
                print(f"[LLM Tower] resolved local model path: {raw} -> {found}")
                return str(found)

    model_id = llm_cfg.get("modelscope_model_id") or llm_cfg.get("hf_model_id") or "Qwen/Qwen2.5-0.5B-Instruct"
    cache_dir = str(p.parent if p.parent != Path("") else Path("/mnt/workspace/models"))
    print(f"[LLM Tower][WARN] local model path does not exist or lacks config.json: {raw}")
    print(f"[LLM Tower] downloading ModelScope model {model_id} to cache_dir={cache_dir}")
    try:
        from modelscope import snapshot_download
        model_dir = snapshot_download(model_id, cache_dir=cache_dir)
        print(f"[LLM Tower] downloaded/resolved model_dir={model_dir}")
        return str(model_dir)
    except Exception as e:
        raise FileNotFoundError(
            f"Cannot resolve local LLM path {raw!r}, and ModelScope download of {model_id!r} failed: {e}. "
            f"Fix config model.llm.model_name_or_path to the actual snapshot_download return path, "
            f"or set model.llm.modelscope_model_id."
        ) from e

    return raw


def build_models(cfg, device):
    mcfg = cfg["model"]
    eeg = EEGNetEncoder(embed_dim=mcfg["embed_dim"], **mcfg["eegnet"]).to(device)
    llm_cfg = mcfg["llm"]
    llm_path = resolve_llm_model_path(llm_cfg)
    text = LoRATextTower(
        model_name_or_path=llm_path,
        embed_dim=mcfg["embed_dim"],
        lora_r=llm_cfg.get("lora_r", 8),
        lora_alpha=llm_cfg.get("lora_alpha", 16),
        lora_dropout=llm_cfg.get("lora_dropout", 0.05),
        target_modules=llm_cfg.get("target_modules"),
        max_length=llm_cfg.get("max_length", 64),
        gradient_checkpointing=llm_cfg.get("gradient_checkpointing", False),
    ).to(device)
    print("[LLM Tower]", text.trainable_parameters_report())
    return eeg, text


def build_optimizer(cfg, eeg, text):
    tcfg = cfg["train"]
    params = [
        {"params": [p for p in eeg.parameters() if p.requires_grad], "lr": tcfg["lr_eeg"]},
        {"params": [p for n, p in text.named_parameters() if p.requires_grad and not n.startswith("proj")], "lr": tcfg["lr_llm"]},
        {"params": list(text.proj.parameters()), "lr": tcfg["lr_proj"]},
    ]
    return torch.optim.AdamW(params, weight_decay=tcfg.get("weight_decay", 1e-5))


@torch.no_grad()
def evaluate(eeg, text, loader, device, text_csv_path):
    """Classify EEG by nearest aggregated class prototype in L2-text embedding space."""
    eeg.eval(); text.eval()
    bank_texts, bank_labels, _ = build_l2_text_bank(text_csv_path)
    bank_z = text(bank_texts)
    protos = []
    for c in range(3):
        idx = torch.tensor(bank_labels == c, dtype=torch.bool, device=bank_z.device)
        proto = bank_z[idx].mean(dim=0)
        protos.append(torch.nn.functional.normalize(proto, dim=0))
    class_z = torch.stack(protos, dim=0)
    ys, preds = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["eeg"].to(device)
        y = batch["label"].to(device)
        z = eeg(x)
        logits = z @ class_z.t()
        pred = logits.argmax(dim=1)
        ys.extend(y.cpu().numpy().tolist())
        preds.extend(pred.cpu().numpy().tolist())
    return accuracy_macro_f1(ys, preds, num_classes=3)


def save_ckpt(path, eeg, text, opt, sched, epoch, step, best_metric, cfg):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "eeg": eeg.state_dict(),
        "text": text.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched else None,
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "cfg": cfg,
    }, path)


def load_ckpt(path, eeg, text, opt=None, sched=None, device="cpu"):
    sd = torch.load(path, map_location=device, weights_only=False)
    eeg.load_state_dict(sd["eeg"])
    text.load_state_dict(sd["text"], strict=False)
    if opt is not None and sd.get("opt") is not None:
        opt.load_state_dict(sd["opt"])
    if sched is not None and sd.get("sched") is not None:
        sched.load_state_dict(sd["sched"])
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--npz-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.npz_dir:
        cfg["data"]["npz_dir"] = args.npz_dir
    if args.output_dir:
        cfg["runtime"]["output_dir"] = args.output_dir

    # Robust path discovery
    dcfg = cfg["data"]
    if dcfg.get("local_dataset_dir"):
        auto_root = Path(dcfg["local_dataset_dir"])
        eeg_root, text_csv = discover_seedvii_paths(auto_root)
        if eeg_root is not None:
            dcfg["eeg_root"] = str(eeg_root)
        if text_csv is not None:
            dcfg["text_csv_path"] = str(text_csv)

    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg["runtime"].get("device", "auto"))
    print(f"[Train] device={device}")
    out_dir = Path(cfg["runtime"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    df = load_index(dcfg["npz_dir"])
    tr_df, va_df = split_index_by_subjects(df, dcfg["train_subjects"], dcfg["val_subjects"])
    print(f"train windows={len(tr_df)} val windows={len(va_df)}")

    tr_labels = tr_df["label3"].tolist()
    from collections import Counter
    print("train class counts:", dict(sorted(Counter(tr_labels).items())))
    va_labels = va_df["label3"].tolist()
    print("val class counts:", dict(sorted(Counter(va_labels).items())))

    # Normalization stats
    norm_stats_path = out_dir / "norm_stats.npz"
    if norm_stats_path.exists():
        print(f"loaded norm stats {norm_stats_path}")
        stats = dict(np.load(norm_stats_path))
    else:
        print("fitting channel stats from train shards...")
        mean, std = fit_channel_stats(tr_df, max_shards=0)
        np.savez_compressed(norm_stats_path, mean=mean, std=std)
        stats = {"mean": mean, "std": std}

    # Datasets
    train_ds = WindowNpzDataset(tr_df, stats["mean"], stats["std"],
                                 text_csv_path=dcfg["text_csv_path"],
                                 cache_size=dcfg.get("cache_size", 8))
    val_ds = WindowNpzDataset(va_df, stats["mean"], stats["std"],
                               text_csv_path=dcfg["text_csv_path"],
                               cache_size=dcfg.get("cache_size", 8))

    # Balanced batch sampler
    tcfg = cfg["train"]
    train_sampler = ClassBalancedBatchSampler(
        train_ds.df["label3"].tolist(),
        batch_size=tcfg["batch_size"],
        steps_per_epoch=tcfg.get("steps_per_epoch", None),
        seed=cfg.get("seed", 42),
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"],
        shuffle=False, num_workers=tcfg.get("num_workers", 2),
        collate_fn=collate,
    )

    # Model
    eeg, text = build_models(cfg, device)

    # Optimizer
    opt = build_optimizer(cfg, eeg, text)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"])

    # Loss
    lcfg = cfg["loss"]
    criterion = TriContrastiveLoss(
        temperature=lcfg.get("temperature", 0.07),
        beta_eeg=lcfg.get("beta_eeg", 0.65),
        beta_llm=lcfg.get("beta_llm", 0.35),
    )

    # Resume checkpoint
    start_epoch, step, best_metric = 0, 0, 0.0
    ckpt_path = out_dir / "last.pt"
    if tcfg.get("resume", True) and ckpt_path.exists():
        sd = load_ckpt(ckpt_path, eeg, text, opt, sched, device)
        start_epoch = sd.get("epoch", 0)
        step = sd.get("step", 0)
        best_metric = sd.get("best_metric", 0.0)
        print(f"[Train] resumed epoch={start_epoch} step={step} best_metric={best_metric:.4f}")

    # Training loop
    print(f"[Train] epochs={tcfg['epochs']} steps_per_epoch={len(train_sampler)}")
    for epoch in range(start_epoch, tcfg["epochs"]):
        train_sampler.set_epoch(epoch)
        eeg.train(); text.train()
        pbar = tqdm(train_sampler, desc=f"epoch {epoch}", leave=False)
        epoch_losses = []

        for batch_idx, indices in enumerate(pbar):
            batch = [train_ds[i] for i in indices]
            batch = collate(batch)

            x = batch["eeg"].to(device)
            y = batch["label"].to(device)
            texts = batch["text"]

            eeg_z = eeg(x)
            text_z = text(texts)

            loss_dict = criterion(eeg_z, text_z, y)
            loss = loss_dict["loss"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(eeg.parameters()) + list(text.parameters()), max_norm=1.0)
            opt.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "inter": f"{loss_dict['inter'].item():.4f}",
                "eeg_intra": f"{loss_dict['eeg_intra'].item():.4f}",
                "llm_intra": f"{loss_dict['llm_intra'].item():.4f}",
            })
            step += 1

        sched.step()

        metrics = evaluate(eeg, text, val_loader, device, dcfg["text_csv_path"])
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"epoch {epoch}: avg_loss={avg_loss:.4f} val_acc={metrics['acc']:.4f} val_f1={metrics['macro_f1']:.4f}")

        if metrics["macro_f1"] >= best_metric:
            best_metric = metrics["macro_f1"]
            save_ckpt(out_dir / "best.pt", eeg, text, opt, sched, epoch, step, best_metric, cfg)
            print(f"[Train] saved best model (f1={best_metric:.4f})")

        save_ckpt(ckpt_path, eeg, text, opt, sched, epoch, step, best_metric, cfg)

    print(f"[Train] done. best_metric={best_metric:.4f}")


if __name__ == "__main__":
    main()
