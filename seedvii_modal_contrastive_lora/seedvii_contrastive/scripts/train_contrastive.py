from __future__ import annotations

# Allow running this file directly, e.g.
#   python seedvii_contrastive/scripts/xxx.py
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
    """Find an actual Transformers model directory under a ModelScope cache/local dir.

    ModelScope snapshot_download may return a nested snapshot path, while notebooks
    often store a guessed path such as /mnt/workspace/models/Qwen2.5-0.5B-Instruct.
    If that guessed path does not exist, Transformers treats it as a Hub repo id
    and raises HFValidationError. This helper searches for a real directory that
    contains config.json.
    """
    if not base.exists():
        return None
    candidates = [p.parent for p in base.rglob("config.json") if p.is_file()]
    if not candidates:
        return None
    # Prefer dirs containing the requested model basename, otherwise choose the
    # shortest path (usually the snapshot root rather than a nested subdir).
    preferred_name = preferred_name.lower()
    if preferred_name:
        hits = [c for c in candidates if preferred_name in str(c).lower()]
        if hits:
            return sorted(hits, key=lambda x: len(str(x)))[0]
    return sorted(candidates, key=lambda x: len(str(x)))[0]


def resolve_llm_model_path(llm_cfg: dict) -> str:
    """Resolve local/ModelScope LLM path robustly.

    If model_name_or_path is an existing local directory, use it. If it is a
    non-existing absolute path, search its parent; if still missing, download the
    ModelScope model id specified by `modelscope_model_id`.
    """
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

    # Non-local string such as 'Qwen/Qwen2.5-0.5B-Instruct'. Let transformers handle it.
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
    """Classify EEG by nearest aggregated class prototype in L2-text embedding space.

    The text tower is fed only the user's L2 trial descriptions from text_protocol.csv.
    Three class prototypes are the mean normalized embeddings of the 80 L2 descriptions
    grouped by aggregated valence label.
    """
    eeg.eval(); text.eval()
    bank_texts, bank_labels, _ = build_l2_text_bank(text_csv_path)
    bank_z = text(bank_texts)  # (80,d)
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

    # Robust path discovery: the user's ModelScope dataset stores 1-20.mat and
    # protocol CSV directly in the dataset root. If configured text/eeg paths are
    # stale (e.g. expecting EEG_preprocessed/), rediscover from local_dataset_dir.
    dcfg = cfg.get("data", {})
    roots_to_try = []
    for key in ("local_dataset_dir", "eeg_root"):
        if dcfg.get(key):
            roots_to_try.append(Path(dcfg[key]))
    if dcfg.get("npz_dir"):
        roots_to_try.append(Path(dcfg["npz_dir"]).parent)
    for r in roots_to_try:
        if not r.exists():
            continue
        eeg_root, text_csv = discover_seedvii_paths(r)
        if eeg_root is not None and (not dcfg.get("eeg_root") or not Path(dcfg["eeg_root"]).exists()):
            cfg["data"]["eeg_root"] = str(eeg_root)
        if text_csv is not None and (not dcfg.get("text_csv_path") or not Path(dcfg["text_csv_path"]).exists()):
            cfg["data"]["text_csv_path"] = str(text_csv)
        if cfg["data"].get("text_csv_path") and Path(cfg["data"]["text_csv_path"]).exists():
            break
    if not cfg["data"].get("text_csv_path") or not Path(cfg["data"]["text_csv_path"]).exists():
        raise FileNotFoundError("Cannot find L2 text protocol CSV. Put text_protocol.csv in dataset root or set data.text_csv_path.")

    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg["runtime"].get("device", "auto"))
    out_dir = Path(cfg["runtime"]["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out_dir / "config.resolved.json")

    df = load_index(cfg["data"]["npz_dir"])
    tr_df, va_df = split_index_by_subjects(df, cfg["data"]["train_subjects"], cfg["data"]["val_subjects"])
    print(f"[{now()}] train windows={len(tr_df)} val windows={len(va_df)}")
    print("train class counts:", tr_df.label3.value_counts().sort_index().to_dict())
    print("val class counts:", va_df.label3.value_counts().sort_index().to_dict())

    stats_path = out_dir / "norm_stats.npz"
    if stats_path.exists():
        z = np.load(stats_path); mean, std = z["mean"], z["std"]
        print("loaded norm stats", stats_path)
    else:
        print("fitting channel stats on training subjects only...")
        mean, std = fit_channel_stats(tr_df)
        np.savez(stats_path, mean=mean, std=std)

    text_csv_path = cfg["data"]["text_csv_path"]
    tr_ds = WindowNpzDataset(tr_df, mean, std, text_csv_path=text_csv_path, cache_size=cfg["data"].get("cache_size", 8))
    va_ds = WindowNpzDataset(va_df, mean, std, text_csv_path=text_csv_path, cache_size=cfg["data"].get("cache_size", 8))
    sampler = ClassBalancedBatchSampler(
        tr_df.label3.tolist(), batch_size=cfg["train"]["batch_size"],
        steps_per_epoch=cfg["train"].get("steps_per_epoch"), seed=cfg.get("seed", 42))
    tr_loader = DataLoader(tr_ds, batch_sampler=sampler, num_workers=cfg["train"].get("num_workers", 2), collate_fn=collate)
    va_loader = DataLoader(va_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                           num_workers=cfg["train"].get("num_workers", 2), collate_fn=collate)

    eeg, text = build_models(cfg, device)
    opt = build_optimizer(cfg, eeg, text)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg["train"]["epochs"]), eta_min=1e-5)
    crit = TriContrastiveLoss(**cfg["loss"]).to(device)

    start_epoch = 0; global_step = 0; best = -1.0
    last_path = out_dir / "last.pt"; best_path = out_dir / "best.pt"
    if cfg["train"].get("resume", True) and last_path.exists():
        print("resuming from", last_path)
        sd = load_ckpt(last_path, eeg, text, opt, sched, device)
        start_epoch = int(sd.get("epoch", -1)) + 1
        global_step = int(sd.get("step", 0)); best = float(sd.get("best_metric", -1.0))

    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["time", "epoch", "step", "loss", "inter", "eeg_intra", "llm_intra", "val_acc", "val_macro_f1"])

    start_time = time.time(); last_save = time.time()
    max_seconds = cfg["runtime"].get("max_hours", 9.5) * 3600
    save_every = cfg["runtime"].get("save_every_minutes", 30) * 60

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        sampler.set_epoch(epoch)
        eeg.train(); text.train()
        pbar = tqdm(tr_loader, desc=f"epoch {epoch}")
        running = []
        for batch in pbar:
            x = batch["eeg"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            eeg_z = eeg(x)
            txt_z = text(batch["text"])
            losses = crit(eeg_z, txt_z, y)
            loss = losses["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(eeg.parameters()) + [p for p in text.parameters() if p.requires_grad], 1.0)
            opt.step()
            global_step += 1
            running.append(float(loss.item()))
            pbar.set_postfix(loss=np.mean(running[-20:]), inter=float(losses["inter"]), eeg=float(losses["eeg_intra"]), llm=float(losses["llm_intra"]))

            if time.time() - last_save > save_every:
                save_ckpt(last_path, eeg, text, opt, sched, epoch, global_step, best, cfg)
                last_save = time.time()
            if time.time() - start_time > max_seconds:
                print("time budget reached; saving and exiting safely")
                save_ckpt(last_path, eeg, text, opt, sched, epoch, global_step, best, cfg)
                return
        sched.step()
        val = evaluate(eeg, text, va_loader, device, cfg["data"]["text_csv_path"] )
        metric = val["macro_f1"]
        if metric > best:
            best = metric
            save_ckpt(best_path, eeg, text, opt, sched, epoch, global_step, best, cfg)
        save_ckpt(last_path, eeg, text, opt, sched, epoch, global_step, best, cfg)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([now(), epoch, global_step, np.mean(running), "", "", "", val["acc"], val["macro_f1"]])
        print(f"[{now()}] epoch={epoch} val={val} best_macro_f1={best:.4f}")


if __name__ == "__main__":
    main()
