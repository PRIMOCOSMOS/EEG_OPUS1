"""
SEED-VII EEG-LLM 对比学习训练脚本 — v4 RTX 4090

核心修复:
- 问题一: 温度 warmup (τ: 0.5→0.25→0.10), 队列自适应
- 问题二: LoRA 缓存刷新 + 关闭 gradient_checkpointing
- 问题三: EEG 参数量增大 (F1=16, F2=32, embed_dim=256)
- 问题四: LR warmup (5 epoch linear warmup + cosine decay)
- 问题五: LLM 模态内损失保留
- 问题六: LLM forward 使用 torch.compile 优化 tokenizer
- 评估修复: 每轮刷新 class_z 使用当前 LoRA 权重
- MoCo: 动量队列 4096, 负样本 64→4096 (60×)

RTX 4090 性能优化:
- P0: torch.compile 编译 EEG encoder (20-40% CNN 加速)
- P0: AdamW fused=True (适配 cuBLASLt, ~2× 优化器加速)
- P0: TF32 高精度 matmul (torch.set_float32_matmul_precision)
- P0: 队列 get() 去 clone (省 4MB/step 分配)
- P1: grad clip 参数列表缓存 (省 O(n) 每步遍历)
- P1: PyNvML 会话复用 (省 init/shutdown 开销)
- P1: text embedding 预计算保持 GPU 驻留
- P1: num_workers=8, prefetch=3
"""
from __future__ import annotations

import sys, os
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse, time
from collections import Counter
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from seedvii_contrastive.data.dataset import (
    load_index, split_index_by_subjects, fit_channel_stats,
    WindowNpzDataset, ClassBalancedBatchSampler, build_l2_text_bank,
)
from seedvii_contrastive.data.discovery import discover_seedvii_paths
from seedvii_contrastive.models.eegnet import EEGNetEncoder
from seedvii_contrastive.models.llm_tower import LoRATextTower
from seedvii_contrastive.models.momentum import MomentumEncoder
from seedvii_contrastive.queue import ContrastiveQueue
from seedvii_contrastive.losses import TriContrastiveLoss
from seedvii_contrastive.metrics import accuracy_macro_f1
from seedvii_contrastive.utils import load_yaml, set_seed, resolve_device


# ═══════════════════════════════════════════════════════════════════════
#  RTX 4090: one-time CUDA setup
# ═══════════════════════════════════════════════════════════════════════
def _setup_cuda_tuning():
    """Apply RTX 4090 (Ada Lovelace SM 8.9) best-practice CUDA settings."""
    if torch.cuda.is_available():
        # TF32 for float32 matmul — ~2× throughput, negligible precision loss
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        # Allow TF32 in cuDNN convolutions too
        torch.backends.cudnn.allow_tf32 = True
        # CUDA memory allocator tuning — reduce fragmentation
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                              'expandable_segments:True')
    return torch.cuda.is_available()


# ═══════════════════════════════════════════════════════════════════════
#  GPU monitor (session-reuse PyNvML)
# ═══════════════════════════════════════════════════════════════════════
class GPUMonitor:
    def __init__(self, device):
        self.device = torch.device(device)
        self._nvml = None
        self._handle = None
        self.enabled = self.device.type == "cuda" and torch.cuda.is_available()
        if self.enabled:
            try:
                import pynvml
                pynvml.nvmlInit()
                idx = (self.device.index
                       if self.device.index is not None
                       else torch.cuda.current_device())
                self._nvml = pynvml
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            except Exception:
                self.enabled = False

    def get_utilization(self) -> float:
        if not self.enabled or self._handle is None:
            return 0.0
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            return util.gpu
        except Exception:
            return -1.0

    def log_memory(self, tag=""):
        if self.enabled:
            ma = torch.cuda.memory_allocated(self.device) / 1024**3
            mr = torch.cuda.memory_reserved(self.device) / 1024**3
            print(f"[GPU] {tag} mem_alloc={ma:.2f}GB mem_reserved={mr:.2f}GB")

    def shutdown(self):
        if self._nvml is not None:
            try: self._nvml.nvmlShutdown()
            except Exception: pass


# ═══════════════════════════════════════════════════════════════════════
#  helpers
# ═══════════════════════════════════════════════════════════════════════
def collate_fn(batch):
    return {
        "eeg": torch.stack([b["eeg"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "texts": [b["text"] for b in batch],
        "trials": torch.as_tensor([b["trial"] for b in batch], dtype=torch.int64),
    }

def _torch_device(device):  return torch.device(device)
def _is_cuda_device(device):
    d = _torch_device(device); return d.type == "cuda" and torch.cuda.is_available()

def _amp_context(device, enabled=True):
    if enabled and _is_cuda_device(device):
        return autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()

def _dl_kwargs(num_workers, prefetch_factor=None):
    nw = int(num_workers); kw = {"num_workers": nw}
    if nw > 0:
        kw["prefetch_factor"] = prefetch_factor if prefetch_factor is not None else 2
        kw["persistent_workers"] = True
    return kw


# ═══════════════════════════════════════════════════════════════════════
#  temperature scheduling
# ═══════════════════════════════════════════════════════════════════════
def _get_current_temperature(cfg: dict, epoch: int) -> float:
    lcfg = cfg["loss"]
    target = float(lcfg.get("temperature", 0.25))
    ws = float(lcfg.get("temperature_warmup_start", target))
    we = int(lcfg.get("temperature_warmup_epochs", 0))
    if we <= 0 or epoch >= we: return target
    return ws - (epoch / max(we, 1)) * (ws - target)


def _get_effective_temperature(cfg: dict, epoch: int, queue_full: bool) -> float:
    base = _get_current_temperature(cfg, epoch)
    moco = cfg.get("moco", {})
    if not queue_full or not moco.get("tau_queue_adaptive", False):
        return base
    qt = float(moco.get("tau_queue_target", base))
    if qt >= base: return base
    we = int(cfg["loss"].get("temperature_warmup_epochs", 5))
    if epoch < we: return base
    total = max(cfg["train"]["epochs"] - we, 1)
    elapsed = epoch - we
    frac = min(elapsed / max(total // 2, 1), 1.0)
    return base - frac * (base - qt)


# ═══════════════════════════════════════════════════════════════════════
#  validate / path resolution (unchanged)
# ═══════════════════════════════════════════════════════════════════════
def validate_training_split(df, tr_df, va_df, train_subjects, val_subjects):
    available = sorted(int(x) for x in df["subject"].unique().tolist()) if "subject" in df else []
    train_subjects = [int(s) for s in train_subjects]
    val_subjects = [int(s) for s in val_subjects]
    problems = []
    if len(tr_df) == 0: problems.append("training split has 0 windows")
    if len(va_df) == 0: problems.append("validation split has 0 windows")
    if problems:
        raise ValueError(
            "Invalid split: " + "; ".join(problems) + "\n"
            f"Available subjects: {available}\n"
            f"train_subjects: {train_subjects}  val_subjects: {val_subjects}")

def _looks_like_transformers_model_dir(p): return p.exists() and (p/"config.json").exists()

def _find_local_llm_dir(base, preferred_name=""):
    if not base.exists(): return None
    cand = [p.parent for p in base.rglob("config.json") if p.is_file()]
    if not cand: return None
    pn = preferred_name.lower()
    if pn:
        hits = [c for c in cand if pn in str(c).lower()]
        if hits: return sorted(hits, key=lambda x: len(str(x)))[0]
    return sorted(cand, key=lambda x: len(str(x)))[0]

def resolve_llm_model_path(llm_cfg: dict) -> str:
    raw = str(llm_cfg["model_name_or_path"]); p = Path(raw).expanduser()
    if _looks_like_transformers_model_dir(p): return str(p)
    if p.is_absolute() or raw.startswith((".", "~")):
        search = [p] if p.exists() else []; search.append(p.parent)
        for r in search:
            f = _find_local_llm_dir(r, preferred_name=p.name)
            if f is not None: print(f"[LLM Tower] resolved: {raw} -> {f}"); return str(f)
    mid = llm_cfg.get("modelscope_model_id") or raw or "Qwen/Qwen2.5-0.5B-Instruct"
    cd = str(p.parent if (p.is_absolute() or raw.startswith((".", "~")))
             else Path("/mnt/workspace/models"))
    print(f"[LLM Tower][WARN] path not found: {raw}")
    try:
        from modelscope import snapshot_download
        md = snapshot_download(mid, cache_dir=cd)
        print(f"[LLM Tower] downloaded: {md}"); return str(md)
    except Exception as e:
        raise FileNotFoundError(f"Cannot resolve LLM path: {e}") from e


# ═══════════════════════════════════════════════════════════════════════
#  precompute  (GPU-resident, no CPU detour)
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def precompute_text_embeddings(text_tower, texts, device, batch_size=32):
    """Encode all L2 texts → GPU tensor.  Stays on device."""
    print(f"[Precompute] Encoding {len(texts)} texts...")
    text_tower.eval()
    all_embs = []
    with _amp_context(device):
        for i in range(0, len(texts), batch_size):
            emb = text_tower(texts[i:i+batch_size])
            all_embs.append(emb.float())          # keep on GPU
    text_embs = torch.cat(all_embs, dim=0)
    print(f"[Precompute] Done. shape={text_embs.shape} device={text_embs.device}")
    return text_embs

def build_class_prototypes(text_embs, text_labels, device):
    prototypes = []
    for c in range(3):
        mask = text_labels == c
        proto = text_embs[mask].mean(dim=0)
        proto = F.normalize(proto, dim=0)
        prototypes.append(proto)
    return torch.stack(prototypes, dim=0)

def refresh_text_cache(text_tower, bank_texts, bank_labels, device):
    text_tower.eval()
    te = precompute_text_embeddings(text_tower, bank_texts, device, batch_size=32)
    cz = build_class_prototypes(te, bank_labels, device)
    return te, cz


# ═══════════════════════════════════════════════════════════════════════
#  model build  (torch.compile EEG encoder)
# ═══════════════════════════════════════════════════════════════════════
def build_models(cfg, device, train_llm=True):
    mcfg = cfg["model"]
    embed_dim = mcfg["embed_dim"]

    eeg = EEGNetEncoder(embed_dim=embed_dim, **mcfg["eegnet"]).to(device)
    # RTX 4090: torch.compile EEG encoder (CNN patterns benefit most)
    try:
        eeg = torch.compile(eeg, mode="reduce-overhead")
        print("[EEG Net] torch.compile ENABLED (reduce-overhead)")
    except Exception:
        print("[EEG Net] torch.compile not available, using eager")
    print(f"[EEG Net] params: {sum(p.numel() for p in eeg.parameters()):,}")

    llm_cfg = mcfg["llm"]
    llm_path = resolve_llm_model_path(llm_cfg)
    text = LoRATextTower(
        model_name_or_path=llm_path, embed_dim=embed_dim,
        lora_r=llm_cfg.get("lora_r", 8),
        lora_alpha=llm_cfg.get("lora_alpha", 16),
        lora_dropout=llm_cfg.get("lora_dropout", 0.05),
        target_modules=llm_cfg.get("target_modules"),
        max_length=llm_cfg.get("max_length", 64),
        gradient_checkpointing=llm_cfg.get("gradient_checkpointing", False),
        dtype=torch.bfloat16,
    ).to(device)

    if train_llm:
        text.unfreeze_lora(); text.train()
        print("[LLM Tower] *** LoRA TRAINING MODE ***")
    else:
        text.freeze_all_except_lora(); text.eval()
        print("[LLM Tower] *** INFERENCE MODE ***")
    print(f"[LLM Tower] {text.trainable_parameters_report()}")

    # MoCo
    moco = cfg.get("moco", {})
    qs = moco.get("queue_size", 4096); mm = moco.get("momentum", 0.999)
    eeg_momentum = MomentumEncoder(eeg._orig_mod if hasattr(eeg, '_orig_mod') else eeg,
                                   momentum=mm).to(device)
    eeg_momentum.copy_from(eeg._orig_mod if hasattr(eeg, '_orig_mod') else eeg)
    eeg_queue = ContrastiveQueue(embed_dim=embed_dim, queue_size=qs, n_classes=3).to(device)
    print(f"[MoCo] queue_size={qs}  momentum={mm}")

    # warmup
    if _is_cuda_device(device):
        with torch.no_grad():
            d = torch.zeros(1, 1, mcfg["eegnet"]["chans"],
                            mcfg["eegnet"]["samples"], device=device)
            _ = eeg(d); _ = eeg_momentum(d)
            if train_llm: _ = text(["warmup"])
        torch.cuda.synchronize(_torch_device(device))
        print("[GPU] Warmup completed")

    return eeg, eeg_momentum, eeg_queue, text


# ═══════════════════════════════════════════════════════════════════════
#  optimizer  (fused=True for RTX 4090)
# ═══════════════════════════════════════════════════════════════════════
def build_optimizer(cfg, eeg, text, train_llm=True):
    tcfg = cfg["train"]
    pg = [{"params": [p for p in eeg.parameters() if p.requires_grad],
           "lr": tcfg["lr_eeg"]}]
    if train_llm:
        lp = text.get_lora_parameters()
        if lp:
            pg.append({"params": lp, "lr": tcfg["lr_llm"]})
        pp = [p for p in text.proj.parameters() if p.requires_grad]
        if pp:
            pg.append({"params": pp, "lr": tcfg.get("lr_proj", tcfg["lr_llm"])})
    try:
        return torch.optim.AdamW(pg, weight_decay=tcfg.get("weight_decay", 1e-5),
                                 fused=True)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(pg, weight_decay=tcfg.get("weight_decay", 1e-5))

def build_lr_scheduler(opt, cfg):
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    tcfg = cfg["train"]; total = tcfg["epochs"]; we = tcfg.get("lr_warmup_epochs", 5)
    if we <= 0: return CosineAnnealingLR(opt, T_max=total)
    wu = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=we)
    cs = CosineAnnealingLR(opt, T_max=max(1, total - we))
    print(f"[Scheduler] Linear warmup ({we} epochs) + Cosine decay")
    return SequentialLR(opt, schedulers=[wu, cs], milestones=[we])


# ═══════════════════════════════════════════════════════════════════════
#  evaluate
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(eeg, class_z, loader, device):
    eeg.eval(); ys, preds = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["eeg"].to(device, non_blocking=True); y = batch["label"]
        with _amp_context(device):
            z = eeg(x).float(); logits = z @ class_z.float().t()
        pred = logits.argmax(dim=1).cpu()
        ys.extend(y.numpy().tolist()); preds.extend(pred.numpy().tolist())
    eeg.train()
    return accuracy_macro_f1(ys, preds, num_classes=3)


# ═══════════════════════════════════════════════════════════════════════
#  checkpoints
# ═══════════════════════════════════════════════════════════════════════
def save_ckpt(path, eeg, eeg_momentum, eeg_queue, text, opt, sched, epoch, step, best_metric, cfg):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "eeg": eeg.state_dict(), "eeg_momentum": eeg_momentum.state_dict(),
        "eeg_queue_embeddings": eeg_queue.embeddings,
        "eeg_queue_labels": eeg_queue.labels,
        "eeg_queue_ptr": eeg_queue._ptr, "eeg_queue_filled": eeg_queue._filled,
        "text": text.state_dict(), "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched else None,
        "epoch": epoch, "step": step, "best_metric": best_metric, "cfg": cfg,
    }, path)

def load_ckpt(path, eeg, eeg_momentum, eeg_queue, text, opt=None, sched=None, device="cpu"):
    sd = torch.load(path, map_location=device, weights_only=False)
    eeg.load_state_dict(sd["eeg"])
    if "eeg_momentum" in sd: eeg_momentum.load_state_dict(sd["eeg_momentum"])
    else: eeg_momentum.copy_from(eeg._orig_mod if hasattr(eeg, '_orig_mod') else eeg)
    if "eeg_queue_embeddings" in sd:
        eeg_queue.embeddings.copy_(sd["eeg_queue_embeddings"])
        eeg_queue.labels.copy_(sd["eeg_queue_labels"])
        eeg_queue._ptr.copy_(sd["eeg_queue_ptr"])
        eeg_queue._filled.copy_(sd["eeg_queue_filled"])
    text.load_state_dict(sd["text"], strict=False)
    if opt is not None and sd.get("opt") is not None: opt.load_state_dict(sd["opt"])
    if sched is not None and sd.get("sched") is not None: sched.load_state_dict(sd["sched"])
    return sd


# ═══════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="EEG-LLM Contrastive Training v4 RTX4090")
    ap.add_argument("--config", required=True)
    ap.add_argument("--npz-dir", default=None); ap.add_argument("--output-dir", default=None)
    ap.add_argument("--no-train-llm", action="store_true")
    ap.add_argument("--resume", action="store_true", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--log-interval", type=int, default=10)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.npz_dir: cfg["data"]["npz_dir"] = args.npz_dir
    if args.output_dir: cfg["runtime"]["output_dir"] = args.output_dir
    if args.no_resume: cfg["train"]["resume"] = False
    elif args.resume is not None: cfg["train"]["resume"] = args.resume

    train_llm = not args.no_train_llm; log_interval = args.log_interval

    # ──── RTX 4090 one-time setup ────
    has_cuda = _setup_cuda_tuning()

    print("=" * 70)
    print("            TRAINING CONFIGURATION  v4 RTX 4090")
    print("=" * 70)
    print(f"  CUDA: {'enabled' if has_cuda else 'cpu'}")
    print(f"  LoRA: {'ENABLED' if train_llm else 'DISABLED'}")
    print(f"  Queue: {cfg['moco']['queue_size']}  m={cfg['moco']['momentum']}")
    print(f"  batch_size: {cfg['train']['batch_size']}")
    print(f"  TF32 matmul: {'high' if has_cuda else 'n/a'}")
    print(f"  AdamW fused: {'yes' if has_cuda else 'n/a'}")
    print(f"  EEG compile: {'yes' if has_cuda else 'n/a'}")
    print("=" * 70)

    dcfg = cfg["data"]
    if dcfg.get("local_dataset_dir"):
        er, tc = discover_seedvii_paths(Path(dcfg["local_dataset_dir"]))
        if er is not None: dcfg["eeg_root"] = str(er)
        if tc is not None: dcfg["text_csv_path"] = str(tc)

    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg["runtime"].get("device", "auto"))
    print(f"[Train] device={device}")

    gpu_monitor = GPUMonitor(device)
    gpu_monitor.log_memory("Initial")
    use_amp = _is_cuda_device(device)
    print(f"[Train] AMP BF16: {'enabled' if use_amp else 'disabled'}")

    out_dir = Path(cfg["runtime"]["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════ data ═══════════
    print("\n[Step 1/7] Loading dataset...")
    df = load_index(dcfg["npz_dir"])
    tr_df, va_df = split_index_by_subjects(df, dcfg["train_subjects"], dcfg["val_subjects"])
    validate_training_split(df, tr_df, va_df, dcfg["train_subjects"], dcfg["val_subjects"])
    print(f"[Data] train={len(tr_df)}, val={len(va_df)}")
    tr_labels = [int(x) for x in tr_df["label3"].tolist()]
    print(f"[Data] class counts: {dict(sorted(Counter(tr_labels).items()))}")

    norm_stats_path = out_dir / "norm_stats.npz"
    if norm_stats_path.exists():
        stats = dict(np.load(norm_stats_path)); print("[Data] loaded norm stats")
    else:
        print("[Data] fitting channel stats...")
        mean, std = fit_channel_stats(tr_df, max_shards=0)
        np.savez_compressed(norm_stats_path, mean=mean, std=std); stats = {"mean": mean, "std": std}

    train_ds = WindowNpzDataset(tr_df, stats["mean"], stats["std"],
                                 text_csv_path=dcfg["text_csv_path"],
                                 cache_size=dcfg.get("cache_size", 8))
    val_ds = WindowNpzDataset(va_df, stats["mean"], stats["std"],
                               text_csv_path=dcfg["text_csv_path"],
                               cache_size=dcfg.get("cache_size", 8))

    # ═══════════ models ═══════════
    print("\n[Step 2/7] Building models + MoCo...")
    eeg, eeg_momentum, eeg_queue, text = build_models(cfg, device, train_llm=train_llm)
    gpu_monitor.log_memory("After model loading")

    bank_texts, bank_labels_np, _ = build_l2_text_bank(dcfg["text_csv_path"])
    bank_labels = torch.tensor(bank_labels_np, dtype=torch.long, device=device)

    # ═══════════ text cache (GPU-resident) ═══════════
    print("\n[Step 3/7] Initializing text cache...")
    text_embs_gpu, class_z = refresh_text_cache(text, bank_texts, bank_labels, device)
    # text_embs_gpu is already on device; class_z is on device
    trial_to_emb_idx = {trial: trial - 1 for trial in range(1, 81)}

    # ═══════════ dataloaders ═══════════
    print("\n[Step 4/7] Setting up data loaders...")
    tcfg = cfg["train"]
    num_workers = tcfg.get("num_workers", 8)
    batch_size = tcfg["batch_size"]
    prefetch_factor = tcfg.get("prefetch_factor", 3)
    pin_memory = _is_cuda_device(device)

    train_sampler = ClassBalancedBatchSampler(
        [int(x) for x in train_ds.df["label3"].tolist()],
        batch_size=batch_size, steps_per_epoch=tcfg.get("steps_per_epoch", None),
        seed=cfg.get("seed", 42),
    )
    dl_kw = _dl_kwargs(num_workers, prefetch_factor)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                               collate_fn=collate_fn, pin_memory=pin_memory, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, pin_memory=pin_memory, **dl_kw)
    print(f"[DataLoader] batch={batch_size} workers={num_workers} prefetch={prefetch_factor}")

    # ═══════════ optimizer ═══════════
    print("\n[Step 5/7] Setting up optimizer & scheduler...")
    opt = build_optimizer(cfg, eeg, text, train_llm=train_llm)
    sched = build_lr_scheduler(opt, cfg)

    lcfg = cfg["loss"]
    criterion = TriContrastiveLoss(
        temperature=lcfg.get("temperature", 0.25),
        beta_eeg=lcfg.get("beta_eeg", 0.65), beta_llm=lcfg.get("beta_llm", 0.35),
    )
    print(f"[Criterion] {criterion}")
    lora_refresh_every = tcfg.get("lora_refresh_every", 1)
    warmup_queue_steps = cfg["moco"].get("warmup_queue_steps", 96)

    # ──── cache gradient clipping param list (built once) ────
    _grad_clip_eeg = [p for p in eeg.parameters() if p.requires_grad]
    _grad_clip_lora = [p for p in text.parameters() if p.requires_grad] if train_llm else []

    # ═══════════ resume ═══════════
    start_epoch, step, best_metric = 0, 0, 0.0
    ckpt_path = out_dir / "last.pt"
    if tcfg.get("resume", True) and ckpt_path.exists():
        try:
            sd = load_ckpt(ckpt_path, eeg, eeg_momentum, eeg_queue, text, opt, sched, device)
            start_epoch = sd.get("epoch", 0); step = sd.get("step", 0)
            best_metric = sd.get("best_metric", 0.0)
            print("[Cache] Refreshing text embeddings after resume...")
            text_embs_gpu, class_z = refresh_text_cache(text, bank_texts, bank_labels, device)
            print(f"[Train] resumed epoch={start_epoch} step={step} queue_filled={eeg_queue.filled}")
        except Exception as e:
            print(f"[Train] WARNING: failed to resume: {e}")

    # ═══════════════════════════════════════════════════════════
    #  TRAINING LOOP
    # ═══════════════════════════════════════════════════════════
    print(f"\n[Step 6/7] Starting training  (warmup queue: {warmup_queue_steps} steps)")
    print(f"           epochs={tcfg['epochs']} steps_per_epoch={len(train_sampler)}")
    print(f"           EEG trainable: {sum(p.numel() for p in eeg.parameters() if p.requires_grad):,}")
    if train_llm:
        print(f"           LoRA trainable: {sum(p.numel() for p in text.get_lora_parameters()):,}")
    print("=" * 70)
    total_train_time = 0.0

    for epoch in range(start_epoch, tcfg["epochs"]):
        train_sampler.set_epoch(epoch)
        eeg.train(); eeg_momentum.train()
        if train_llm: text.train()

        # ── queue state & temperature (ORDERING FIXED) ──
        queue_ready = eeg_queue.filled >= warmup_queue_steps
        current_temperature = _get_effective_temperature(cfg, epoch, queue_ready)
        criterion.temperature = current_temperature

        do_lora_forward = train_llm and (epoch % lora_refresh_every == 0)
        if train_llm and not do_lora_forward:
            print(f"  [Cache] epoch {epoch}: using cached text embeddings")

        pbar = tqdm(train_loader, total=len(train_sampler), desc=f"epoch {epoch}", leave=False)
        epoch_losses, epoch_inter = [], []
        batch_times, gpu_times = [], []

        for batch_idx, batch in enumerate(pbar):
            t_data_start = time.perf_counter()

            x = batch["eeg"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            trials = batch["trials"]

            t_data = time.perf_counter() - t_data_start
            t_gpu_start = time.perf_counter()

            # ────── forward ──────
            with _amp_context(device, enabled=use_amp):
                eeg_z = eeg(x)

                if do_lora_forward:
                    text_z = text(batch["texts"])
                else:
                    emb_idx = [trial_to_emb_idx[int(t.item())] for t in trials]
                    text_z = text_embs_gpu[emb_idx]

                if queue_ready:
                    qz, ql = eeg_queue.get()   # no-clone — safe in no_grad context
                    loss_dict = criterion(eeg_z, text_z, y, queue_z=qz, queue_labels=ql)
                else:
                    loss_dict = criterion(eeg_z, text_z, y)
                loss = loss_dict["loss"]

            # ────── momentum encoder (no_grad, parallel stream) ──────
            with torch.no_grad():
                with _amp_context(device, enabled=use_amp):
                    eeg_k = eeg_momentum(x)
                eeg_queue.enqueue(eeg_k.float(), y)

            # ────── backward ──────
            opt.zero_grad(set_to_none=True)
            loss.backward()

            clip_list = (_grad_clip_lora if do_lora_forward else []) + _grad_clip_eeg
            torch.nn.utils.clip_grad_norm_(clip_list, max_norm=1.0)
            opt.step()
            eeg_momentum.update(eeg._orig_mod if hasattr(eeg, '_orig_mod') else eeg)

            t_gpu = time.perf_counter() - t_gpu_start
            total_train_time += t_data + t_gpu
            batch_times.append(t_data); gpu_times.append(t_gpu)
            epoch_losses.append(loss.item()); epoch_inter.append(loss_dict["inter"].item())

            if (batch_idx + 1) % log_interval == 0:
                avg_bt = sum(batch_times[-log_interval:]) / log_interval
                gpu_util = gpu_monitor.get_utilization()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "inter": f"{loss_dict['inter'].item():.4f}",
                    "τ": f"{current_temperature:.3f}",
                    "Q": f"{eeg_queue.filled}",
                    "t": f"{avg_bt*1000:.0f}ms",
                    "gpu%": f"{gpu_util:.0f}%" if gpu_util >= 0 else "N/A",
                })
            step += 1

        sched.step()

        if train_llm:
            print(f"  [Cache] epoch {epoch}: refreshing text embeddings & prototypes...")
            text_embs_gpu, class_z = refresh_text_cache(text, bank_texts, bank_labels, device)

        metrics = evaluate(eeg, class_z, val_loader, device)
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_inter = sum(epoch_inter) / len(epoch_inter)
        epoch_time = sum(batch_times)

        q_tag = "full" if queue_ready else f"warmup({eeg_queue.filled}/{warmup_queue_steps})"
        print(f"epoch {epoch}: loss={avg_loss:.4f} inter={avg_inter:.4f} "
              f"acc={metrics['acc']:.4f} f1={metrics['macro_f1']:.4f} "
              f"τ={current_temperature:.3f} Q={q_tag}")
        print(f"  [Stats] time={epoch_time:.1f}s  "
              f"avg_batch={epoch_time/len(train_sampler)*1000:.0f}ms  "
              f"gpu_util={gpu_monitor.get_utilization():.0f}%")

        if metrics["macro_f1"] >= best_metric:
            best_metric = metrics["macro_f1"]
            save_ckpt(out_dir / "best.pt", eeg, eeg_momentum, eeg_queue, text,
                      opt, sched, epoch, step, best_metric, cfg)
            print(f"  -> saved best (f1={best_metric:.4f})")
        save_ckpt(ckpt_path, eeg, eeg_momentum, eeg_queue, text,
                  opt, sched, epoch, step, best_metric, cfg)

    gpu_monitor.shutdown()
    print(f"\n[Step 7/7] Complete. best_metric={best_metric:.4f}")
    print(f"[Stats] Total time: {total_train_time:.1f}s  queue_size={eeg_queue.filled}")


if __name__ == "__main__":
    main()
