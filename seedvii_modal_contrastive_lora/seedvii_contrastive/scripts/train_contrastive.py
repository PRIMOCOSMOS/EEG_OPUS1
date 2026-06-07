"""
SEED-VII EEG-LLM 对比学习训练脚本 - GPU优化版

优化目标: 最大化GPU利用率，减少CPU-GPU数据传输瓶颈
"""
from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path
from collections import Counter
import time
import threading
from queue import Queue

import numpy as np
import torch
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
from seedvii_contrastive.losses import TriContrastiveLoss
from seedvii_contrastive.metrics import accuracy_macro_f1
from seedvii_contrastive.utils import load_yaml, save_json, set_seed, resolve_device


# =============== GPU 监控工具 ===============
class GPUMonitor:
    """GPU利用率监控器"""
    def __init__(self, device):
        self.device = device
        self.enabled = torch.cuda.is_available()
        self._utilization_history = []
        
    def get_utilization(self) -> float:
        """获取GPU利用率百分比"""
        if not self.enabled:
            return 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(
                0 if self.device.type == 'cuda' else int(str(self.device).split(':')[-1])
            )
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pynvml.nvmlShutdown()
            return util.gpu
        except:
            return -1.0
    
    def log_memory(self, tag: str = ""):
        """记录GPU内存使用"""
        if self.enabled:
            mem_alloc = torch.cuda.memory_allocated(self.device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print(f"[GPU] {tag} mem_alloc={mem_alloc:.2f}GB mem_reserved={mem_reserved:.2f}GB")


# =============== 优化后的collate_fn ===============
def collate_fn(batch):
    """优化的collate函数，减少Python对象创建开销"""
    return {
        "eeg": torch.stack([b["eeg"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "texts": [b["text"] for b in batch],  # 重命名避免冲突
        "subjects": [b["subject"] for b in batch],
        "trials": torch.as_tensor([b["trial"] for b in batch], dtype=torch.int64),  # 使用tensor
    }


# =============== 异步数据预加载器 ===============
class AsyncDataLoader:
    """异步数据预加载器，将CPU负载与GPU计算重叠"""
    def __init__(self, dataset, sampler, batch_size, device, pin_memory=True, num_workers=4, prefetch_factor=2):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=num_workers > 0,
        )
        self.device = device
        self.queue = Queue(maxsize=2)
        self.thread = None
        self._stop = False
        
    def start(self):
        """启动后台预加载线程"""
        if self.thread is None:
            self._stop = False
            self.thread = threading.Thread(target=self._preload_loop, daemon=True)
            self.thread.start()
            
    def _preload_loop(self):
        """后台预加载循环"""
        iterator = iter(self.dataloader)
        while not self._stop:
            try:
                batch = next(iterator)
                # 将数据放到GPU上（异步）
                batch_gpu = {
                    "eeg": batch["eeg"].to(self.device, non_blocking=True),
                    "label": batch["label"].to(self.device, non_blocking=True),
                    "texts": batch["texts"],
                    "trials": batch["trials"].to(self.device, non_blocking=True),
                }
                self.queue.put((batch_gpu, batch["texts"]), timeout=0.1)
            except StopIteration:
                break
            except Exception as e:
                if not self._stop:
                    pass
                    
    def get_batch(self, timeout=1.0):
        """获取预加载的batch"""
        return self.queue.get(timeout=timeout)
    
    def stop(self):
        """停止预加载"""
        self._stop = True
        if self.thread:
            self.thread.join(timeout=1.0)


# =============== 路径解析函数 ===============
def _looks_like_transformers_model_dir(path: Path) -> bool:
    return path.exists() and (path / "config.json").exists()


def _find_local_llm_dir(base: Path, preferred_name: str = "") -> Path | None:
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
    print(f"[LLM Tower][WARN] local model path does not exist: {raw}")
    
    try:
        from modelscope import snapshot_download
        model_dir = snapshot_download(model_id, cache_dir=cache_dir)
        print(f"[LLM Tower] downloaded model_dir={model_dir}")
        return str(model_dir)
    except Exception as e:
        raise FileNotFoundError(f"Cannot resolve LLM path: {e}") from e


# =============== 预计算函数 ===============
def precompute_text_embeddings(text_tower, texts: list, device, batch_size: int = 16):
    """预计算所有文本嵌入，使用优化配置"""
    print(f"[Precompute] Encoding {len(texts)} text embeddings...")
    text_tower.eval()
    
    all_embeddings = []
    with torch.no_grad():
        # 使用BF16加速预计算
        with autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                emb = text_tower(batch)
                # 转换为float32存储（与EEG塔一致）
                all_embeddings.append(emb.float().cpu())
                
    text_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"[Precompute] Done. Embeddings shape: {text_embeddings.shape}, dtype={text_embeddings.dtype}")
    return text_embeddings


def build_class_prototypes(text_embeddings, text_labels, device):
    """构建类别原型向量"""
    print("[Precompute] Building class prototypes...")
    prototypes = []
    for c in range(3):
        mask = text_labels == c
        proto = text_embeddings[mask].mean(dim=0)
        proto = torch.nn.functional.normalize(proto, dim=0)
        prototypes.append(proto)
    
    class_z = torch.stack(prototypes, dim=0).to(device, non_blocking=True)
    print(f"[Precompute] Class prototypes: {class_z.shape}")
    return class_z


# =============== 模型构建 ===============
def build_models(cfg, device, train_llm: bool = False):
    """构建双塔模型"""
    mcfg = cfg["model"]
    
    # 构建EEG编码器（float32以确保数值稳定）
    eeg = EEGNetEncoder(embed_dim=mcfg["embed_dim"], **mcfg["eegnet"]).to(device)
    eeg = eeg.float()  # 确保float32
    print(f"[EEG Net] parameters: {sum(p.numel() for p in eeg.parameters()):,}")
    
    # 构建LLM文本塔
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
        dtype=torch.bfloat16,  # LLM使用BF16加速，损失函数会转换
    ).to(device)
    
    # 配置LoRA训练模式
    if train_llm:
        text.unfreeze_lora()
        print(f"[LLM Tower] Training mode: LoRA ENABLED")
    else:
        text.freeze_all_except_lora()
        text.eval()  # 推理模式设置eval
        print(f"[LLM Tower] Inference mode: LLM FROZEN")
    
    print(f"[LLM Tower] {text.trainable_parameters_report()}")
    
    # 预热GPU（减少首次计算延迟）
    if torch.cuda.is_available():
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 62, 800, device=device)
            _ = eeg(dummy)
            if train_llm:
                dummy_text = ["warmup"]
                _ = text(dummy_text)
        torch.cuda.synchronize()
        print("[GPU] Warmup completed")
    
    return eeg, text


def build_optimizer(cfg, eeg, text, train_llm: bool = False):
    """构建优化器"""
    tcfg = cfg["train"]
    
    param_groups = [
        {"params": [p for p in eeg.parameters() if p.requires_grad], "lr": tcfg["lr_eeg"]},
    ]
    
    if train_llm:
        lora_params = text.get_lora_parameters()
        if lora_params:
            param_groups.append({"params": lora_params, "lr": tcfg["lr_llm"]})
            print(f"[Optimizer] LoRA params: {len(lora_params)}, lr={tcfg['lr_llm']}")
        
        proj_params = [p for p in text.proj.parameters() if p.requires_grad]
        if proj_params:
            param_groups.append({"params": proj_params, "lr": tcfg.get("lr_proj", tcfg["lr_llm"])})
            print(f"[Optimizer] proj params: {len(proj_params)}")
    
    return torch.optim.AdamW(param_groups, weight_decay=tcfg.get("weight_decay", 1e-5))


# =============== 评估函数 ===============
@torch.no_grad()
def evaluate(eeg, class_z, loader, device):
    """评估模式"""
    eeg.eval()
    ys, preds = [], []
    
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["eeg"].to(device, non_blocking=True)
        y = batch["label"]
        
        with autocast("cuda", dtype=torch.bfloat16):
            z = eeg(x).float()
            logits = z @ class_z.float().t()
        
        pred = logits.argmax(dim=1).cpu()
        
        ys.extend(y.numpy().tolist())
        preds.extend(pred.numpy().tolist())
    
    eeg.train()
    return accuracy_macro_f1(ys, preds, num_classes=3)


# =============== 检查点函数 ===============
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


# =============== 主训练循环 ===============
def main():
    ap = argparse.ArgumentParser(description="Optimized EEG-LLM Contrastive Training")
    ap.add_argument("--config", required=True)
    ap.add_argument("--npz-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--train-llm", action="store_true")
    ap.add_argument("--resume", action="store_true", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--log-interval", type=int, default=10, help="Log interval in steps")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.npz_dir:
        cfg["data"]["npz_dir"] = args.npz_dir
    if args.output_dir:
        cfg["runtime"]["output_dir"] = args.output_dir
    
    if args.no_resume:
        cfg["train"]["resume"] = False
    elif args.resume is not None:
        cfg["train"]["resume"] = args.resume
    
    train_llm = args.train_llm
    log_interval = args.log_interval
    
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"  train_llm: {train_llm}")
    print(f"  log_interval: {log_interval}")
    print("=" * 70)

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
    
    # 初始化GPU监控
    gpu_monitor = GPUMonitor(device)
    gpu_monitor.log_memory("Initial")
    
    # 混合精度配置 (使用新版API)
    use_amp = torch.cuda.is_available()
    print(f"[Train] Mixed Precision (AMP BF16): {'enabled' if use_amp else 'disabled'}")
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    out_dir = Path(cfg["runtime"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 数据准备 ====================
    print("\n[Step 1/6] Loading dataset...")
    df = load_index(dcfg["npz_dir"])
    tr_df, va_df = split_index_by_subjects(df, dcfg["train_subjects"], dcfg["val_subjects"])
    print(f"[Data] train windows={len(tr_df)} val windows={len(va_df)}")

    tr_labels = tr_df["label3"].tolist()
    print(f"[Data] class counts: {dict(sorted(Counter(tr_labels).items()))}")

    norm_stats_path = out_dir / "norm_stats.npz"
    if norm_stats_path.exists():
        print(f"[Data] loaded norm stats")
        stats = dict(np.load(norm_stats_path))
    else:
        print("[Data] fitting channel stats...")
        mean, std = fit_channel_stats(tr_df, max_shards=0)
        np.savez_compressed(norm_stats_path, mean=mean, std=std)
        stats = {"mean": mean, "std": std}

    train_ds = WindowNpzDataset(tr_df, stats["mean"], stats["std"],
                                 text_csv_path=dcfg["text_csv_path"],
                                 cache_size=dcfg.get("cache_size", 8))
    val_ds = WindowNpzDataset(va_df, stats["mean"], stats["std"],
                               text_csv_path=dcfg["text_csv_path"],
                               cache_size=dcfg.get("cache_size", 8))

    # ==================== 模型构建 ====================
    print("\n[Step 2/6] Building models...")
    eeg, text = build_models(cfg, device, train_llm=train_llm)
    gpu_monitor.log_memory("After model loading")
    
    bank_texts, bank_labels_np, _ = build_l2_text_bank(dcfg["text_csv_path"])
    bank_labels = torch.tensor(bank_labels_np, dtype=torch.long)
    
    # ==================== 预计算文本嵌入 ====================
    print(f"\n[Step 3/6] Precomputing text embeddings...")
    text_embeddings = precompute_text_embeddings(text, bank_texts, device, batch_size=32)
    class_z = build_class_prototypes(text_embeddings, bank_labels, device)
    
    # 预计算的文本嵌入移到GPU（一次性）
    text_embeddings_gpu = text_embeddings.to(device, non_blocking=True)
    
    # Trial索引映射（保持简单列表查找）
    trial_to_emb_idx = [trial - 1 for trial in range(1, 81)]
    
    # ==================== 数据加载器 ====================
    print("\n[Step 4/6] Setting up optimized data loaders...")
    tcfg = cfg["train"]
    num_workers = tcfg.get("num_workers", 4)
    batch_size = tcfg["batch_size"]
    prefetch_factor = tcfg.get("prefetch_factor", 2)
    
    # 训练采样器
    train_sampler = ClassBalancedBatchSampler(
        train_ds.df["label3"].tolist(),
        batch_size=batch_size,
        steps_per_epoch=tcfg.get("steps_per_epoch", None),
        seed=cfg.get("seed", 42),
    )
    
    # 优化的训练数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,  # 保持worker进程
    )
    
    # 验证数据加载器
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )
    
    print(f"[DataLoader] batch_size={batch_size}, num_workers={num_workers}, prefetch={prefetch_factor}")
    print(f"[DataLoader] train samples={len(train_sampler)}, val samples={len(val_loader)}")

    # ==================== 优化器设置 ====================
    print("\n[Step 5/6] Setting up optimizer...")
    opt = build_optimizer(cfg, eeg, text, train_llm=train_llm)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"])
    
    lcfg = cfg["loss"]
    criterion = TriContrastiveLoss(
        temperature=lcfg.get("temperature", 0.07),
        beta_eeg=lcfg.get("beta_eeg", 0.65),
        beta_llm=lcfg.get("beta_llm", 0.35),
    )

    # 恢复检查点
    start_epoch, step, best_metric = 0, 0, 0.0
    ckpt_path = out_dir / "last.pt"
    if tcfg.get("resume", True) and ckpt_path.exists():
        try:
            sd = load_ckpt(ckpt_path, eeg, text, opt, sched, device)
            start_epoch = sd.get("epoch", 0)
            step = sd.get("step", 0)
            best_metric = sd.get("best_metric", 0.0)
            print(f"[Train] resumed from epoch={start_epoch}, step={step}")
        except Exception as e:
            print(f"[Train] WARNING: failed to resume: {e}")

    # ==================== 训练循环 ====================
    print(f"\n[Step 6/6] Starting training")
    print(f"           epochs={tcfg['epochs']}, steps_per_epoch={len(train_sampler)}")
    print(f"           EEG trainable: {sum(p.numel() for p in eeg.parameters() if p.requires_grad):,}")
    if train_llm:
        print(f"           LoRA trainable: {sum(p.numel() for p in text.get_lora_parameters()):,}")
    print("=" * 70)
    
    # 训练统计
    total_train_time = 0.0
    total_gpu_time = 0.0
    
    for epoch in range(start_epoch, tcfg["epochs"]):
        train_sampler.set_epoch(epoch)
        eeg.train()
        if train_llm:
            text.train()
        
        pbar = tqdm(train_sampler, desc=f"epoch {epoch}", leave=False)
        epoch_losses = []
        epoch_inter = []
        
        # 启用cudnn benchmark
        torch.backends.cudnn.benchmark = True
        
        batch_times = []
        gpu_times = []
        
        for batch_idx, _ in enumerate(pbar):
            # 计时：数据加载
            t_data_start = time.perf_counter()
            
            # 获取数据（使用迭代器避免重复创建开销）
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            x = batch["eeg"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            trials = batch["trials"]
            
            t_data_end = time.perf_counter()
            t_data = t_data_end - t_data_start
            
            # 计时：GPU计算
            t_gpu_start = time.perf_counter()
            
            # 前向传播
            with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                eeg_z = eeg(x)
                
                if train_llm:
                    # 训练模式：实时计算文本嵌入
                    text_z = text(batch["texts"])
                else:
                    # 推理模式：使用缓存的文本嵌入
                    emb_indices = [trial_to_emb_idx[t.item()] for t in trials]
                    text_z = text_embeddings_gpu[emb_indices]
                
                loss_dict = criterion(eeg_z, text_z, y)
                loss = loss_dict["loss"]
            
            # 反向传播
            opt.zero_grad(set_to_none=True)  # 更高效的梯度清零
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                
                clip_params = [p for p in eeg.parameters() if p.requires_grad]
                if train_llm:
                    clip_params.extend([p for p in text.parameters() if p.requires_grad])
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
                
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                
                clip_params = [p for p in eeg.parameters() if p.requires_grad]
                if train_llm:
                    clip_params.extend([p for p in text.parameters() if p.requires_grad])
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
                
                opt.step()
            
            t_gpu_end = time.perf_counter()
            t_gpu = t_gpu_end - t_gpu_start
            
            total_train_time += t_data + t_gpu
            batch_times.append(t_data)
            gpu_times.append(t_gpu)
            
            # 延迟更新进度条（减少同步开销）
            epoch_losses.append(loss.item())
            epoch_inter.append(loss_dict["inter"].item())
            
            if (batch_idx + 1) % log_interval == 0:
                avg_batch_time = sum(batch_times[-log_interval:]) / log_interval
                avg_gpu_time = sum(gpu_times[-log_interval:]) / log_interval
                gpu_util = gpu_monitor.get_utilization()
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "inter": f"{loss_dict['inter'].item():.4f}",
                    "batch_time": f"{avg_batch_time*1000:.1f}ms",
                    "gpu_time": f"{avg_gpu_time*1000:.1f}ms",
                    "gpu%": f"{gpu_util:.0f}%" if gpu_util >= 0 else "N/A",
                })
            
            step += 1
        
        # 数据迭代器重置
        data_iter = iter(train_loader)
        
        sched.step()
        
        # 验证
        metrics = evaluate(eeg, class_z, val_loader, device)
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_inter = sum(epoch_inter) / len(epoch_inter)
        
        # Epoch统计
        epoch_time = sum(batch_times)
        avg_gpu_util = gpu_monitor.get_utilization()
        
        print(f"epoch {epoch}: loss={avg_loss:.4f} inter={avg_inter:.4f} "
              f"acc={metrics['acc']:.4f} f1={metrics['macro_f1']:.4f}")
        print(f"  [Stats] epoch_time={epoch_time:.1f}s, avg_batch={epoch_time/len(train_sampler)*1000:.1f}ms, "
              f"avg_gpu_util={avg_gpu_util:.1f}%" if avg_gpu_util >= 0 else "")
        
        gpu_monitor.log_memory(f"After epoch {epoch}")
        
        # 保存检查点
        if metrics["macro_f1"] >= best_metric:
            best_metric = metrics["macro_f1"]
            save_ckpt(out_dir / "best.pt", eeg, text, opt, sched, epoch, step, best_metric, cfg)
            print(f"  -> saved best model (f1={best_metric:.4f})")

        save_ckpt(ckpt_path, eeg, text, opt, sched, epoch, step, best_metric, cfg)

    print(f"\n[Train] Complete. best_metric={best_metric:.4f}")
    print(f"[Stats] Total training time: {total_train_time:.1f}s")


if __name__ == "__main__":
    main()