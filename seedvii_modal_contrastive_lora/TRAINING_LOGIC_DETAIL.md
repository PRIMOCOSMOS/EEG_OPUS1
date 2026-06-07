# SEED-VII EEG-LLM 对比学习 - 训练逻辑详解

## 📋 完整训练流程

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           完整训练流程                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────┐                                           │
│  │ Stage 1: 初始化              │                                           │
│  │ - 加载配置                   │                                           │
│  │ - 设置随机种子               │                                           │
│  │ - 初始化GPU                 │                                           │
│  └──────────────┬───────────────┘                                           │
│                 │                                                           │
│                 ▼                                                           │
│  ┌──────────────────────────────┐                                           │
│  │ Stage 2: 数据加载            │                                           │
│  │ - 加载NPZ索引               │ ← 15008训练样本, 3752验证样本             │
│  │ - 划分train/val             │                                           │
│  │ - 计算通道统计量             │                                           │
│  │ - 构建Dataset               │                                           │
│  └──────────────┬───────────────┘                                           │
│                 │                                                           │
│                 ▼                                                           │
│  ┌──────────────────────────────┐                                           │
│  │ Stage 3: 模型构建            │                                           │
│  │ - 构建EEGNetEncoder         │ ← 可训练参数: 55,232                     │
│  │ - 构建LoRATextTower         │ ← LoRA参数: 1,196,160 (~0.24%)           │
│  │ - 配置LoRA训练模式           │                                           │
│  │ - GPU预热                   │                                           │
│  └──────────────┬───────────────┘                                           │
│                 │                                                           │
│                 ▼                                                           │
│  ┌──────────────────────────────┐                                           │
│  │ Stage 4: 预计算文本嵌入      │                                           │
│  │ - 编码80个trial文本         │ ← 一次性计算，O(1)查找                   │
│  │ - 构建类别原型向量           │ ← 3个类别中心                           │
│  └──────────────┬───────────────┘                                           │
│                 │                                                           │
│                 ▼                                                           │
│  ┌──────────────────────────────┐                                           │
│  │ Stage 5: 训练循环            │ ← 主要计算阶段                          │
│  │ ┌────────────────────────┐  │                                           │
│  │ │ per epoch:             │  │                                           │
│  │ │   - 设置模型train模式  │  │                                           │
│  │ │   ┌──────────────────┐ │  │                                           │
│  │ │   │ per batch:       │ │  │                                           │
│  │ │   │   1. DataLoader  │ │  │ ← pin_memory, prefetch_factor=2         │
│  │ │   │   2. EEG forward │ │  │ ← GPU计算                               │
│  │ │   │   3. Text lookup │ │  │ ← 缓存查找 or LLM forward              │
│  │ │   │   4. Loss计算    │ │  │ ← inter + intra_eeg + intra_llm        │
│  │ │   │   5. backward    │ │  │ ← 梯度计算                              │
│  │ │   │   6. optimizer   │ │  │ ← 参数更新                              │
│  │ │   └──────────────────┘ │  │                                           │
│  │ └────────────────────────┘  │                                           │
│  │   - 验证集评估             │                                           │
│  │   - 保存检查点             │                                           │
│  └──────────────┬───────────────┘                                           │
│                 │                                                           │
│                 ▼                                                           │
│  ┌──────────────────────────────┐                                           │
│  │ Stage 6: 完成                │                                           │
│  │ - 输出最佳验证指标           │                                           │
│  │ - 保存最终模型               │                                           │
│  └──────────────────────────────┘                                           │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 🎯 双塔对比学习架构

### 1. 模型结构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Dual-Tower Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐                    ┌─────────────┐            │
│   │  EEG Tower  │                    │  LLM Tower  │            │
│   │             │                    │             │            │
│   │  (可训练)    │      Sim()        │  (LoRA可训练) │            │
│   │             │ ◄─────────────────► │             │            │
│   │  embed_dim  │                    │  embed_dim  │            │
│   └──────┬──────┘                    └──────┬──────┘            │
│          │                                 │                    │
│          │    ┌────────────────────────────┘                    │
│          │    │                                                  │
│          ▼    ▼                                                  │
│   ┌──────────────────────────────────────┐                     │
│   │        TriContrastiveLoss            │                     │
│   │                                      │                     │
│   │  L_inter = EEG↔LLM (跨模态)          │                     │
│   │  L_eeg   = EEG→EEG (模态内)          │                     │
│   │  L_llm   = LLM→LLM (模态内)          │                     │
│   │                                      │                     │
│   │  L_total = L_inter + β1*L_eeg + β2*L_llm                 │
│   └──────────────────────────────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 训练模式对比

| 模式 | `--train-llm` | LoRA参数 | 文本处理 | 速度 |
|------|---------------|----------|----------|------|
| 推理模式 | False | 冻结 | 缓存查找 | 快 (~1-2s/batch) |
| 训练模式 | True | 可训练 | 实时计算 | 慢 (~30s/batch) |

## 🔧 训练循环详解

### 核心代码流程

```python
# ========== 初始化 ==========
eeg, text = build_models(cfg, device, train_llm=train_llm)
text_embeddings = precompute_text_embeddings(text, bank_texts, device)  # 缓存
opt = build_optimizer(cfg, eeg, text, train_llm=train_llm)
scaler = GradScaler()  # 混合精度

# ========== 训练循环 ==========
for epoch in range(epochs):
    eeg.train()
    text.train()  # 如果train_llm
    
    for batch in train_loader:
        # 1. 数据获取
        x, y = batch["eeg"].to(device), batch["label"].to(device)
        trials = batch["trials"]
        
        # 2. 前向传播
        with autocast(dtype=torch.bfloat16):
            eeg_z = eeg(x)  # EEG塔
            
            if train_llm:
                text_z = text(batch["texts"])  # LLM塔（实时）
            else:
                text_z = text_embeddings[trials]  # 缓存查找
            
            loss_dict = criterion(eeg_z, text_z, y)
            loss = loss_dict["loss"]
        
        # 3. 反向传播
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        clip_grad_norm_(params, max_norm=1.0)
        scaler.step(opt)
        scaler.update()
```

### 损失函数计算

```python
def TriContrastiveLoss.forward(eeg_z, text_z, labels):
    # 1. 跨模态对比 (EEG ↔ LLM)
    L_inter = 0.5 * (
        supcon(eeg_z, text_z, labels, labels) +
        supcon(text_z, eeg_z, labels, labels)
    )
    
    # 2. EEG模态内对比
    L_eeg = supcon(eeg_z, eeg_z, labels, labels, exclude_self=True)
    
    # 3. LLM模态内对比
    L_llm = supcon(text_z, text_z, labels, labels, exclude_self=True)
    
    # 4. 总损失
    return L_inter + 0.65*L_eeg + 0.35*L_llm
```

## 📊 GPU利用率优化

### 瓶颈分析

| 瓶颈 | 原因 | 解决方案 |
|------|------|----------|
| DataLoader慢 | CPU-GPU传输阻塞 | `pin_memory=True`, `persistent_workers=True` |
| Worker启动慢 | 每epoch重新创建 | `persistent_workers=True` |
| GPU空闲 | CPU处理瓶颈 | `prefetch_factor=2`, 异步加载 |
| 同步开销 | `.item()` 调用 | 延迟打印，按interval更新 |
| LLM推理慢 | 实时计算 | 推理模式使用缓存 |
| 首次计算慢 | CUDA初始化 | GPU预热 |

### 优化后的DataLoader配置

```python
DataLoader(
    dataset,
    batch_size=96,
    num_workers=4,
    pin_memory=True,           # CPU→GPU DMA传输
    prefetch_factor=2,         # 每个worker预加载2个batch
    persistent_workers=True,   # 保持worker进程
    collate_fn=collate_fn,
)
```

### 混合精度配置

```python
# 使用BF16加速训练
scaler = GradScaler()

with autocast(enabled=True, dtype=torch.bfloat16):
    eeg_z = eeg(x)           # EEG塔：BF16 → FP32
    text_z = text(texts)     # LLM塔：BF16
    loss = criterion(eeg_z, text_z, y)  # 自动转换FP32

scaler.scale(loss).backward()  # BF16梯度
scaler.step(opt)               # 梯度缩放
```

## 📈 预期性能

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| batch时间 | ~39s | ~5-10s (推理模式) |
| GPU利用率 | <50% | >80% |
| 内存占用 | 高 | 优化 (梯度检查点) |

## 🔍 验证LoRA训练

### 检查点1: 模型参数

```python
# EEG塔
print(f"EEG params: {sum(p.numel() for p in eeg.parameters() if p.requires_grad):,}")

# LoRA塔
print(f"LoRA params: {sum(p.numel() for p in text.get_lora_parameters()):,}")
print(text.trainable_parameters_report())
```

### 检查点2: 梯度流

```python
# 训练一个step后检查梯度
for name, param in text.llm.named_parameters():
    if 'lora_' in name and param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.6f}")
```

### 检查点3: 损失下降

```python
# 观察日志
# epoch 0: loss=9.0651, inter=4.5898, acc=0.35, f1=0.28
# epoch 10: loss=3.5210, inter=2.1012, acc=0.62, f1=0.58
```

## ⚠️ 注意事项

1. **LoRA训练是必须的**：用户要求LoRA不可妥协，确保`--train-llm`开启
2. **BF16 vs FP32**：LLM使用BF16加速，损失函数自动转换为FP32
3. **缓存vs实时**：推理模式用缓存，训练模式实时计算
4. **GPU预热**：首次GPU计算较慢，需要预热