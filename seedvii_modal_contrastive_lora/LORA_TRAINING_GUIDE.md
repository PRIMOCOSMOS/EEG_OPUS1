# SEED-VII EEG-LLM 对比学习库 - LoRA 训练完整指南

## 🔑 核心设计：LoRA微调不可妥协

本库实现 **EEG编码器 + LoRA微调LLM** 的双塔对比学习架构，其中 **LoRA训练是方案中的关键环节**。

### 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dual-Tower Contrastive Learning              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │    EEG Tower     │         │    LLM Tower     │             │
│  │  (EEGNetEncoder) │         │  (Qwen2.5-0.5B   │             │
│  │                  │   Sim   │   + LoRA Adapter)│             │
│  │  trainable       │◄────────►                 │             │
│  │  (full backprop) │         │  trainable       │             │
│  └────────┬─────────┘         └────────┬─────────┘             │
│           │                            │                       │
│           │   ┌────────────────────────┘                       │
│           │   │                                                │
│           ▼   ▼                                                │
│  ┌─────────────────────────────────────────┐                  │
│  │      TriContrastiveLoss                 │                  │
│  │  - inter_modal: EEG ↔ LLM               │                  │
│  │  - intra_modal_eeg: EEG → EEG           │                  │
│  │  - intra_modal_llm: LLM → LLM           │                  │
│  └─────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### LoRA训练的关键特性

| 组件 | 说明 |
|------|------|
| **Base LLM** | Qwen2.5-0.5B-Instruct, 参数冻结 |
| **LoRA适配器** | q_proj, k_proj, v_proj, o_proj (r=8, alpha=16) |
| **训练参数** | 仅LoRA层 (~1.2M params), 约占总参数的0.24% |
| **投影层** | Linear(hidden→128), 可选择训练 |

---

## 🚀 使用方法

### 模式一：快速推理模式（不训练LLM）

```bash
python -m seedvii_contrastive.scripts.train_contrastive \
  --config configs/modelscope_default.yaml
```

- LLM塔完全冻结，使用预计算文本嵌入
- 仅训练EEG编码器
- 训练速度：~1-2秒/batch

### 模式二：完整训练模式（训练LoRA + EEG）

```bash
python -m seedvii_contrastive.scripts.train_contrastive \
  --config configs/modelscope_default.yaml \
  --train-llm
```

- **LoRA参数解冻**，通过反向传播更新
- **EEG编码器** 正常训练
- **投影层** 参与训练
- 训练速度：~30秒/batch（取决于GPU）

---

## 📋 LoRA训练流程详解

### 1. 模型初始化 (`llm_tower.py`)

```python
class LoRATextTower(nn.Module):
    def __init__(self, ...):
        # 1. 加载Base LLM
        base = AutoModelForCausalLM.from_pretrained(...)
        
        # 2. 应用LoRA适配器
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.llm = get_peft_model(base, lora_cfg)
        
        # 3. 添加投影层
        self.proj = nn.Linear(hidden_size, embed_dim)
```

### 2. 参数管理 (`llm_tower.py`)

```python
def unfreeze_lora(self):
    """解冻LoRA参数用于训练"""
    for name, param in self.llm.named_parameters():
        if 'lora_' in name.lower():
            param.requires_grad = True

def get_lora_parameters(self) -> List[nn.Parameter]:
    """获取所有LoRA参数（用于优化器）"""
    return [p for n, p in self.llm.named_parameters() if 'lora_' in n]
```

### 3. 优化器配置 (`train_contrastive.py`)

```python
def build_optimizer(cfg, eeg, text, train_llm):
    param_groups = [
        {"params": [p for p in eeg.parameters() if p.requires_grad], "lr": lr_eeg},
    ]
    
    if train_llm:
        # LoRA参数单独学习率
        param_groups.append({"params": text.get_lora_parameters(), "lr": lr_llm})
        # 投影层可选训练
        param_groups.append({"params": list(text.proj.parameters()), "lr": lr_proj})
    
    return torch.optim.AdamW(param_groups)
```

### 4. 训练循环中的双塔前向

```python
for batch in dataloader:
    if train_llm:
        # 实时通过LLM计算文本嵌入（支持梯度）
        eeg_z = eeg(x)                    # EEG塔前向
        text_z = text(text_batch)         # LLM塔前向（带LoRA梯度）
    else:
        # 使用预计算嵌入
        text_z = cached_embeddings[indices]
    
    loss = criterion(eeg_z, text_z, y)
    loss.backward()  # 同时更新两个塔
```

---

## ⚙️ 配置文件说明

```yaml
model:
  embed_dim: 128
  llm:
    lora_r: 8                    # LoRA秩
    lora_alpha: 16               # LoRA缩放因子
    lora_dropout: 0.05           # Dropout
    target_modules: [q_proj, k_proj, v_proj, o_proj]  # 目标层
    gradient_checkpointing: true # 显存优化

train:
  batch_size: 96
  lr_eeg: 0.001         # EEG学习率
  lr_llm: 0.00005       # LoRA学习率
  lr_proj: 0.0002       # 投影层学习率
```

---

## 🔍 验证LoRA训练

### 检查可训练参数

```python
print(text.trainable_parameters_report())
# 输出: trainable=1,196,160 / total=495,228,928 (0.2415%)
```

### 验证梯度流

```python
# 检查LoRA参数是否有梯度
for name, param in text.llm.named_parameters():
    if 'lora_' in name and param.requires_grad:
        print(f"{name}: requires_grad={param.requires_grad}, grad={param.grad is not None}")
```

### 检查两塔同时训练

训练日志应显示：
```
[LLM Tower] LoRA trainable: 1,196,160 / total: 495,228,928 (0.2415%)
[EEG Net] parameters: 55,232
Training: epochs=50, steps_per_epoch=300
    EEG trainable: 55,232
    LLM LoRA trainable: 1,196,160
```

---

## 🐛 常见问题排查

### 1. LoRA参数未更新

检查：
- `train_llm=True` 已设置
- `text.unfreeze_lora()` 已被调用
- 优化器包含LoRA参数组

### 2. dtype不匹配错误

检查：
- EEG输出: float32
- LLM输出: 与base model dtype一致
- `losses.py` 中的 `_ensure_dtype()` 已处理

### 3. 显存不足

解决方案：
- 启用 `gradient_checkpointing: true`
- 减小 `batch_size`
- 使用 `--train-llm` 时减少 `num_workers`

---

## 📁 文件结构

```
seedvii_modal_contrastive_lora/
├── seedvii_contrastive/
│   ├── models/
│   │   ├── llm_tower.py      # LoRA塔实现（关键）
│   │   └── eegnet.py         # EEG编码器
│   ├── scripts/
│   │   └── train_contrastive.py  # 训练脚本（关键）
│   ├── losses.py             # 对比损失
│   ├── metrics.py            # 评估指标
│   └── data/
│       └── dataset.py        # 数据加载
├── configs/
│   └── modelscope_default.yaml
├── requirements.txt
└── README.md
```

---

## ✅ 完整检查清单

- [x] LoRA适配器正确应用到Qwen模型
- [x] `unfreeze_lora()` 方法解冻LoRA参数
- [x] `get_lora_parameters()` 获取LoRA参数列表
- [x] 优化器正确配置LoRA参数组
- [x] 训练时两塔同时前向传播
- [x] 梯度正确反向传播到两塔
- [x] dtype一致性处理（BFloat16/float32）
- [x] 梯度裁剪和混合精度支持
- [x] 检查点保存/恢复正确处理LoRA状态