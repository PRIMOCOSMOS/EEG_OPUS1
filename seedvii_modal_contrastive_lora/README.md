# SEED-VII EEGNet x LoRA-LLM 三分类对比学习库

本库按新要求重写：借鉴 Brain-CLIPLM 的跨模态对比学习框架，但适配 **ModelScope 数据集 `DEREKVERSE/SEED-VII` 根目录中的 `1-20.mat` HDF5 文件**，并且 **LLM Tower 只使用你参考库 CSV 协议里的 L2 文本**。

## 关键修正

1. **ModelScope 数据拉取**
   使用 `dataset_snapshot_download` / `repo_type='dataset'` 的数据集协议。

2. **H5 真实格式**
   每个 subject 文件内部 key 为 `1`...`80`，每个 key 对应 MATLAB 变量尺寸 `62xN double`。

3. **L2 文本监督**
   LLM Tower 输入不是粗糙 label prompt，而是 Brain-CLIPLM 风格 `text_protocol.csv` 中每个 trial 的 `l2_text`。

## 标签聚合

细粒度标签保持参考库一致：
```
neutral, joy, sadness, fear, disgust, anger, surprise
```

三分类聚合：
```
negative: sadness, fear, disgust, anger
neutral : neutral
positive: joy, surprise
```

## 模型

### EEG Tower
经典小型 EEGNet：
```
Input [B,1,62,800]
Temporal Conv -> Spatial Depthwise Conv -> Separable Conv -> Pool -> Linear -> L2 norm
```

### LLM Tower
```
L2 trial text -> AutoModelForCausalLM + LoRA -> mean pooling -> Linear -> L2 norm
```

默认推荐小模型：`Qwen/Qwen2.5-0.5B-Instruct`。

## 损失函数

```
Loss = L_inter + beta1 * L_eeg_intra + beta2 * L_llm_intra
beta1 + beta2 = 1
```

默认：beta_eeg=0.65, beta_llm=0.35

## 主要入口

### 下载 ModelScope 数据
```bash
python -m seedvii_contrastive.scripts.download_modelscope_seedvii \
  --dataset-id DEREKVERSE/SEED-VII \
  --local-dir /mnt/workspace/seedvii_ms_dataset
```

### NPZ 预处理
```bash
python -m seedvii_contrastive.scripts.preprocess_npz \
  --input-root /mnt/workspace/seedvii_ms_dataset \
  --output-dir /mnt/workspace/seedvii_npz
```

### 训练
```bash
python -m seedvii_contrastive.scripts.train_contrastive \
  --config configs/modelscope_default.yaml
```

## 问题修复

### dtype 不匹配问题

修复了 `RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float` 错误。

该错误发生在 `llm_tower.py` 的 `forward` 方法中，当 LLM 模型以 BFloat16 运行时，其输出的隐藏状态为 BFloat16，但投影层 `self.proj` 默认创建为 Float32，导致矩阵乘法时数据类型不匹配。

修复方法：
1. 在初始化时，将 `self.proj` 转换为与 LLM 模型相同的 dtype
2. 在 `forward` 中，确保 `pooled` 张量与 `proj` 权重的数据类型一致

关键代码修改在 `llm_tower.py`:
```python
# 获取 LLM 模型的 dtype 和 device
sample_param = next(base_model.parameters(), None)
if sample_param is not None:
    llm_dtype = sample_param.dtype
    llm_device = sample_param.device
else:
    llm_dtype = torch.float32
    llm_device = next(self.llm.parameters()).device if len(list(self.llm.parameters())) > 0 else torch.device("cpu")

# 将 proj 层转换为与 LLM 模型相同的 dtype
self.proj = self.proj.to(dtype=llm_dtype, device=llm_device)
```
