# SEED-VII EEGNet × LoRA-LLM 三分类对比学习库

本库按新要求重写：借鉴 Brain-CLIPLM 的跨模态对比学习框架，但适配 **ModelScope 数据集 `DEREKVERSE/SEED-VII` 根目录中的 `1-20.mat` HDF5 文件**，并且 **LLM Tower 只使用你参考库 CSV 协议里的 L2 文本**。

## 关键修正

1. **ModelScope 数据拉取**  
   使用 `dataset_snapshot_download` / `repo_type='dataset'` 的数据集协议，不使用模型协议。脚本会先 listing/search 数据集文件，筛选 `1.mat`...`20.mat`，并下载根目录/子目录下所有 CSV，再自动识别含 `trial` + `l2_text` 的协议 CSV。

2. **H5 真实格式**  
   每个 subject 文件内部 key 为 `1`...`80`，每个 key 对应 MATLAB 变量尺寸 `62×N double`。读取器兼容 h5py 暴露为 `(62,N)` 或 `(N,62)`，统一返回 `(62,N)`。

3. **L2 文本监督**  
   LLM Tower 输入不是粗糙 label prompt，而是 Brain-CLIPLM 风格 `text_protocol.csv` 中每个 trial 的 `l2_text`。80 clips 对应 80 条描述文本。  
   训练时正负样本关系只由聚合后三分类标签决定：同类为正、异类为负，不看 subject，也不看 video id。

## 标签聚合

细粒度标签保持参考库一致：

```text
neutral, joy, sadness, fear, disgust, anger, surprise
```

三分类聚合：

```text
negative: sadness, fear, disgust, anger
neutral : neutral
positive: joy, surprise
```

## 模型

### EEG Tower

经典小型 EEGNet：

```text
Input [B,1,62,800]
Temporal Conv → Spatial Depthwise Conv → Separable Conv → Pool → Linear → L2 norm
```

### LLM Tower

```text
L2 trial text → AutoModelForCausalLM + LoRA → mean pooling → Linear → L2 norm
```

默认推荐小模型：`Qwen/Qwen2.5-0.5B-Instruct`。

## 损失函数

```text
Loss = L_inter + β1 * L_eeg_intra + β2 * L_llm_intra
β1 + β2 = 1
```

默认：

```yaml
beta_eeg: 0.65
beta_llm: 0.35
```

三项均为 label-based multi-positive supervised contrastive loss：

- `L_inter`: EEG ↔ L2 text 跨模态；
- `L_eeg_intra`: EEG 内模态，同三分类标签为正；
- `L_llm_intra`: L2 text 内模态，同三分类标签为正。

## 主要入口

### ModelScope Notebook

```text
notebooks/modelscope_pipeline.ipynb
```

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
  --output-dir /mnt/workspace/seedvii_npz \
  --subjects 1-20 \
  --window-sec 4 --stride-sec 4 \
  --center-ratio 0.60 \
  --max-windows-per-clip 12
```

### 训练

```bash
python -m seedvii_contrastive.scripts.train_contrastive \
  --config configs/modelscope_default.yaml
```

### 推理/编码

```bash
python -m seedvii_contrastive.scripts.encode_eeg \
  --config configs/modelscope_default.yaml \
  --checkpoint /mnt/workspace/seedvii_contrastive_runs/run_valence3/best.pt \
  --split val \
  --out /mnt/workspace/seedvii_embeddings_val.npz
```

## 文件结构

```text
seedvii_contrastive/
  data/
    protocol.py       # SEED-VII session 标签协议 + L2 CSV loader
    h5io.py           # HDF5 .mat 读取，兼容 62×N / N×62
    windowing.py      # 中间60%裁剪 + 4s window
    preprocess.py     # 写 NPZ shards
    dataset.py        # NPZ Dataset + L2 text + 均衡 sampler
  models/
    eegnet.py         # EEGNet encoder
    llm_tower.py      # LoRA LLM text tower
  losses.py           # L_inter + beta_eeg L_eeg + beta_llm L_llm
  scripts/
    download_modelscope_seedvii.py  # 支持 1-20.mat 和 CSV 均在数据集根目录
    preprocess_npz.py
    train_contrastive.py
    encode_eeg.py
    merge_upload_modelscope_zip.py
```
