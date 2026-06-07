"""
LoRA-enabled LLM Text Tower for EEG-LLM Contrastive Learning

关键特性：
- LoRA适配器高效微调Qwen模型
- 支持BF16/FP32混合精度
- 优化tokenization和前向传播
"""
from __future__ import annotations

from typing import List, Optional, Dict
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class LoRATextTower(nn.Module):
    """LoRA-tunable LLM text tower with optimized inference and training.
    
    架构:
    - Base LLM: Qwen2.5-0.5B (冻结)
    - LoRA Adapter: q_proj, k_proj, v_proj, o_proj (可训练)
    - Projection: hidden_size → embed_dim
    """
    
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def __init__(
        self,
        model_name_or_path: str,
        embed_dim: int = 128,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        trust_remote_code: bool = True,
        max_length: int = 64,
        gradient_checkpointing: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, TaskType, get_peft_model

        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Tokenizer初始化
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 默认使用BF16以加速训练
        if dtype is None:
            dtype = torch.bfloat16
        self._dtype = dtype
        
        # 加载基础模型。新版 transformers 推荐 dtype=，旧版仍使用 torch_dtype=；
        # 这里做兼容，避免训练日志里出现无意义的 deprecation warning。
        try:
            base = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
            )
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=dtype,
            )

        self.hidden_size = base.config.hidden_size
        
        # 梯度检查点（节省显存）
        if gradient_checkpointing:
            print("[LLM Tower] Enabling gradient checkpointing...")
            if hasattr(base, "gradient_checkpointing_enable"):
                base.gradient_checkpointing_enable()
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()

        # LoRA配置
        if target_modules is None:
            target_modules = self.LORA_TARGET_MODULES

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        self.llm = get_peft_model(base, lora_cfg)
        self._print_trainable_params()

        # 投影层（与模型dtype保持一致）
        self.proj = nn.Linear(self.hidden_size, embed_dim)
        self._sync_proj_with_llm()

    def _print_trainable_params(self) -> None:
        """打印可训练参数统计"""
        total = sum(p.numel() for p in self.llm.parameters())
        trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        print(f"[LLM Tower] LoRA trainable: {trainable:,} / total: {total:,} ({trainable/max(total,1):.4%})")

    def _sync_proj_with_llm(self) -> None:
        """同步proj层dtype和设备"""
        base_model = self.llm.get_base_model()
        sample_param = next(base_model.parameters(), None)
        
        if sample_param is not None:
            llm_dtype = sample_param.dtype
            llm_device = sample_param.device
        else:
            llm_dtype = torch.float32
            llm_device = next(self.llm.parameters()).device
        
        self.proj = self.proj.to(dtype=llm_dtype, device=llm_device)
        self._llm_dtype = llm_dtype
        self._llm_device = llm_device

    def unfreeze_lora(self) -> None:
        """解冻LoRA参数用于训练"""
        print("[LLM Tower] Unfreezing LoRA parameters...")
        
        for name, param in self.llm.named_parameters():
            if 'lora_' in name.lower():
                param.requires_grad = True
        
        for param in self.proj.parameters():
            param.requires_grad = True
        
        self._print_trainable_params()

    def freeze_all_except_lora(self) -> None:
        """Freeze the whole text tower for cached-embedding/inference mode.

        The historical method name is kept for compatibility, but inference mode
        must not leave LoRA tensors trainable; otherwise users see misleading
        trainable-parameter reports and may accidentally allocate gradients.
        """
        print("[LLM Tower] Freezing all text-tower parameters...")
        
        for _, param in self.llm.named_parameters():
            param.requires_grad = False
        
        for param in self.proj.parameters():
            param.requires_grad = False
        
        self._print_trainable_params()

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """获取LoRA参数列表"""
        return [p for n, p in self.llm.named_parameters() if 'lora_' in n.lower()]

    def trainable_parameters_report(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"trainable={trainable:,} / total={total:,} ({trainable / max(total, 1):.4%})"

    @torch.compile(mode="reduce-overhead")
    def _tokenize_batch(self, texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        """编译优化tokenization"""
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in tok.items()}

    def _tokenize(self, texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        """标准tokenization（无编译）"""
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in tok.items()}

    def forward(self, texts: List[str]) -> torch.Tensor:
        """前向传播，返回L2归一化文本嵌入
        
        Args:
            texts: 文本列表
            
        Returns:
            L2归一化embedding, shape (batch_size, embed_dim)
        """
        device = self.proj.weight.device
        tok = self._tokenize(texts, device)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 必须设置output_hidden_states=True才能获取隐藏状态
            out = self.llm(**tok, use_cache=False, output_hidden_states=True)
        
        h = self._extract_hidden_states(out)
        
        # 注意力掩码加权平均池化
        mask = tok["attention_mask"].unsqueeze(-1).to(dtype=h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        
        # 投影
        pooled = pooled.to(dtype=self.proj.weight.dtype)
        z = self.proj(pooled)
        
        return F.normalize(z, dim=-1)
    
    def _extract_hidden_states(self, output) -> torch.Tensor:
        """从模型输出提取隐藏状态
        
        处理Qwen等模型的不同输出格式:
        - 有些模型使用output.hidden_states元组
        - 有些模型使用output.last_hidden_state
        """
        # 方法1: 检查last_hidden_state (最常用)
        if hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
            hs = output.last_hidden_state
            if isinstance(hs, torch.Tensor):
                return hs
        
        # 方法2: 检查hidden_states元组
        if hasattr(output, 'hidden_states'):
            hs = output.hidden_states
            # 如果是特殊对象，尝试转换为元组
            if hasattr(hs, 'to_tuple'):
                hs = hs.to_tuple()
            if hs is not None and len(hs) > 0:
                # hidden_states通常是最后一层
                last_hs = hs[-1]
                if isinstance(last_hs, torch.Tensor):
                    return last_hs
        
        # 方法3: 尝试从past_key_values重建 (部分模型)
        if hasattr(output, 'logits') and output.logits is not None:
            # 如果无法获取hidden_states，至少返回logits前的投影
            # 这是一个fallback方案
            pass
        
        raise RuntimeError(
            f"Cannot extract hidden states from model output. "
            f"Available: {[a for a in dir(output) if not a.startswith('_')]}"
        )

    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """批量编码（用于预计算缓存）"""
        device = self.proj.weight.device
        tok = self._tokenize(texts, device)
        # 必须设置output_hidden_states=True
        out = self.llm(**tok, use_cache=False, output_hidden_states=True)
        h = self._extract_hidden_states(out)
        
        mask = tok["attention_mask"].unsqueeze(-1).to(dtype=h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = pooled.to(dtype=self.proj.weight.dtype)
        z = self.proj(pooled)
        
        return F.normalize(z, dim=-1)
    
    @torch.no_grad()
    def encode_batch_fast(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """快速批量编码（分批处理大文本列表）"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = self.encode_batch(batch)
            all_embeddings.append(emb.float())  # 转为float32
        return torch.cat(all_embeddings, dim=0)
    
    def __repr__(self):
        return (
            f"LoRATextTower(\n"
            f"  model: {self.llm.__class__.__name__},\n"
            f"  embed_dim: {self.embed_dim},\n"
            f"  hidden_size: {self.hidden_size},\n"
            f"  max_length: {self.max_length},\n"
            f"  dtype: {self._dtype},\n"
            f"  {self.trainable_parameters_report()}\n"
            f")"
        )