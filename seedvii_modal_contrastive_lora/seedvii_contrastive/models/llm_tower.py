from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRATextTower(nn.Module):
    """LoRA-tunable LLM text tower.

    It loads a causal LLM, attaches LoRA adapters, mean-pools the last hidden states,
    projects them to the shared contrastive dimension, and L2-normalizes the output.
    Use a small ModelScope/HF causal LM such as Qwen/Qwen2.5-0.5B-Instruct for practicality.
    """

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
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, TaskType, get_peft_model

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            output_hidden_states=True,
        )

        if gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()

        if target_modules is None:
            # Works for Qwen/LLaMA-like attention blocks; PEFT silently requires exact names.
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.llm = get_peft_model(base, lora_cfg)

        # Get the dtype and device from the LLM's base model to ensure consistency
        base_model = self.llm.get_base_model()
        # Get a sample parameter to determine the dtype
        sample_param = next(base_model.parameters(), None)
        if sample_param is not None:
            llm_dtype = sample_param.dtype
            llm_device = sample_param.device
        else:
            llm_dtype = torch.float32
            llm_device = next(self.llm.parameters()).device if len(list(self.llm.parameters())) > 0 else torch.device("cpu")

        hidden = base_model.config.hidden_size
        self.proj = nn.Linear(hidden, embed_dim)

        # FIX: Cast the projection layer to match the LLM model's dtype
        # This prevents the RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
        self.proj = self.proj.to(dtype=llm_dtype, device=llm_device)

    def trainable_parameters_report(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"trainable={trainable:,} / total={total:,} ({trainable / max(total, 1):.4%})"

    def _tokenize(self, texts: List[str], device: torch.device):
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in tok.items()}

    def forward(self, texts: List[str]) -> torch.Tensor:
        device = self.proj.weight.device
        tok = self._tokenize(texts, device)
        out = self.llm(**tok, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1]  # (B,L,H)

        # Ensure mask has the same dtype as hidden states for consistent computation
        mask = tok["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        # Cast pooled to the same dtype as proj weight to ensure compatibility
        pooled = pooled.to(dtype=self.proj.weight.dtype)
        z = self.proj(pooled)

        return F.normalize(z, dim=-1)
