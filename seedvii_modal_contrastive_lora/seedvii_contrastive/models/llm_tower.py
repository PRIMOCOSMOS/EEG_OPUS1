from __future__ import annotations

from typing import List, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRATextTower(nn.Module):
    """LoRA-tunable LLM text tower.

    It loads a causal LLM, attaches LoRA adapters, mean-pools the last hidden states,
    projects them to the shared contrastive dimension, and L2-normalizes the output.
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
        torch_dtype: Optional[torch.dtype] = None,  # FIX: Add explicit dtype control
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

        # FIX: Explicitly control dtype to avoid mixed precision issues
        if torch_dtype is None:
            torch_dtype = torch.float32  # Default to float32 for stability

        load_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        
        # Only add output_hidden_states if supported (suppress warning)
        try:
            load_kwargs["output_hidden_states"] = True
        except Exception:
            pass
            
        # Add torch_dtype for consistent dtype handling
        load_kwargs["torch_dtype"] = torch_dtype

        base = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **load_kwargs
        )

        if gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()

        if target_modules is None:
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

        # Get dtype and device from base model for consistent projection
        base_model = self.llm.get_base_model()
        sample_param = next(base_model.parameters(), None)
        if sample_param is not None:
            llm_dtype = sample_param.dtype
            llm_device = sample_param.device
        else:
            llm_dtype = torch.float32
            llm_device = next(self.llm.parameters()).device if len(list(self.llm.parameters())) > 0 else torch.device("cpu")

        hidden = base_model.config.hidden_size
        self.proj = nn.Linear(hidden, embed_dim)
        
        # FIX: Cast projection to match LLM dtype for compatibility
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
        """Forward pass returning L2-normalized text embeddings."""
        device = self.proj.weight.device
        tok = self._tokenize(texts, device)
        
        # FIX: Suppress warnings and handle different model output formats
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.llm(**tok, output_hidden_states=True, use_cache=False)
        
        # Handle different output formats from various transformers versions
        # Some models use .hidden_states, others use .last_hidden_state
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            h = out.hidden_states[-1]
        elif hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
            h = out.last_hidden_state
        else:
            raise RuntimeError(
                f"Model output does not have hidden_states or last_hidden_state. "
                f"Available attributes: {dir(out)}"
            )

        # Ensure mask has the same dtype as hidden states
        mask = tok["attention_mask"].unsqueeze(-1).to(dtype=h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        # FIX: Cast pooled to match proj weight dtype
        pooled = pooled.to(dtype=self.proj.weight.dtype)
        z = self.proj(pooled)

        return F.normalize(z, dim=-1)
