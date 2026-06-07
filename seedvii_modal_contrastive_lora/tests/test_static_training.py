from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "seedvii_contrastive" / "scripts" / "train_contrastive.py"


def _contains_scaler_call(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            value = child.func.value
            if isinstance(value, ast.Name) and value.id == "scaler":
                return True
    return False


def test_bf16_amp_does_not_use_grad_scaler_unconditionally():
    """BF16 AMP must not enter GradScaler path based only on use_amp=True.

    CUDA + BF16 autocast is valid, but GradScaler is for FP16.  Calling
    scaler.unscale_ on BF16 gradients raises a CUDA NotImplementedError on some
    PyTorch builds.  Any scaler operation must be guarded by ``scaler is not None``.
    """
    src = TRAIN_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)

    guarded = []
    unguarded = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _contains_scaler_call(node):
            cond = ast.get_source_segment(src, node.test) or ""
            if "scaler is not None" in cond:
                guarded.append(cond)
            else:
                unguarded.append(cond)

    assert guarded, "expected scaler operations to be guarded by 'scaler is not None'"
    assert not unguarded, f"scaler operations have unsafe guards: {unguarded}"
    assert "scaler = None" in src
