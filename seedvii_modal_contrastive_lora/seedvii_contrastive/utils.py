from __future__ import annotations

from pathlib import Path
import json
import random
import time
import numpy as np
import torch
import yaml


def load_yaml(path):
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    """Save object as JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> str:
    """Resolve device string to actual device."""
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def now() -> str:
    """Get current timestamp string."""
    return time.strftime("%Y-%m-%d %H:%M:%S")
