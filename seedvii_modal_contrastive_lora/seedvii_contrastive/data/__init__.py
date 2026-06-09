"""Data package.

NOTE (fix): imports are made *lazy* so that the preprocessing pipeline
(``preprocess.py`` / ``preprocess_npz``) can run WITHOUT importing ``torch``.

Previously this module eagerly imported ``dataset.py`` which does
``import torch``.  That made ``preprocess_npz`` fail/stall on machines where
torch was not yet installed (or was still installing), which is one cause of the
"NPZ preprocessing cell hangs / dies" symptom.  Preprocessing only needs numpy +
scipy + h5py, so we defer the torch-dependent imports until they are actually
requested.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Lightweight, torch-free imports are safe to do eagerly.
from seedvii_contrastive.data.protocol import (
    load_l2_text_protocol,
    trial_to_labels,
    VALENCE_NAMES,
    TRIAL_FINE,
)

# Names that live in dataset.py (which imports torch) -> resolved lazily.
_LAZY = {
    "load_index": "seedvii_contrastive.data.dataset",
    "split_index_by_subjects": "seedvii_contrastive.data.dataset",
    "fit_channel_stats": "seedvii_contrastive.data.dataset",
    "WindowNpzDataset": "seedvii_contrastive.data.dataset",
    "ClassBalancedBatchSampler": "seedvii_contrastive.data.dataset",
    "build_l2_text_bank": "seedvii_contrastive.data.dataset",
}

__all__ = [
    "load_index",
    "split_index_by_subjects",
    "fit_channel_stats",
    "WindowNpzDataset",
    "ClassBalancedBatchSampler",
    "build_l2_text_bank",
    "load_l2_text_protocol",
    "trial_to_labels",
    "VALENCE_NAMES",
    "TRIAL_FINE",
]

if TYPE_CHECKING:  # only for type checkers / IDEs, never at runtime
    from seedvii_contrastive.data.dataset import (  # noqa: F401
        load_index,
        split_index_by_subjects,
        fit_channel_stats,
        WindowNpzDataset,
        ClassBalancedBatchSampler,
        build_l2_text_bank,
    )


def __getattr__(name: str):
    """PEP 562 lazy attribute loading for torch-dependent symbols."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    return getattr(module, name)
