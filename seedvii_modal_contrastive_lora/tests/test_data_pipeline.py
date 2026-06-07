from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest


def install_fake_torch():
    """The CI used for lightweight data tests may not have torch installed."""
    if "torch" in sys.modules:
        return
    torch = types.ModuleType("torch")
    torch.long = "long"

    class _Dataset:
        pass

    class _Sampler:
        def __class_getitem__(cls, item):
            return cls

    data = types.ModuleType("torch.utils.data")
    data.Dataset = _Dataset
    data.Sampler = _Sampler
    utils = types.ModuleType("torch.utils")
    utils.data = data
    torch.utils = utils
    sys.modules["torch"] = torch
    sys.modules["torch.utils"] = utils
    sys.modules["torch.utils.data"] = data


install_fake_torch()

from seedvii_contrastive.data.dataset import (  # noqa: E402
    ClassBalancedBatchSampler,
    fit_channel_stats,
    load_index,
    split_index_by_subjects,
)


def write_minimal_index(npz_dir: Path) -> None:
    x0 = np.arange(2 * 62 * 4, dtype=np.float32).reshape(2, 62, 4)
    x1 = np.ones((1, 62, 4), dtype=np.float32)
    np.savez_compressed(npz_dir / "shard_000000.npz", x=x0)
    np.savez_compressed(npz_dir / "shard_000001.npz", x=x1)
    pd.DataFrame(
        [
            {"shard": "shard_000000.npz", "idx": "0", "subject": "1", "trial": "1", "label3": "2"},
            {"shard": "shard_000000.npz", "idx": "1", "subject": "2", "trial": "2", "label3": "1"},
            {"shard": "shard_000001.npz", "idx": "0", "subject": "3", "trial": "3", "label3": "0"},
        ]
    ).to_csv(npz_dir / "index.csv", index=False)


def test_load_index_adds_abs_shard_and_keeps_numeric_subjects(tmp_path: Path):
    write_minimal_index(tmp_path)

    df = load_index(tmp_path)

    assert "abs_shard" in df.columns
    assert df["abs_shard"].map(lambda p: Path(p).exists()).all()
    assert np.issubdtype(df["subject"].dtype, np.integer)
    assert np.issubdtype(df["trial"].dtype, np.integer)
    assert np.issubdtype(df["label3"].dtype, np.integer)

    tr, va = split_index_by_subjects(df, train_subjects=[1, 2], val_subjects=[3])
    assert len(tr) == 2
    assert len(va) == 1


def test_fit_channel_stats_rejects_empty_dataframe(tmp_path: Path):
    with pytest.raises(ValueError, match="training dataframe is empty"):
        fit_channel_stats(pd.DataFrame(columns=["abs_shard"]))


def test_balanced_batch_sampler_is_a_batch_sampler():
    sampler = ClassBalancedBatchSampler([0, 0, 1, 1, 2, 2], batch_size=6, steps_per_epoch=2, seed=7)
    batches = list(iter(sampler))
    assert len(batches) == 2
    assert all(len(batch) == 6 for batch in batches)
    assert all(isinstance(batch, list) for batch in batches)


def test_balanced_batch_sampler_rejects_empty_labels():
    with pytest.raises(ValueError, match="empty labels"):
        ClassBalancedBatchSampler([], batch_size=4)
