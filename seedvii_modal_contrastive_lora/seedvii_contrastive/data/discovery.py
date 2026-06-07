from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import csv

SUBJECT_FILE_NAMES = {f"{i}.mat" for i in range(1, 21)} | {f"{i:02d}.mat" for i in range(1, 21)}


def _is_l2_text_csv(path: Path) -> bool:
    """Return True if CSV looks like Brain-CLIPLM text protocol."""
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return False
            lower = {c.lower().strip() for c in reader.fieldnames}
            has_trial = bool(lower & {"trial", "video_id", "videoid", "id"})
            has_text = bool(lower & {"l2_text", "text", "sentence", "description"})
            return has_trial and has_text
    except Exception:
        return False


def find_eeg_root(root: str | Path) -> Optional[Path]:
    """Find the directory containing SEED-VII subject files 1-20.mat."""
    root = Path(root)
    if not root.exists():
        return None
    mat_files = [p for p in root.rglob("*.mat") if p.name in SUBJECT_FILE_NAMES]
    if not mat_files:
        return None
    counts = {}
    for p in mat_files:
        counts[p.parent] = counts.get(p.parent, 0) + 1
    return max(counts, key=counts.get)


def find_text_protocol_csv(root: str | Path) -> Optional[Path]:
    """Find the user's L2 text protocol CSV."""
    root = Path(root)
    if not root.exists():
        return None

    cands = sorted(root.glob("text_protocol*.csv"))
    for p in cands:
        if _is_l2_text_csv(p):
            return p

    cands = sorted(root.rglob("text_protocol*.csv"))
    for p in cands:
        if _is_l2_text_csv(p):
            return p

    cands = sorted(root.glob("*.csv"))
    for p in cands:
        if _is_l2_text_csv(p):
            return p

    cands = sorted(root.rglob("*.csv"))
    for p in cands:
        if _is_l2_text_csv(p):
            return p
    return None


def discover_seedvii_paths(root: str | Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (eeg_root, text_csv) from a ModelScope dataset local directory."""
    root = Path(root)
    return find_eeg_root(root), find_text_protocol_csv(root)
