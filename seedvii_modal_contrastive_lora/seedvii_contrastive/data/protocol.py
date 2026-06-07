from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import re

# Fine-grained SEED-VII emotions, kept consistent with Brain-CLIPLM naming.
CODE_TO_FINE = {
    "H": "joy",
    "U": "surprise",
    "N": "neutral",
    "D": "disgust",
    "F": "fear",
    "S": "sadness",
    "A": "anger",
}

FINE_TO_PROMPT_WORD = {
    "neutral": "neutral",
    "joy": "happy",
    "sadness": "sad",
    "fear": "fearful",
    "disgust": "disgusted",
    "anger": "angry",
    "surprise": "surprised",
}

# Aggregated three-class valence protocol.
VALENCE_NAMES = ["negative", "neutral", "positive"]
VALENCE_TO_ID = {name: i for i, name in enumerate(VALENCE_NAMES)}
FINE_TO_VALENCE = {
    "sadness": "negative",
    "fear": "negative",
    "disgust": "negative",
    "anger": "negative",
    "neutral": "neutral",
    "joy": "positive",
    "surprise": "positive",
}

# Design.md session protocol: 4 sessions × 20 trials, 4 folds × 5 trials.
SESSION_SEQUENCES = {
    1: [
        ["H", "N", "D", "S", "A"],
        ["A", "S", "D", "N", "H"],
        ["H", "N", "D", "S", "A"],
        ["A", "S", "D", "N", "H"],
    ],
    2: [
        ["A", "S", "F", "N", "U"],
        ["U", "N", "F", "S", "A"],
        ["A", "S", "F", "N", "U"],
        ["U", "N", "F", "S", "A"],
    ],
    3: [
        ["H", "U", "D", "F", "A"],
        ["A", "F", "D", "U", "H"],
        ["H", "U", "D", "F", "A"],
        ["A", "F", "D", "U", "H"],
    ],
    4: [
        ["D", "S", "F", "U", "H"],
        ["H", "U", "F", "S", "D"],
        ["D", "S", "F", "U", "H"],
        ["H", "U", "F", "S", "D"],
    ],
}


def build_trial_fine_emotions() -> Dict[int, str]:
    """Return {trial_id 1..80: fine_emotion_name}."""
    out: Dict[int, str] = {}
    for session_id in range(1, 5):
        trial_base = (session_id - 1) * 20
        folds = SESSION_SEQUENCES[session_id]
        for fold_idx, fold in enumerate(folds):
            for j, code in enumerate(fold):
                trial_id = trial_base + fold_idx * 5 + j + 1
                out[trial_id] = CODE_TO_FINE[code]
    assert len(out) == 80
    return out

TRIAL_FINE = build_trial_fine_emotions()


def trial_to_labels(trial_id: int) -> Tuple[str, str, int]:
    fine = TRIAL_FINE[int(trial_id)]
    val = FINE_TO_VALENCE[fine]
    return fine, val, VALENCE_TO_ID[val]


def prompt_for_fine(fine: str) -> str:
    return FINE_TO_PROMPT_WORD[fine]


def _extract_trial_id(raw: str) -> Optional[int]:
    if raw is None:
        return None
    m = re.search(r"\d+", str(raw))
    return int(m.group(0)) if m else None


def load_l2_text_protocol(csv_path: str | Path) -> Dict[int, str]:
    """Load Brain-CLIPLM-style text_protocol.csv and return {trial_id: l2_text}.

    Expected protocol columns are compatible with the reference repo:
      emotion,l1_text,trial,l2_text
    Only l2_text is used as LLM Tower input. Emotion labels for positives/negatives
    still come from the SEED-VII session protocol and three-class aggregation.

    The loader is tolerant to common aliases: video_id/id for trial, text/sentence
    for l2_text.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"text protocol CSV not found: {path}")
    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {path}")
        lower = {c.lower().strip(): c for c in reader.fieldnames}
        trial_col = lower.get("trial") or lower.get("video_id") or lower.get("videoid") or lower.get("id")
        text_col = lower.get("l2_text") or lower.get("text") or lower.get("sentence") or lower.get("description")
        if not trial_col or not text_col:
            raise ValueError(
                f"{path} must contain trial/video_id and l2_text/text columns; got {reader.fieldnames}")
        for row in reader:
            tid = _extract_trial_id(row.get(trial_col, ""))
            txt = (row.get(text_col) or "").strip()
            if tid is None or not txt or txt.upper().startswith("TODO"):
                continue
            out[tid] = txt
    missing = [i for i in range(1, 81) if i not in out]
    if missing:
        raise ValueError(f"L2 text protocol incomplete: missing trials {missing[:20]}{'...' if len(missing)>20 else ''}")
    return out


def write_label_protocol_csvs(out_dir: str | Path) -> None:
    """Write label-only protocol files for reproducibility.

    This does NOT generate L2 supervision text. L2 text must come from the user's
    carefully designed Brain-CLIPLM-style text_protocol.csv.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "videoid_to_emotion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "emotion", "valence", "label3"])
        w.writeheader()
        for tid in range(1, 81):
            fine, val, y = trial_to_labels(tid)
            w.writerow({"video_id": tid, "emotion": fine, "valence": val, "label3": y})
