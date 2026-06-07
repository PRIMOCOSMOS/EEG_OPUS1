from __future__ import annotations

# Allow running this file directly, e.g.
#   python seedvii_contrastive/scripts/xxx.py
# without requiring `pip install -e .` or setting PYTHONPATH.
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from seedvii_contrastive.data.protocol import (
    TRIAL_FINE, trial_to_labels, load_l2_text_protocol,
    FINE_TO_VALENCE, VALENCE_NAMES,
)

ALIASES = {
    "happy": "joy", "joy": "joy", "happiness": "joy",
    "surprise": "surprise", "surprised": "surprise",
    "neutral": "neutral", "calm": "neutral",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "fearful": "fear",
    "sad": "sadness", "sadness": "sadness",
    "anger": "anger", "angry": "anger",
}


def _tid(raw):
    m = re.search(r"\d+", str(raw or ""))
    return int(m.group(0)) if m else None


def _norm_emo(raw):
    key = str(raw or "").strip().lower().replace(" ", "")
    return ALIASES.get(key, key)


def load_videoid_emotion(path: str | Path):
    out = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        lower = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        id_col = lower.get("video_id") or lower.get("videoid") or lower.get("trial") or lower.get("id")
        emo_col = lower.get("emotion") or lower.get("fine_emotion") or lower.get("label")
        if not id_col or not emo_col:
            raise ValueError(f"{path} must contain video_id/trial and emotion columns; got {reader.fieldnames}")
        for r in reader:
            tid = _tid(r.get(id_col))
            if tid is not None:
                out[tid] = _norm_emo(r.get(emo_col))
    return out


def main():
    ap = argparse.ArgumentParser(description="Check SEED-VII label aggregation and L2 text protocol")
    ap.add_argument("--text-csv", default=None, help="Brain-CLIPLM-style text_protocol.csv with trial,l2_text")
    ap.add_argument("--videoid-emotion-csv", default=None, help="Optional official/reference videoid_to_emotion.csv for cross-check")
    args = ap.parse_args()

    print("[Label protocol] n_trials =", len(TRIAL_FINE))
    print("[Fine counts]", dict(sorted(Counter(TRIAL_FINE.values()).items())))
    vals, ids = [], []
    for tid in range(1, 81):
        fine, val, y = trial_to_labels(tid)
        vals.append(val); ids.append(y)
    print("[Valence counts]", dict(sorted(Counter(vals).items())))
    print("[ID counts]", dict(sorted(Counter(ids).items())))
    print("[ID mapping]", {i: name for i, name in enumerate(VALENCE_NAMES)})
    print("[First 10]")
    for tid in range(1, 11):
        print(f"  {tid:02d}: fine={trial_to_labels(tid)[0]:8s} valence={trial_to_labels(tid)[1]:8s} id={trial_to_labels(tid)[2]}")

    if args.text_csv:
        texts = load_l2_text_protocol(args.text_csv)
        lengths = [len(t.split()) for t in texts.values()]
        print(f"[L2 text] loaded {len(texts)} trials from {args.text_csv}")
        print(f"[L2 text] word length min/mean/max = {min(lengths)}/{sum(lengths)/len(lengths):.1f}/{max(lengths)}")

    if args.videoid_emotion_csv:
        ref = load_videoid_emotion(args.videoid_emotion_csv)
        mismatches = []
        for tid in range(1, 81):
            if tid in ref and ref[tid] != TRIAL_FINE[tid]:
                mismatches.append((tid, TRIAL_FINE[tid], ref[tid]))
        print(f"[videoid_to_emotion] loaded {len(ref)} rows from {args.videoid_emotion_csv}")
        if mismatches:
            print("[WARN] mismatches found:")
            for item in mismatches[:20]:
                print("  trial=%s hardcoded=%s csv=%s" % item)
            raise SystemExit(2)
        print("[videoid_to_emotion] cross-check PASS")


if __name__ == "__main__":
    main()
