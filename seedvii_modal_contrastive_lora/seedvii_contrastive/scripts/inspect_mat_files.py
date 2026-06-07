from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from seedvii_contrastive.data.h5io import subject_mat_path, sniff_mat_file, SubjectMatReader, explain_bad_mat_file


def parse_subjects(s: str):
    out = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a,b = map(int, part.split('-'))
            out.extend(range(a,b+1))
        else:
            out.append(int(part))
    return out or list(range(1,21))


def main():
    ap = argparse.ArgumentParser(description="Inspect SEED-VII .mat file signatures and trial keys")
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--subjects", default="1-20")
    ap.add_argument("--read-first-trial", action="store_true")
    args = ap.parse_args()
    for sid in parse_subjects(args.subjects):
        p = subject_mat_path(args.input_root, sid)
        info = sniff_mat_file(p)
        print(f"S{sid:02d}: path={p} size={info['size']} kind={info['kind']}")
        if info['kind'] not in {'matlab_v7.3_hdf5', 'matlab_v5_v7_classic'}:
            print(explain_bad_mat_file(p))
            continue
        try:
            with SubjectMatReader(p) as r:
                keys = r.trial_keys()
                print(f" trial_keys_count={len(keys)} first={keys[:5]} last={keys[-5:] if keys else []}")
                if args.read_first_trial:
                    arr = r.read_trial(1)
                    print(f" trial1_shape={arr.shape} dtype={arr.dtype} min={arr.min():.4g} max={arr.max():.4g}")
        except Exception as e:
            print(f" ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
