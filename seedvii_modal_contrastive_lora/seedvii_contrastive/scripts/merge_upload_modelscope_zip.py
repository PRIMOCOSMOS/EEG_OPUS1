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
from pathlib import Path
import re
import shutil
import os


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def main():
    ap = argparse.ArgumentParser(description="Merge multipart SEED-VII zip and optionally upload to ModelScope dataset")
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--pattern", default="*.zip.*")
    ap.add_argument("--output-zip", required=True)
    ap.add_argument("--delete-parts-after-copy", action="store_true", help="Only use if parts are writable; saves disk during merge")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--dataset-id", default="DEREKVERSE/SEED-VII")
    ap.add_argument("--path-in-repo", default="EEG_preprocessed.zip")
    ap.add_argument("--token-env", default="MODELSCOPE_TOKEN")
    args = ap.parse_args()

    parts = sorted(Path(args.parts_dir).glob(args.pattern), key=natural_key)
    if not parts:
        raise FileNotFoundError(f"no parts matched {args.pattern} in {args.parts_dir}")
    total = sum(p.stat().st_size for p in parts)
    out = Path(args.output_zip); out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out.parent).free
    if free < total * 1.05 and not args.delete_parts_after_copy:
        print(f"[WARN] free space {free/1e9:.1f}GB < merged size {total/1e9:.1f}GB. "
              "On a 100GB persistent disk this will fail for a 160GB zip. "
              "Use a larger ephemeral path, mounted dataset file, or --delete-parts-after-copy if parts are writable.")
    print(f"merging {len(parts)} parts -> {out} ({total/1e9:.2f} GB)")
    with open(out, "wb") as dst:
        for p in parts:
            print("  +", p.name)
            with open(p, "rb") as src:
                shutil.copyfileobj(src, dst, length=1024 * 1024 * 64)
            if args.delete_parts_after_copy:
                try:
                    p.unlink()
                except OSError as e:
                    print("cannot delete", p, e)
    print("merged zip:", out, out.stat().st_size / 1e9, "GB")

    if args.upload:
        token = os.environ.get(args.token_env)
        if not token:
            raise RuntimeError(f"set {args.token_env} env var before upload")
        from modelscope.hub.api import HubApi
        api = HubApi()
        try:
            api.login(access_token=token)
        except TypeError:
            api.login(token)
        print(f"uploading to modelscope dataset {args.dataset_id}:{args.path_in_repo}")
        try:
            api.upload_file(
                path_or_fileobj=str(out),
                path_in_repo=args.path_in_repo,
                repo_id=args.dataset_id,
                repo_type="dataset",
                commit_message="upload merged EEG_preprocessed zip",
            )
        except TypeError:
            api.upload_file(str(out), args.path_in_repo, args.dataset_id, repo_type="dataset")
        print("upload done")


if __name__ == "__main__":
    main()
