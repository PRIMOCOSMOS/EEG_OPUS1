from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import os
import subprocess
from typing import List

from seedvii_contrastive.data.discovery import discover_seedvii_paths, SUBJECT_FILE_NAMES


def _file_path(entry: dict) -> str:
    return entry.get("Path") or entry.get("path") or entry.get("Name") or entry.get("name") or ""


def list_dataset_files(dataset_id: str, revision: str = "master", token: str | None = None) -> List[str]:
    """List ModelScope dataset files using the dataset Hub API."""
    from modelscope.hub.api import HubApi
    api = HubApi(token=token)
    endpoint = api.get_endpoint_for_read(repo_id=dataset_id, repo_type="dataset", token=token)
    namespace, name = dataset_id.split("/", 1)
    hub_id, _ = api.get_dataset_id_and_type(dataset_name=name, namespace=namespace, endpoint=endpoint, token=token)
    out: List[str] = []
    page = 1
    while True:
        files = api.get_dataset_files(
            repo_id=dataset_id,
            revision=revision,
            root_path="/",
            recursive=True,
            page_number=page,
            page_size=200,
            endpoint=endpoint,
            token=token,
            dataset_hub_id=hub_id,
        )
        if not files:
            break
        for f in files:
            if f.get("Type") == "tree":
                continue
            p = _file_path(f)
            if p:
                out.append(p.lstrip("/"))
        if len(files) < 200:
            break
        page += 1
    return sorted(set(out))


def default_allow_patterns() -> List[str]:
    """Patterns that cover root-level and nested layouts."""
    pats: List[str] = []
    for i in range(1, 21):
        for name in (f"{i}.mat", f"{i:02d}.mat"):
            pats.extend([name, f"*/{name}", f"*/*/{name}", f"**/{name}"])
    pats.extend(["*.csv", "*/*.csv", "*/*/*.csv", "**/*.csv"])
    return pats


def select_seedvii_files(files: List[str]) -> List[str]:
    """Select 1-20.mat plus all CSVs."""
    selected = []
    for p in files:
        base = Path(p).name
        if base in SUBJECT_FILE_NAMES:
            selected.append(p)
        elif p.lower().endswith(".csv"):
            selected.append(p)
    return sorted(set(selected))


def dataset_snapshot_download_compat(dataset_id: str, local_dir: str, allow_patterns: List[str],
                                     revision: str = "master", token: str | None = None,
                                     max_workers: int = 4) -> str:
    """Download selected dataset files via ModelScope dataset snapshot API."""
    last_err = None
    try:
        from modelscope.hub.snapshot_download import dataset_snapshot_download
        return dataset_snapshot_download(
            dataset_id=dataset_id,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            token=token,
            max_workers=max_workers,
        )
    except Exception as e:
        last_err = e

    try:
        from modelscope import snapshot_download
        return snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            token=token,
            max_workers=max_workers,
        )
    except Exception as e:
        raise RuntimeError(f"dataset snapshot download failed: {last_err}; fallback error={e}")


def cli_download_fallback(dataset_id: str, local_dir: str, token: str | None = None) -> None:
    """Last-resort CLI fallback."""
    cmd = ["modelscope", "download", "--dataset", dataset_id, "--local_dir", local_dir]
    env = os.environ.copy()
    if token:
        env["MODELSCOPE_TOKEN"] = token
    print("[WARN] Falling back to CLI:", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def find_downloaded_paths(local_dir: str | Path) -> tuple:
    """Find subject .mat root and L2 text protocol CSV after ModelScope download."""
    return discover_seedvii_paths(local_dir)


def _print_discovery(local_dir: str | Path) -> tuple:
    eeg_root, text_csv = find_downloaded_paths(local_dir)
    print(f"[DISCOVERY] local_dir={local_dir}")
    print(f"[DISCOVERY] EEG_ROOT={eeg_root}")
    print(f"[DISCOVERY] TEXT_CSV={text_csv}")
    if eeg_root:
        mats = sorted([p.name for p in Path(eeg_root).glob("*.mat") if p.name in SUBJECT_FILE_NAMES])
        print(f"[DISCOVERY] subject mat count in EEG_ROOT={len(mats)}; first={mats[:5]}")
    return eeg_root, text_csv


def main():
    ap = argparse.ArgumentParser(description="Download/discover SEED-VII from ModelScope dataset")
    ap.add_argument("--dataset-id", default="DEREKVERSE/SEED-VII")
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--revision", default="master")
    ap.add_argument("--token", default=None, help="or set MODELSCOPE_TOKEN")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--direct-pattern-download", action="store_true")
    ap.add_argument("--cli-fallback", action="store_true")
    args = ap.parse_args()
    token = args.token or os.environ.get("MODELSCOPE_TOKEN")
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    eeg_root, text_csv = _print_discovery(local_dir)
    if eeg_root is not None and text_csv is not None and not args.force_download:
        print("[OK] Required files already present; skip download.")
        return

    allow_patterns = default_allow_patterns()

    if args.list_only or not args.direct_pattern_download:
        try:
            print(f"[ModelScope] listing dataset files: {args.dataset_id}")
            files = list_dataset_files(args.dataset_id, args.revision, token)
            print(f"[ModelScope] total files listed: {len(files)}")
            selected = select_seedvii_files(files)
            print("[ModelScope] selected files:")
            for p in selected:
                print(" ", p)
            if args.list_only:
                return
            if selected:
                allow_patterns = selected
        except Exception as e:
            if args.list_only:
                raise
            print(f"[WARN] Dataset listing failed ({type(e).__name__}: {e}); using direct allow_patterns download.")

    try:
        dataset_snapshot_download_compat(args.dataset_id, str(local_dir), allow_patterns, args.revision, token, args.max_workers)
    except Exception as e:
        if args.cli_fallback:
            cli_download_fallback(args.dataset_id, str(local_dir), token)
        else:
            raise

    eeg_root, text_csv = _print_discovery(local_dir)
    if eeg_root is None:
        raise RuntimeError("Downloaded files do not include 1-20.mat.")
    if text_csv is None:
        raise RuntimeError("Downloaded files do not include a CSV with trial + l2_text columns.")
    print("[OK] discovery complete.")


if __name__ == "__main__":
    main()
