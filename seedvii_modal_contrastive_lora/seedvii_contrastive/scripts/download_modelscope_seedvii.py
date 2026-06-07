from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
from typing import List


def _file_path(entry: dict) -> str:
    return entry.get("Path") or entry.get("path") or entry.get("Name") or entry.get("name") or ""


def list_dataset_files(dataset_id: str, revision: str = "master", token: str | None = None) -> List[str]:
    """List ModelScope dataset files using the dataset Hub API, not model protocol."""
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


def select_seedvii_files(files: List[str]) -> List[str]:
    """Select 1-20.mat plus Brain-CLIPLM-style text protocol CSV.

    The true EEG_preprocessed subject files are HDF5 MATLAB files named 1.mat ... 20.mat.
    We intentionally exclude zip / split-volume files here.
    """
    selected = []
    subject_names = {f"{i}.mat" for i in range(1, 21)} | {f"{i:02d}.mat" for i in range(1, 21)}
    for p in files:
        base = Path(p).name
        low = p.lower()
        if base in subject_names:
            selected.append(p)
        elif fnmatch.fnmatch(low, "*text_protocol*.csv"):
            selected.append(p)
        elif fnmatch.fnmatch(low, "*videoid_to_emotion*.csv"):
            selected.append(p)
    return sorted(set(selected))


def dataset_download(dataset_id: str, local_dir: str, allow_patterns: List[str], revision: str = "master", token: str | None = None, max_workers: int = 4) -> str:
    """Download selected dataset files via ModelScope dataset snapshot API."""
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
    except Exception as e1:
        # Compatibility with SDK versions exposing only snapshot_download.
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
        except Exception as e2:
            raise RuntimeError(f"dataset download failed. dataset_snapshot_download error={e1}; snapshot_download error={e2}")


def find_downloaded_paths(local_dir: str | Path) -> tuple[Path | None, Path | None]:
    root = Path(local_dir)
    mat_files = [p for p in root.rglob("*.mat") if p.name in {f"{i}.mat" for i in range(1,21)} | {f"{i:02d}.mat" for i in range(1,21)}]
    text_files = sorted(root.rglob("text_protocol*.csv"))
    eeg_root = None
    if mat_files:
        # Choose the directory containing the most subject mats.
        counts = {}
        for p in mat_files:
            counts[p.parent] = counts.get(p.parent, 0) + 1
        eeg_root = max(counts, key=counts.get)
    return eeg_root, (text_files[0] if text_files else None)


def main():
    ap = argparse.ArgumentParser(description="Download SEED-VII EEG_preprocessed subject H5 .mat and text_protocol.csv from ModelScope dataset")
    ap.add_argument("--dataset-id", default="DEREKVERSE/SEED-VII")
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--revision", default="master")
    ap.add_argument("--token", default=None, help="or set MODELSCOPE_TOKEN")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()
    token = args.token or os.environ.get("MODELSCOPE_TOKEN")

    print(f"[ModelScope] listing dataset files with repo_type='dataset': {args.dataset_id}")
    files = list_dataset_files(args.dataset_id, args.revision, token)
    print(f"[ModelScope] total files listed: {len(files)}")
    selected = select_seedvii_files(files)
    print("[ModelScope] selected files:")
    for p in selected:
        print("  ", p)
    if args.list_only:
        return
    if not selected:
        raise RuntimeError("No 1-20.mat or text_protocol*.csv selected. Please check dataset file layout.")
    local = dataset_download(args.dataset_id, args.local_dir, selected, args.revision, token, args.max_workers)
    eeg_root, text_csv = find_downloaded_paths(local)
    print(f"[OK] downloaded to: {local}")
    print(f"[OK] EEG_ROOT={eeg_root}")
    print(f"[OK] TEXT_CSV={text_csv}")
    if text_csv is None:
        raise RuntimeError("text_protocol*.csv was not found after download. The LLM Tower requires your L2 text protocol.")


if __name__ == "__main__":
    main()
