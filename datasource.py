"""
Where the app's data lives, and fetching it when deployed.

data_dir() picks, in order:
  1. $REVIEW_DATA_DIR, if set (for testing a clean download)
  2. data/processed, the full local pipeline output (development)
  3. app_data/, the slim copy built by pipeline/05_app_data.py

On Streamlit Community Cloud only the third exists, and it starts empty: app.py calls
download() first, which pulls the private Hugging Face dataset repo (HF_DATA_REPO, read with
HF_TOKEN from Streamlit secrets) and checks every file against the manifest's checksums.
The review text never goes through GitHub (see CLAUDE.md, Deployment).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / "data" / "processed"
APP_DATA = ROOT / "app_data"
MANIFEST = "manifest.json"
REQUIRED = ["reviews.parquet", "products.parquet", "review_tags.parquet", "review_themes.parquet",
            "price_checks.parquet", "review_embeddings.npz"]


def data_dir() -> Path:
    if os.environ.get("REVIEW_DATA_DIR"):
        return Path(os.environ["REVIEW_DATA_DIR"])
    if (LOCAL / "reviews.parquet").exists():
        return LOCAL
    return APP_DATA


def has_data(path: Path | None = None) -> bool:
    path = path or data_dir()
    return all((path / name).exists() for name in REQUIRED)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path) -> list[str]:
    """Files that are missing or don't match the manifest (empty list = all good)."""
    manifest = json.loads((path / MANIFEST).read_text())
    return [name for name, f in manifest["files"].items()
            if not (path / name).exists() or _sha256(path / name) != f["sha256"]]


def download(repo: str, token: str | None, target: Path | None = None) -> Path:
    """Fetch the private dataset repo into target (default: data_dir()) and verify it."""
    from huggingface_hub import snapshot_download
    target = target or data_dir()
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, repo_type="dataset", token=token, local_dir=str(target))
    bad = verify(target)
    if bad:
        raise RuntimeError(f"Downloaded app data failed its checksum check: {', '.join(bad)}")
    return target
