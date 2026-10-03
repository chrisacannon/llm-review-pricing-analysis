"""
Phase 4: package the deployed app's data and upload it to a PRIVATE Hugging Face dataset repo.

The review text is never committed to GitHub: the dataset states no license for it, and a public
commit is hard to undo (see CLAUDE.md, Deployment). The public app downloads this data at startup
with a read token from Streamlit secrets.

  1. Build the slim folder (only the columns the app reads; no reviewer IDs):
       python pipeline\\05_app_data.py build
  2. Upload it (needs a Hugging Face token with write access in $env:HF_TOKEN):
       python pipeline\\05_app_data.py upload --repo your-username/review-pricing-intel-data

Re-run both after changing the data, tags, price checks, embeddings or example answers.
The upload refuses to continue if the repo exists and is public.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed"
OUT = ROOT / "app_data"  # git-ignored

# Only what app.py, analytics.py and qa.py read
PARQUET_COLUMNS = {
    "products.parquet": ["parent_asin", "title", "store", "price", "price_quintile",
                         "average_rating", "rating_number"],
    "reviews.parquet": ["review_id", "parent_asin", "rating", "helpful_vote", "date", "title", "text"],
    "review_tags.parquet": ["review_id", "value_sentiment", "mentions_price", "themes_json"],
    "review_themes.parquet": None,  # all columns
    "price_checks.parquet": None,
}
COPY_AS_IS = ["review_embeddings.npz", "review_embeddings.json", "example_answers.json"]
MANIFEST = "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cmd_build(args) -> None:
    # Empty the folder rather than deleting it: on Windows, OneDrive or Explorer can hold the folder
    # itself open, and removing it then fails with "Access is denied"
    OUT.mkdir(exist_ok=True)
    for p in OUT.iterdir():
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    for name, cols in PARQUET_COLUMNS.items():
        pd.read_parquet(SRC / name, columns=cols).to_parquet(OUT / name, index=False)
    missing = []
    for name in COPY_AS_IS:
        if (SRC / name).exists():
            shutil.copy2(SRC / name, OUT / name)
        else:
            missing.append(name)

    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(OUT.iterdir())}
    manifest = {"built": time.strftime("%Y-%m-%d %H:%M"), "files": files,
                "source": "Amazon Reviews 2023 (McAuley Lab, UCSD), Electronics, Headphones & Earbuds sample"}
    (OUT / MANIFEST).write_text(json.dumps(manifest, indent=2))

    total = sum(f["bytes"] for f in files.values())
    for name, f in files.items():
        print(f"  {name:28} {f['bytes'] / 1e6:6.1f} MB")
    print(f"\nBuilt {OUT} ({total / 1e6:.1f} MB, {len(files)} files)")
    if "example_answers.json" in missing:
        print("Note: no example answers yet; the app's example buttons will only fill in the question.\n"
              "      Run  python qa.py --cache-examples  and then build again.")
    other = [m for m in missing if m != "example_answers.json"]
    if other:
        sys.exit(f"Missing required files: {', '.join(other)}")


def cmd_upload(args) -> None:
    from huggingface_hub import HfApi
    if not (OUT / MANIFEST).exists():
        sys.exit("Build the folder first:  python pipeline\\05_app_data.py build")
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set a Hugging Face token with write access first, e.g. in PowerShell:\n"
                 '  $env:HF_TOKEN = "hf_..."')
    api = HfApi(token=token)
    url = api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(args.repo, repo_type="dataset").private:
        sys.exit(f"{args.repo} exists and is PUBLIC. Not uploading review text to a public repo.\n"
                 "Make it private on huggingface.co (Settings > Change visibility) or pick another name.")
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(OUT),
                      commit_message=f"App data built {json.loads((OUT / MANIFEST).read_text())['built']}")
    print(f"Uploaded {OUT} to private dataset {url}")
    print("For the app, add to Streamlit secrets:\n"
          f'  HF_DATA_REPO = "{args.repo}"\n'
          '  HF_TOKEN = "hf_..."   # a READ-only token is enough for the app')


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="Build the slim app_data folder from data/processed")
    su = sub.add_parser("upload", help="Upload app_data to a private Hugging Face dataset repo")
    su.add_argument("--repo", required=True, help="e.g. your-username/review-pricing-intel-data")
    args = p.parse_args(argv)
    {"build": cmd_build, "upload": cmd_upload}[args.cmd](args)


if __name__ == "__main__":
    main()
