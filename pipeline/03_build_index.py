"""
Phase 3: build the searchable review index.

Embeds every review locally (no API cost) with BAAI/bge-small-en-v1.5 and saves the
unit-length vectors with their review IDs to data/processed/review_embeddings.npz (~22 MB),
plus review_embeddings.json with build details. qa.py searches it with a matrix product and
filters on the parquet files, so re-tagging reviews doesn't need a rebuild; adding or
removing reviews does. Re-running rebuilds from scratch.

    python pipeline\\03_build_index.py

The first run downloads the embedding model (~130 MB, cached in your user folder).
Embedding ~14,000 reviews takes about an hour on a laptop CPU (~53 min measured 2026-10-02);
only needed when reviews are added or removed.

(Until 2026-10-02 the index was a Chroma database at data/chroma. Replaced to make the
deployed app lighter; search results were checked to be identical on 18 test queries.)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import qa  # noqa: E402

EMBED_BATCH = 256


def main() -> None:
    start = time.time()
    reviews = pd.read_parquet(qa.DATA / "reviews.parquet", columns=["review_id", "title", "text"])
    reviews = reviews.sort_values("review_id").reset_index(drop=True)
    docs = [qa.review_doc(t, x) for t, x in zip(reviews["title"], reviews["text"])]
    print(f"Embedding {len(docs):,} reviews with {qa.EMBED_MODEL} (first run downloads the model)")

    vectors = []
    for i in range(0, len(docs), EMBED_BATCH):
        vectors.append(qa.embed_documents(docs[i:i + EMBED_BATCH]))
        done = min(i + EMBED_BATCH, len(docs))
        print(f"  {done:,}/{len(docs):,} embedded ({time.time() - start:,.0f}s)")
    vectors = np.vstack(vectors).astype(np.float32)

    np.savez(qa.EMBEDDINGS, ids=reviews["review_id"].to_numpy(dtype=str), vectors=vectors)
    tags = pd.read_parquet(qa.DATA / "review_tags.parquet", columns=["review_id"])
    info = {"reviews": len(reviews), "embed_model": qa.EMBED_MODEL, "dims": int(vectors.shape[1]),
            "untagged": int((~reviews["review_id"].isin(tags["review_id"])).sum()),
            "built": time.strftime("%Y-%m-%d %H:%M")}
    qa.EMBEDDINGS.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"\nIndexed {info['reviews']:,} reviews ({info['dims']} dims) in {time.time() - start:,.0f}s; "
          f"{qa.EMBEDDINGS.stat().st_size / 1e6:,.0f} MB at {qa.EMBEDDINGS}")
    print('Try it:  python qa.py "What do buyers of flagship headphones complain about?" --band flagship')


if __name__ == "__main__":
    main()
