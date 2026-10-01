"""
Phase 3: build the searchable review index.

Embeds every review locally (no API cost) with BAAI/bge-small-en-v1.5 and stores it in a
Chroma database at data/chroma, along with metadata for filtering: price, price tier,
stars, value verdict and one field per theme. Re-running rebuilds the index from scratch.

    python pipeline\\03_build_index.py

The first run downloads the embedding model (~130 MB, cached in your user folder).
Embedding ~9,600 reviews takes a few minutes on a laptop CPU.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import qa  # noqa: E402

MAX_DOC_CHARS = 2000
EMBED_BATCH = 256
ADD_BATCH = 1000


def load() -> tuple[pd.DataFrame, list[str]]:
    reviews = pd.read_parquet(qa.DATA / "reviews.parquet")
    products = pd.read_parquet(qa.DATA / "products.parquet")[
        ["parent_asin", "title", "store", "price", "price_quintile"]].rename(columns={"title": "product_title"})
    tags = pd.read_parquet(qa.DATA / "review_tags.parquet")[
        ["review_id", "value_sentiment", "mentions_price", "themes_json"]]
    df = reviews.merge(products, on="parent_asin", how="left").merge(tags, on="review_id", how="left")
    themes = sorted(pd.read_parquet(qa.DATA / "review_themes.parquet")["theme"].unique())
    return df, themes


def metadata(r, themes: list[str]) -> dict:
    tagged = isinstance(r.themes_json, str)
    polarity = {t["theme"]: t["polarity"] for t in json.loads(r.themes_json)} if tagged else {}
    meta = {
        "review_id": r.review_id,
        "parent_asin": r.parent_asin,
        "product_title": str(r.product_title or "")[:150],
        "store": str(r.store or ""),
        "price": float(r.price),
        "tier": int(str(r.price_quintile)[1]),  # "Q1 lowest" -> 1
        "rating": float(r.rating or 0),
        "helpful_vote": int(r.helpful_vote or 0),
        "date": str(r.date or ""),
        "value_sentiment": r.value_sentiment if tagged else "untagged",
        "mentions_price": bool(r.mentions_price) if tagged else False,
    }
    for t in themes:
        meta[f"theme_{t}"] = polarity.get(t, "none")
    return meta


def main() -> None:
    import chromadb

    start = time.time()
    df, themes = load()
    docs = [(f"{t}. " if t else "") + str(x)[:MAX_DOC_CHARS]
            for t, x in zip(df["title"].fillna("").str.strip(), df["text"])]
    print(f"Embedding {len(docs):,} reviews with {qa.EMBED_MODEL} "
          f"(first run downloads the model)")

    vectors = []
    for i in range(0, len(docs), EMBED_BATCH):
        vectors.append(qa.embed_documents(docs[i:i + EMBED_BATCH]))
        done = min(i + EMBED_BATCH, len(docs))
        print(f"  {done:,}/{len(docs):,} embedded ({time.time() - start:,.0f}s)")
    vectors = __import__("numpy").vstack(vectors)

    qa.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(qa.CHROMA_DIR))
    if qa.COLLECTION in [c if isinstance(c, str) else c.name for c in client.list_collections()]:
        client.delete_collection(qa.COLLECTION)  # rebuild from scratch
    col = client.create_collection(qa.COLLECTION)  # vectors are unit length, so L2 ranks like cosine
    metas = [metadata(r, themes) for r in df.itertuples()]
    ids = df["review_id"].tolist()
    for i in range(0, len(ids), ADD_BATCH):
        col.add(ids=ids[i:i + ADD_BATCH], embeddings=vectors[i:i + ADD_BATCH].tolist(),
                documents=docs[i:i + ADD_BATCH], metadatas=metas[i:i + ADD_BATCH])

    info = {"reviews": col.count(), "embed_model": qa.EMBED_MODEL, "dims": int(vectors.shape[1]),
            "themes": themes, "untagged": int((df["value_sentiment"].isna()).sum()),
            "built": time.strftime("%Y-%m-%d %H:%M")}
    (qa.CHROMA_DIR / "index_info.json").write_text(json.dumps(info, indent=2))
    size_mb = sum(f.stat().st_size for f in qa.CHROMA_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"\nIndexed {info['reviews']:,} reviews ({info['dims']} dims) in "
          f"{time.time() - start:,.0f}s; index is {size_mb:,.0f} MB at {qa.CHROMA_DIR}")
    print('Try it:  python qa.py "What do buyers of flagship headphones complain about?" --band flagship')


if __name__ == "__main__":
    main()
