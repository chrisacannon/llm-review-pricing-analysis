"""
Retrieval and Q&A over the tagged reviews (Phase 3). Used by the Streamlit app and
runnable from the command line.

Ask one question (from the project folder, environment active, API key set):
    python qa.py "What do buyers of premium headphones complain about?" --tier 5
    python qa.py "Do cheap earbuds hold up?" --max-price 20 --theme build_durability

Run the 10-question check and write data/processed/qa_eval.md:
    python qa.py --eval

Cache the app's example answers (re-run after changing EXAMPLES, the prompt or the index):
    python qa.py --cache-examples

How it works: the question is embedded with the same local model used for the reviews
(BAAI/bge-small-en-v1.5 via fastembed), the closest reviews are pulled from the Chroma
store (optionally filtered by price tier, price, value verdict, theme or stars), and
Claude answers from those reviews only, citing review IDs like [r001234]. Citations are
then checked against the reviews actually retrieved.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "reviews"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_CACHE = Path.home() / ".cache" / "fastembed"  # outside OneDrive; ~130 MB, downloaded once
ANSWER_MODEL = "claude-sonnet-5-5"
PRICE_ANSWER = (2.00, 10.00)  # USD per million tokens, Sonnet 5.5 (checked 2026-09-30)
CITE_RE = re.compile(r"\br\d{6}\b")

# Named price bands (USD, min inclusive, max exclusive). Edit thresholds here.
PRICE_BANDS = {
    "budget": (0, 25),
    "value": (25, 50),
    "mid": (50, 100),
    "premium": (100, 200),
    "flagship": (200, None),
}


def band_label(name: str) -> str:
    lo, hi = PRICE_BANDS[name]
    rng = f"${lo:,}+" if hi is None else (f"under ${hi:,}" if lo == 0 else f"${lo:,}-{hi - 1:,}")
    return f"{name.capitalize()} ({rng})"


# ---------------------------------------------------------------- embeddings

@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding
    EMBED_CACHE.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(EMBED_MODEL, cache_dir=str(EMBED_CACHE))


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def embed_documents(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return _normalize(np.array(list(_embedder().embed(texts, batch_size=batch_size)), dtype=np.float32))


def embed_query(text: str) -> np.ndarray:
    # query_embed adds the retrieval instruction bge models expect on the query side
    return _normalize(np.array(list(_embedder().query_embed(text)), dtype=np.float32))[0]


# ---------------------------------------------------------------- store

@lru_cache(maxsize=1)
def collection():
    import chromadb
    if not CHROMA_DIR.exists():
        sys.exit("No review index yet. Run:  python pipeline\\03_build_index.py")
    return chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)


def index_info() -> dict:
    path = CHROMA_DIR / "index_info.json"
    return json.loads(path.read_text()) if path.exists() else {}


@lru_cache(maxsize=1)
def excluded_asins() -> tuple[str, ...]:
    """Products left out of every statistic by the price check (suspect listings, non-headphones,
    hand-check exclusions). Q&A leaves them out too, so a $20 earbud listed at $803 isn't
    quoted as flagship feedback."""
    path = DATA / "price_checks.parquet"
    if not path.exists():
        return ()
    import pandas as pd
    checks = pd.read_parquet(path, columns=["parent_asin", "exclude"])
    return tuple(sorted(checks.loc[checks["exclude"], "parent_asin"]))


def band_of(price: float) -> str:
    for name, (lo, hi) in PRICE_BANDS.items():
        if price >= lo and (hi is None or price < hi):
            return name
    raise ValueError(f"No band for price {price}")


def build_where(tiers: list[int] | None = None, bands: list[str] | None = None, min_price: float | None = None,
                max_price: float | None = None, value: list[str] | None = None,
                themes: list[str] | None = None, theme_polarity: str | None = None,
                min_rating: float | None = None, max_rating: float | None = None,
                include_excluded: bool = False) -> dict | None:
    """Translate simple filters into a Chroma `where` clause. Themes are OR-ed."""
    clauses = []
    if not include_excluded and excluded_asins():
        clauses.append({"parent_asin": {"$nin": list(excluded_asins())}})
    if bands:
        ors = []
        for b in bands:
            lo, hi = PRICE_BANDS[b]
            rng = [{"price": {"$gte": float(lo)}}] + ([{"price": {"$lt": float(hi)}}] if hi else [])
            ors.append(rng[0] if len(rng) == 1 else {"$and": rng})
        clauses.append(ors[0] if len(ors) == 1 else {"$or": ors})
    if tiers:
        clauses.append({"tier": {"$in": [int(t) for t in tiers]}})
    if min_price is not None:
        clauses.append({"price": {"$gte": float(min_price)}})
    if max_price is not None:
        clauses.append({"price": {"$lte": float(max_price)}})
    if value:
        clauses.append({"value_sentiment": {"$in": list(value)}})
    if min_rating is not None:
        clauses.append({"rating": {"$gte": float(min_rating)}})
    if max_rating is not None:
        clauses.append({"rating": {"$lte": float(max_rating)}})
    if themes:
        allowed = [theme_polarity] if theme_polarity else ["positive", "negative", "mixed"]
        ors = [{f"theme_{t}": {"$in": allowed}} for t in themes]
        clauses.append(ors[0] if len(ors) == 1 else {"$or": ors})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def retrieve(question: str, k: int = 15, max_per_product: int | None = 3, **filters) -> list[dict]:
    """The k closest reviews, at most max_per_product from any one product (None = no cap).
    Guards against one product dominating an answer (Phase 3: Bang & Olufsen; Phase 4: 4 of 15
    flagship-complaint reviews were the Master & Dynamic MW08)."""
    res = collection().query(
        query_embeddings=[embed_query(question).tolist()],
        n_results=k * 3 if max_per_product else k,  # extra candidates to fill the cap from
        where=build_where(**filters),
        include=["documents", "metadatas", "distances"],
    )
    out, per_product = [], {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        asin = meta.get("parent_asin")
        if max_per_product and per_product.get(asin, 0) >= max_per_product:
            continue
        per_product[asin] = per_product.get(asin, 0) + 1
        out.append({**meta, "text": doc, "distance": round(float(dist), 4)})
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------- answering

SYSTEM = """You answer questions about Amazon headphone reviews for a pricing analyst.

Rules:
- Use ONLY the reviews provided. If they don't answer the question, say so plainly.
- Cite the review IDs that support each point in square brackets, e.g. [r001234] or [r001234, r005678].
- These reviews are the closest matches retrieved for the question, not a random sample.
  Describe what they say ("several reviewers...", "two reviewers of $15 earbuds...");
  never present counts or percentages as if they describe all reviews.
- Mention prices and price bands when they matter to the point.
- Be concise: a one-sentence answer, then 2 to 5 short bullet points."""


def _format_reviews(reviews: list[dict]) -> str:
    parts = []
    for r in reviews:
        parts.append(
            f"<review id=\"{r['review_id']}\" product=\"{r['product_title'][:80]}\" "
            f"price_usd=\"{r['price']:.2f}\" band=\"{band_label(band_of(r['price']))}\" stars=\"{r['rating']:.0f}\" "
            f"value_tag=\"{r['value_sentiment']}\">\n{r['text']}\n</review>"
        )
    return "\n\n".join(parts)


def answer(question: str, k: int = 15, client=None, **filters) -> dict:
    """Retrieve reviews, ask Claude, and check its citations."""
    reviews = retrieve(question, k=k, **filters)
    if not reviews:
        return {"answer": "No reviews match those filters.", "reviews": [], "cited": [],
                "invalid_citations": [], "cost": 0.0}
    if client is None:
        import anthropic
        client = anthropic.Anthropic(max_retries=3)
    msg = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content":
                   f"<reviews>\n{_format_reviews(reviews)}\n</reviews>\n\nQuestion: {question}"}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    if getattr(msg, "stop_reason", None) == "max_tokens":
        text += "\n\n[Answer cut off at the length limit.]"
    retrieved = {r["review_id"] for r in reviews}
    cited = list(dict.fromkeys(CITE_RE.findall(text)))
    cost = (msg.usage.input_tokens * PRICE_ANSWER[0] + msg.usage.output_tokens * PRICE_ANSWER[1]) / 1e6
    return {
        "answer": text,
        "reviews": reviews,
        "cited": [c for c in cited if c in retrieved],
        "invalid_citations": [c for c in cited if c not in retrieved],
        "cost": cost,
    }


# ---------------------------------------------------------------- command line

EVAL_QUESTIONS = [
    ("What do buyers of flagship headphones complain about most?", {"bands": ["flagship"]}),
    ("Do cheap earbuds hold up over time?", {"bands": ["budget"], "themes": ["build_durability"]}),
    ("What do reviewers say about noise cancelling on headphones under $50?",
     {"max_price": 50, "themes": ["noise_cancelling"]}),
    ("What problems do people have with Bluetooth connections dropping?", {"themes": ["connectivity"]}),
    ("What do parents say about headphones they bought for their kids?", {}),
    ("Why do some reviewers feel their headphones were overpriced?", {"value": ["negative"]}),
    ("How comfortable are these for long listening sessions?", {"themes": ["comfort_fit"]}),
    ("How good are the microphones for phone and video calls?", {"themes": ["microphone_calls"]}),
    ("Are there complaints about counterfeit, used or wrong items?", {"themes": ["authenticity_condition"]}),
    ("What makes reviewers call a pair of headphones a great deal?", {"value": ["positive"]}),
]


# The app's example questions. Their answers are generated once (--cache-examples) and served
# from EXAMPLE_CACHE, so clicking an example is instant, free and doesn't count against a visitor's limit.
EXAMPLES = [
    ("What do flagship buyers complain about?", {"bands": ["flagship"]}),
    ("Why do buyers feel premium headphones are overpriced?", {"bands": ["premium"], "value": ["negative"]}),
    ("Do budget earbuds hold up over time?", {"bands": ["budget"], "themes": ["build_durability"]}),
    ("What makes reviewers call a pair of headphones a great deal?", {"value": ["positive"]}),
]
EXAMPLE_CACHE = DATA / "example_answers.json"


def example_key(question: str, filters: dict) -> str:
    return json.dumps([question, {k: v for k, v in sorted(filters.items()) if v}])


def load_example_cache() -> dict:
    return json.loads(EXAMPLE_CACHE.read_text(encoding="utf-8")) if EXAMPLE_CACHE.exists() else {}


def cache_examples(k: int = 15) -> None:
    """Answer each example question once and save the results for the app."""
    cache, total = {}, 0.0
    for question, filters in EXAMPLES:
        res = answer(question, k=k, **filters)
        res["cached_on"] = time.strftime("%Y-%m-%d")
        cache[example_key(question, filters)] = res
        total += res["cost"]
        print(f"  {question}: {len(res['cited'])} cited, {len(res['invalid_citations'])} invalid, ${res['cost']:.3f}")
    EXAMPLE_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    print(f"\nCached {len(cache)} answers to {EXAMPLE_CACHE} for ${total:.3f}")


def show_bands() -> None:
    import pandas as pd
    products = pd.read_parquet(DATA / "products.parquet")
    reviews = pd.read_parquet(DATA / "reviews.parquet", columns=["parent_asin"])
    counts = reviews.groupby("parent_asin").size()
    print(f"{'Band':<24}{'Products':>9}{'Reviews':>9}   Example products")
    for name, (lo, hi) in PRICE_BANDS.items():
        g = products[(products.price >= lo) & ((products.price < hi) if hi else True)]
        n_rev = int(counts.reindex(g.parent_asin).fillna(0).sum())
        examples = "; ".join(g.sort_values("rating_number", ascending=False).title.str[:28].head(2))
        print(f"{band_label(name):<24}{len(g):>9}{n_rev:>9}   {examples}")


def run_eval(k: int) -> None:
    lines = ["# Q&A check", "", f"Model {ANSWER_MODEL}, top {k} reviews per question, "
             f"run {time.strftime('%Y-%m-%d %H:%M')}.", "",
             "For each answer: are the claims supported by the cited reviews, "
             "and are the right reviews cited?", ""]
    total, bad = 0.0, 0
    for i, (q, f) in enumerate(EVAL_QUESTIONS, 1):
        res = answer(q, k=k, **f)
        total += res["cost"]
        bad += len(res["invalid_citations"])
        print(f"{i:>2}. {q}\n    {len(res['cited'])} reviews cited, "
              f"{len(res['invalid_citations'])} invalid, ${res['cost']:.3f}")
        lines += [f"## {i}. {q}", "", f"Filters: `{json.dumps(f) if f else 'none'}`", "",
                  res["answer"], ""]
        if res["invalid_citations"]:
            lines += [f"**Invalid citations (not in the retrieved reviews):** "
                      f"{', '.join(res['invalid_citations'])}", ""]
        by_id = {r["review_id"]: r for r in res["reviews"]}
        lines += ["<details><summary>Cited reviews</summary>", ""]
        for cid in res["cited"]:
            r = by_id[cid]
            snippet = " ".join(r["text"].split())[:400]
            lines += [f"- **{cid}**, {r['rating']:.0f} stars, ${r['price']:.2f}, "
                      f"{r['product_title'][:60]}: {snippet}"]
        lines += ["", "</details>", ""]
    out = DATA / "qa_eval.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTotal cost ${total:.2f}; {bad} invalid citations. Written to {out.resolve()}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?")
    p.add_argument("--eval", action="store_true", help="Run the 10-question check")
    p.add_argument("--k", type=int, default=15, help="Reviews retrieved per question")
    p.add_argument("--band", action="append", choices=list(PRICE_BANDS),
                   help="Price band (repeatable): " + ", ".join(band_label(b) for b in PRICE_BANDS))
    p.add_argument("--bands", action="store_true", help="Show product and review counts per price band")
    p.add_argument("--cache-examples", action="store_true",
                   help="Answer the app's example questions once and save them (~$0.08)")
    p.add_argument("--tier", type=int, action="append", help="Price quintile 1-5 (repeatable)")
    p.add_argument("--min-price", type=float)
    p.add_argument("--max-price", type=float)
    p.add_argument("--value", action="append",
                   choices=["positive", "negative", "neutral", "not_mentioned"])
    p.add_argument("--theme", action="append", help="e.g. comfort_fit (repeatable, OR-ed)")
    p.add_argument("--min-rating", type=float)
    p.add_argument("--max-rating", type=float)
    a = p.parse_args(argv)

    if a.eval:
        run_eval(a.k)
        return
    if a.bands:
        show_bands()
        return
    if a.cache_examples:
        cache_examples(a.k)
        return
    if not a.question:
        p.error("ask a question in quotes, or use --eval")
    res = answer(a.question, k=a.k, bands=a.band, tiers=a.tier, min_price=a.min_price, max_price=a.max_price,
                 value=a.value, themes=a.theme, min_rating=a.min_rating, max_rating=a.max_rating)
    print(res["answer"])
    print(f"\n{len(res['reviews'])} reviews retrieved, {len(res['cited'])} cited, "
          f"cost ${res['cost']:.3f}")
    if res["invalid_citations"]:
        print(f"Warning: cited IDs not in the retrieved reviews: {', '.join(res['invalid_citations'])}")


if __name__ == "__main__":
    main()
