"""
Phase 1: load, clean and sample one category of Amazon Reviews 2023.

What it does
  1. Reads the category's item metadata and keeps products that have a usable
     price and enough ratings (optionally narrowed by a keyword, e.g. "headphone").
  2. Samples products evenly across price quintiles, so every price tier is
     represented for the tier analysis in Phase 4.
  3. Reads the category's reviews and keeps a random sample per selected product
     (reservoir sampling, so the sample isn't biased toward the top of the file).
  4. Writes data/processed/products.parquet, reviews.parquet and summary.json,
     and prints a short summary.

Data source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

Examples
  # Small category, files downloaded once and cached (~1.2 GB total)
  python pipeline/01_load_data.py --category Musical_Instruments

  # Sub-category inside a huge file: stream it instead of saving it to disk
  python pipeline/01_load_data.py --category Electronics --keyword "headphone|earbud" --stream

  # Files you already downloaded yourself
  python pipeline/01_load_data.py --meta-file meta_Appliances.jsonl --reviews-file Appliances.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
META_PATH = "raw/meta_categories/meta_{category}.jsonl"
REVIEWS_PATH = "raw/review_categories/{category}.jsonl"

PRICE_RE = re.compile(r"\d[\d,]*\.?\d*")


# ---------------------------------------------------------------- reading files

def open_source(category: str | None, local_file: str | None, template: str,
                stream: bool) -> Iterator[str]:
    """Yield lines from a local file, the Hugging Face cache, or a live stream."""
    if local_file:
        path = Path(local_file)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            yield from f
        return

    filename = template.format(category=category)
    if stream:
        import requests
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(REPO_ID, filename, repo_type="dataset")
        print(f"  streaming {filename}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if raw:
                    yield raw.decode("utf-8")
    else:
        from huggingface_hub import hf_hub_download

        print(f"  downloading {filename} (cached after the first run)")
        path = hf_hub_download(REPO_ID, filename, repo_type="dataset")
        with open(path, "rt", encoding="utf-8") as f:
            yield from f


def iter_json(lines: Iterator[str], label: str) -> Iterator[dict]:
    bad = 0
    start = time.time()
    for i, line in enumerate(lines, 1):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            bad += 1
        if i % 500_000 == 0:
            print(f"    {label}: {i:,} lines read ({time.time() - start:,.0f}s)")
    if bad:
        print(f"    {label}: skipped {bad:,} malformed lines")


# ---------------------------------------------------------------- cleaning

def parse_price(value) -> float | None:
    """Return a positive float price, or None for missing, ranges and junk."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    numbers = PRICE_RE.findall(text)
    if len(numbers) != 1:  # "12.99 - 19.99" style ranges are ambiguous; drop them
        return None
    try:
        price = float(numbers[0].replace(",", ""))
    except ValueError:
        return None
    return price if price > 0 else None


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value)


# ---------------------------------------------------------------- steps

def load_products(lines, keyword: re.Pattern | None, min_ratings: int,
                  max_price: float | None, category_match: re.Pattern | None = None,
                  exclude: re.Pattern | None = None) -> tuple[pd.DataFrame, dict]:
    stats = defaultdict(int)
    rows = []
    for item in iter_json(lines, "metadata"):
        stats["items_read"] += 1
        price = parse_price(item.get("price"))
        if price is None:
            stats["dropped_no_price"] += 1
            continue
        if max_price and price > max_price:
            stats["dropped_over_max_price"] += 1
            continue
        n_ratings = item.get("rating_number") or 0
        if n_ratings < min_ratings:
            stats["dropped_few_ratings"] += 1
            continue
        title = as_text(item.get("title"))
        cats = item.get("categories") or []
        categories = " > ".join(str(c) for c in cats) if isinstance(cats, list) else str(cats)
        if category_match and not category_match.search(categories):
            stats["dropped_category"] += 1
            continue
        if keyword and not (keyword.search(title) or keyword.search(categories)):
            stats["dropped_keyword"] += 1
            continue
        if exclude and exclude.search(title):
            stats["dropped_excluded_title"] += 1
            continue
        if not item.get("parent_asin"):
            stats["dropped_no_id"] += 1
            continue
        rows.append({
            "parent_asin": item["parent_asin"],
            "title": title,
            "store": item.get("store"),
            "price": price,
            "average_rating": item.get("average_rating"),
            "rating_number": n_ratings,
            "main_category": item.get("main_category"),
            "categories": categories,
            "description": as_text(item.get("description"))[:2000],
        })
    df = pd.DataFrame(rows).drop_duplicates("parent_asin")
    stats["products_eligible"] = len(df)
    return df, dict(stats)


def sample_products(df: pd.DataFrame, max_products: int, seed: int) -> pd.DataFrame:
    """Take an equal share from each price quintile so every tier is represented."""
    if len(df) <= max_products:
        out = df.copy()
    else:
        quintile = pd.qcut(df["price"].rank(method="first"), 5, labels=False)
        per_bin = max_products // 5
        out = pd.concat(
            df[quintile == q].sample(min(int((quintile == q).sum()), per_bin),
                                     random_state=seed)
            for q in range(5)
        )
    return out.sort_values("price").reset_index(drop=True)


def load_reviews(lines, keep_ids: set[str], per_product: int, min_chars: int,
                 seed: int) -> tuple[pd.DataFrame, dict]:
    """Reservoir-sample up to `per_product` reviews for each selected product."""
    rng = random.Random(seed)
    seen = defaultdict(int)
    reservoir: dict[str, list[dict]] = defaultdict(list)
    stats = defaultdict(int)

    for r in iter_json(lines, "reviews"):
        stats["reviews_read"] += 1
        pid = r.get("parent_asin")
        if pid not in keep_ids:
            continue
        text = (r.get("text") or "").strip()
        if len(text) < min_chars:
            stats["dropped_too_short"] += 1
            continue
        row = {
            "parent_asin": pid,
            "user_id": r.get("user_id"),
            "rating": r.get("rating"),
            "title": (r.get("title") or "").strip(),
            "text": text,
            "helpful_vote": r.get("helpful_vote", 0),
            "verified_purchase": r.get("verified_purchase"),
            "timestamp": r.get("timestamp"),
        }
        seen[pid] += 1
        bucket = reservoir[pid]
        if len(bucket) < per_product:
            bucket.append(row)
        else:
            j = rng.randrange(seen[pid])
            if j < per_product:
                bucket[j] = row

    df = pd.DataFrame([row for bucket in reservoir.values() for row in bucket])
    if not df.empty:
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce").dt.date
        # Stable ID used for citations in Phase 3
        df = df.sort_values(["parent_asin", "timestamp"]).reset_index(drop=True)
        df.insert(0, "review_id", [f"r{i:06d}" for i in range(len(df))])
    stats["reviews_kept"] = len(df)
    return df, dict(stats)


# ---------------------------------------------------------------- main

def next_review_number(out: Path) -> int:
    """First unused review number, so appended reviews never reuse an existing or tagged ID."""
    used = []
    if (out / "reviews.parquet").exists():
        used += pd.read_parquet(out / "reviews.parquet", columns=["review_id"])["review_id"].tolist()
    if (out / "tags.jsonl").exists():
        with open(out / "tags.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    used.append(json.loads(line)["review_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    nums = [int(x[1:]) for x in used if isinstance(x, str) and x[1:].isdigit()]
    return max(nums) + 1 if nums else 0


def drop_duplicate_reviews(reviews: pd.DataFrame, min_chars: int = 40) -> tuple[pd.DataFrame, int]:
    """Drop repeated reviews of the same product; keep the first.

    A repeat is the same product and the same text, written by the same user, or by anyone if
    the text is at least `min_chars` long. Short generic texts ("Great headphones!") from
    different people are kept, since they can legitimately match.
    """
    if reviews.empty:
        return reviews, 0
    norm = reviews["text"].str.lower().str.split().str.join(" ")
    key = pd.DataFrame({"p": reviews["parent_asin"], "t": norm,
                        "u": reviews["user_id"].fillna("").astype(str)})
    same_user = key.duplicated(keep="first")
    long_text = key[["p", "t"]].duplicated(keep="first") & (norm.str.len() >= min_chars)
    dup = same_user | long_text
    return reviews[~dup].reset_index(drop=True), int(dup.sum())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("source")
    src.add_argument("--category", help="Dataset category, e.g. Musical_Instruments")
    src.add_argument("--meta-file", help="Local metadata .jsonl(.gz) instead of downloading")
    src.add_argument("--reviews-file", help="Local reviews .jsonl(.gz) instead of downloading")
    src.add_argument("--stream", action="store_true",
                     help="Stream from Hugging Face without saving the files (for huge categories)")
    flt = p.add_argument_group("filters")
    flt.add_argument("--keyword", help="Regex matched against title/categories, e.g. 'headphone|earbud'")
    flt.add_argument("--category-match",
                     help="Regex matched against the category path only, "
                          "e.g. 'Headphones & Earbuds' (paths look like 'Electronics > ... > ...')")
    flt.add_argument("--exclude", help="Regex; drop products whose TITLE matches, e.g. 'replacement|ear pads'")
    flt.add_argument("--min-ratings", type=int, default=30)
    flt.add_argument("--min-price", type=float, help="Drop products priced below this")
    flt.add_argument("--max-price", type=float, help="Drop products priced above this")
    flt.add_argument("--max-products", type=int, default=300)
    flt.add_argument("--reviews-per-product", type=int, default=60)
    flt.add_argument("--min-review-chars", type=int, default=20)
    p.add_argument("--append", action="store_true",
                   help="Add new products to the existing data instead of replacing it. Existing "
                        "products are skipped, review IDs continue where they left off, and the "
                        "old files are backed up first.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/processed")
    args = p.parse_args(argv)

    if not args.category and not (args.meta_file and args.reviews_file):
        p.error("give --category, or both --meta-file and --reviews-file")

    keyword = re.compile(args.keyword, re.I) if args.keyword else None
    category_match = re.compile(args.category_match, re.I) if args.category_match else None
    exclude = re.compile(args.exclude, re.I) if args.exclude else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    existing_products = existing_reviews = None
    if args.append:
        if not (out / "products.parquet").exists():
            p.error(f"--append needs existing data in {out}")
        existing_products = pd.read_parquet(out / "products.parquet")
        existing_reviews = pd.read_parquet(out / "reviews.parquet")
        print(f"Appending to {len(existing_products):,} products / {len(existing_reviews):,} reviews")

    print("1/3 Reading item metadata")
    meta_lines = open_source(args.category, args.meta_file, META_PATH, args.stream)
    eligible, meta_stats = load_products(meta_lines, keyword, args.min_ratings, args.max_price,
                                         category_match, exclude)
    if args.min_price is not None and not eligible.empty:
        below = eligible["price"] < args.min_price
        meta_stats["dropped_under_min_price"] = int(below.sum())
        eligible = eligible[~below]
    if existing_products is not None and not eligible.empty:
        already = eligible["parent_asin"].isin(existing_products["parent_asin"])
        meta_stats["skipped_already_have"] = int(already.sum())
        eligible = eligible[~already]
    if eligible.empty:
        print("No products passed the filters. Loosen --min-ratings or --keyword.")
        return 1

    products = sample_products(eligible, args.max_products, args.seed)
    print(f"    {len(eligible):,} eligible products, {len(products):,} sampled")

    print("2/3 Reading reviews for the sampled products")
    review_lines = open_source(args.category, args.reviews_file, REVIEWS_PATH, args.stream)
    reviews, review_stats = load_reviews(review_lines, set(products["parent_asin"]),
                                         args.reviews_per_product, args.min_review_chars,
                                         args.seed)
    reviews, review_stats["dropped_duplicates"] = drop_duplicate_reviews(reviews)
    if args.append and not reviews.empty:
        start = next_review_number(out)
        reviews["review_id"] = [f"r{i:06d}" for i in range(start, start + len(reviews))]

    counts = reviews.groupby("parent_asin").size() if not reviews.empty else pd.Series(dtype=int)
    products["reviews_sampled"] = products["parent_asin"].map(counts).fillna(0).astype(int)
    products = products[products["reviews_sampled"] > 0].reset_index(drop=True)
    new_products, new_reviews = products, reviews

    if args.append:
        backup = out / f"backup_{time.strftime('%Y%m%d_%H%M%S')}"
        backup.mkdir()
        for name in ("products.parquet", "reviews.parquet", "summary.json"):
            if (out / name).exists():
                shutil.copy2(out / name, backup / name)
        print(f"    previous files backed up to {backup}")
        old_reviews, old_dups = drop_duplicate_reviews(existing_reviews)
        review_stats["dropped_duplicates_existing"] = old_dups
        reviews = pd.concat([old_reviews, new_reviews], ignore_index=True)
        products = pd.concat([existing_products.drop(columns=["price_quintile"], errors="ignore"),
                              new_products], ignore_index=True)
        counts = reviews.groupby("parent_asin").size()
        products["reviews_sampled"] = products["parent_asin"].map(counts).fillna(0).astype(int)
        products = products[products["reviews_sampled"] > 0].reset_index(drop=True)

    tiers = pd.qcut(products["price"].rank(method="first"), 5,
                    labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
    products["price_quintile"] = tiers.astype(str)

    print("3/3 Writing outputs")
    products.to_parquet(out / "products.parquet", index=False)
    reviews.to_parquet(out / "reviews.parquet", index=False)

    def describe(prods, revs):
        c = revs.groupby("parent_asin").size() if not revs.empty else pd.Series(dtype=int)
        return {
            "products": len(prods),
            "reviews": len(revs),
            "price_usd": {k: round(float(v), 2) for k, v in
                          prods["price"].quantile([0, .2, .4, .6, .8, 1]).rename(
                              {0: "min", .2: "p20", .4: "p40", .6: "p60", .8: "p80", 1: "max"}).items()},
            "reviews_per_product": {"median": float(c.median()) if len(c) else 0,
                                    "min": int(c.min()) if len(c) else 0},
            "rating_mix": (revs["rating"].value_counts(normalize=True).sort_index()
                           .round(3).to_dict() if not revs.empty else {}),
        }

    summary = {
        "category": args.category or Path(args.meta_file).stem,
        "keyword": args.keyword,
        "category_match": args.category_match,
        "exclude": args.exclude,
        "filters": {k: getattr(args, k) for k in
                    ["min_ratings", "min_price", "max_price", "max_products",
                     "reviews_per_product", "min_review_chars", "seed"]},
        "metadata": meta_stats,
        "reviews": review_stats,
        "this_run": describe(new_products, new_reviews),
        "all_data": describe(products, reviews),
    }
    name = f"summary_append_{time.strftime('%Y%m%d_%H%M%S')}.json" if args.append else "summary.json"
    (out / name).write_text(json.dumps(summary, indent=2, default=str))

    print()
    if args.append:
        t = summary["this_run"]
        print(f"Added: {t['products']:,} products, {t['reviews']:,} reviews "
              f"(${t['price_usd']['min']:,.2f} to ${t['price_usd']['max']:,.2f}); "
              f"{review_stats['dropped_duplicates']} duplicate reviews skipped")
        if review_stats.get("dropped_duplicates_existing"):
            print(f"Removed {review_stats['dropped_duplicates_existing']} duplicate reviews "
                  "from the existing data")
    a = summary["all_data"]
    print(f"Products: {a['products']:,}   Reviews: {a['reviews']:,}")
    print("Price (USD): " + "  ".join(f"{k} {v:,.2f}" for k, v in a["price_usd"].items()))
    print("Products per price quintile:")
    print(products.groupby("price_quintile")["price"].agg(["count", "min", "max"])
          .round(2).to_string())
    if len(products) < 150:
        print("\nHeads-up: under 150 products. Consider a lower --min-ratings "
              "or a broader --keyword before moving to Phase 2.")
    print(f"\nSaved to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
