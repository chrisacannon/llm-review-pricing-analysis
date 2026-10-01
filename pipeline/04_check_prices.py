"""
Phase 4 step 1: price plausibility check with Claude Haiku.

The dataset's listed price is a single snapshot and is sometimes wrong (third-party
reseller markups, e.g. a ~$20 Sony earbud listed at $803). For each product Claude sees
the title, brand, category, listed price and a few reviews that mention price, and returns:

  verdict        plausible / suspect / unsure
                 suspect = the listed price is clearly far from what this product
                 normally sold for new, so its band label is probably wrong
  typical_low/high  Claude's estimate of the normal new price range (USD)
  is_headphone   false for items that aren't headphones or earbuds (cables, terminators, beanies)
  reason         one sentence

Suspect listings are excluded from band statistics and shown in the app as "suspect listings".
Non-headphones are flagged separately; whether to exclude them is a judgment call for Chris.

Workflow (run from the project folder, environment active, API key set):

  1. Pilot on known problem listings plus a random sample per band:
       python pipeline\\04_check_prices.py pilot
  2. Check everything not yet checked:
       python pipeline\\04_check_prices.py run
  3. Rebuild the output and print the summary any time:
       python pipeline\\04_check_prices.py summary

Resumable: products already checked are skipped. Outputs in data/processed/:
  price_checks.jsonl     raw log, one line per checked product (the source of truth)
  price_checks.parquet   one row per product, latest check wins
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

DATA = Path("data/processed")
CHECKS_LOG = DATA / "price_checks.jsonl"

MODEL = "claude-haiku-4-5-20251001"
PRODUCTS_PER_REQUEST = 10
REVIEW_SNIPPETS = 3        # price-mentioning reviews shown per product
SNIPPET_CHARS = 300
# USD per million tokens, Claude Haiku 4.5 (checked 2026-09-30)
PRICE_STANDARD = (1.00, 5.00)
PROMPT_VERSION = "v1"

# Known problem listings from the project log, always included in the pilot
PILOT_KNOWN = [
    "B073JQXC1J",  # Sony MDR-EX155AP at $803 (~$20 earbud)
    "B01NCIGVK9",  # Brookstone Ariana Grande cat-ear headphones at $639.99
    "B01JP436TS",  # Sennheiser HD 598 Cs at $449.99; reviewers say "under $100"
    "B00NOHT69M",  # A-Audio A11 at $176.45; a reviewer says "$20"
    "B07C512SGP",  # Monoprice DMX terminator at $2.99 (not a headphone)
]

# Same thresholds as qa.py PRICE_BANDS, used here only to stratify the pilot sample
BANDS = [("budget", 0, 25), ("value", 25, 50), ("mid", 50, 100),
         ("premium", 100, 200), ("flagship", 200, float("inf"))]

SYSTEM = """You check whether Amazon headphone listings have a believable price, for a pricing analysis
that groups products into price bands. The listed price is a single snapshot from around 2023 and is
sometimes wrong, mostly because a third-party reseller listed a cheap product at a large markup.

For each product judge whether the LISTED price is in line with what this product normally sold for
NEW at mainstream retail (around its release through 2023).
- plausible: the listed price is within the normal range, including ordinary sales, list prices
  and the normal premium of a flagship model.
- suspect: the listed price is clearly far from the normal new price, roughly more than double or less
  than half, so the product would land in the wrong price band. Typical case: a $20 earbud listed at $300+.
- unsure: you don't recognize the product and nothing in the listing or reviews tells you its price.
  Do not guess suspect for obscure brands just because the price seems high or low.

Use the reviews as evidence: reviewers often say what they paid ("got these for $20 on sale").
One reviewer's sale price is weak evidence; several reviewers far from the listing, or a well-known
model with a well-known price, is strong evidence.

Also say whether the product is actually headphones or earbuds (including headsets and kids' headphones).
Cables, adapters, cases, ear pads, cleaning kits, speakers, beanies and car stereos are not.

Give typical_low and typical_high as your estimate of the normal new price in USD (null if unsure),
and a one-sentence reason. Check every product_id you are given, exactly once."""

TOOL = {
    "name": "record_checks",
    "description": "Record the price check for every product in the request.",
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string"},
                        "verdict": {"type": "string", "enum": ["plausible", "suspect", "unsure"]},
                        "typical_low": {"type": ["number", "null"]},
                        "typical_high": {"type": ["number", "null"]},
                        "is_headphone": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["product_id", "verdict", "typical_low", "typical_high",
                                 "is_headphone", "reason"],
                },
            }
        },
        "required": ["results"],
    },
}


# ---------------------------------------------------------------- data helpers

def band_of(price: float) -> str:
    return next(name for name, lo, hi in BANDS if lo <= price < hi)


def load_products() -> pd.DataFrame:
    products = pd.read_parquet(DATA / "products.parquet")
    reviews = pd.read_parquet(DATA / "reviews.parquet", columns=["review_id", "parent_asin", "rating", "text"])
    tags = pd.read_parquet(DATA / "review_tags.parquet", columns=["review_id", "mentions_price"])
    priced = reviews.merge(tags, on="review_id")
    priced = priced[priced["mentions_price"]]
    # Prefer reviews that state a dollar amount; they carry the most price evidence
    priced = priced.assign(has_dollar=priced["text"].str.contains(r"\$\s?\d", regex=True))
    priced = priced.sort_values(["parent_asin", "has_dollar", "review_id"], ascending=[True, False, True])
    snippets = (priced.groupby("parent_asin")
                .head(REVIEW_SNIPPETS)
                .groupby("parent_asin")
                .apply(lambda g: [(r.rating, str(r.text)[:SNIPPET_CHARS]) for r in g.itertuples()],
                       include_groups=False)
                .rename("snippets"))
    products = products.join(snippets, on="parent_asin")
    products["snippets"] = products["snippets"].apply(lambda s: s if isinstance(s, list) else [])
    products["band"] = products["price"].apply(band_of)
    return products


def checked_ids() -> set[str]:
    if not CHECKS_LOG.exists():
        return set()
    ids = set()
    with open(CHECKS_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["parent_asin"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def chunks(df: pd.DataFrame, size: int):
    for i in range(0, len(df), size):
        yield df.iloc[i:i + size]


def build_params(chunk: pd.DataFrame) -> dict:
    parts = []
    for p in chunk.itertuples():
        category = str(p.categories or "").split(" > ")[-1]
        reviews = "\n".join(f"  [{rating} stars] {text}" for rating, text in p.snippets) or "  (none)"
        parts.append(
            f"<product id=\"{p.parent_asin}\" listed_price_usd=\"{p.price:.2f}\">\n"
            f"Title: {p.title}\nBrand/store: {p.store}\nCategory: {category}\n"
            f"Reviews that mention price:\n{reviews}\n</product>"
        )
    return {
        "model": MODEL,
        "max_tokens": 4096,
        "system": SYSTEM,
        "tools": [TOOL],
        "tool_choice": {"type": "tool", "name": "record_checks"},
        "messages": [{"role": "user", "content": "\n\n".join(parts)}],
    }


def parse_message(message, chunk: pd.DataFrame) -> list[dict]:
    """Pull valid results for the expected products out of a Claude response."""
    prices = dict(zip(chunk["parent_asin"], chunk["price"]))
    out, seen = [], set()
    for block in message.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        for item in (block.input or {}).get("results", []):
            pid = item.get("product_id")
            if pid not in prices or pid in seen:
                continue
            if item.get("verdict") not in ("plausible", "suspect", "unsure"):
                continue
            out.append({"parent_asin": pid, "listed_price": prices[pid],
                        "verdict": item["verdict"],
                        "typical_low": item.get("typical_low"), "typical_high": item.get("typical_high"),
                        "is_headphone": bool(item.get("is_headphone", True)),
                        "reason": str(item.get("reason", ""))})
            seen.add(pid)
    return out


def append_checks(rows: list[dict], source: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with open(CHECKS_LOG, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({**row, "model": MODEL, "prompt": PROMPT_VERSION, "source": source}) + "\n")


def cost(input_tokens: int, output_tokens: int, prices: tuple[float, float]) -> float:
    return input_tokens / 1e6 * prices[0] + output_tokens / 1e6 * prices[1]


def client():
    try:
        import anthropic
    except ImportError:
        sys.exit("The anthropic package isn't installed. Run: pip install -r requirements.txt")
    try:
        return anthropic.Anthropic(max_retries=5)
    except Exception as e:  # missing key
        sys.exit(f"Couldn't create the Claude client: {e}\n"
                 "Set your key first, e.g. in PowerShell:  $env:ANTHROPIC_API_KEY = \"sk-ant-...\"")


# ---------------------------------------------------------------- outputs

def finalize(show_ids: set[str] | None = None) -> None:
    """Rebuild the parquet output from the log and print a summary.
    show_ids: print the full result for these products (used by the pilot)."""
    if not CHECKS_LOG.exists():
        print("No checks yet.")
        return
    rows = {}
    with open(CHECKS_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                rows[r["parent_asin"]] = r  # latest wins
            except (json.JSONDecodeError, KeyError):
                pass
    products = pd.read_parquet(DATA / "products.parquet", columns=["parent_asin", "title", "price"])
    checks = pd.DataFrame(rows.values())
    checks = checks[checks["parent_asin"].isin(products["parent_asin"])]
    checks.to_parquet(DATA / "price_checks.parquet", index=False)

    df = checks.merge(products, on="parent_asin")
    df["band"] = df["price"].apply(band_of)
    print(f"\nChecked {len(df):,} of {len(products):,} products")
    print("\nVerdict by band (listed price):")
    print(pd.crosstab(df["band"], df["verdict"], margins=True)
          .reindex([b for b, _, _ in BANDS] + ["All"]).fillna(0).astype(int).to_string())
    print(f"\nNot headphones: {(~df['is_headphone']).sum()}")

    flagged = df[(df["verdict"] == "suspect") | ~df["is_headphone"]]
    if show_ids is not None:
        flagged = df[df["parent_asin"].isin(show_ids)]
    pd.set_option("display.width", 200)
    for r in flagged.sort_values("price", ascending=False).itertuples():
        typical = (f"${r.typical_low:,.0f}-{r.typical_high:,.0f}"
                   if pd.notna(r.typical_low) and pd.notna(r.typical_high) else "?")
        tag = "" if r.is_headphone else "  [NOT HEADPHONE]"
        print(f"\n  {r.verdict.upper():9} ${r.price:,.2f} (typical {typical}){tag}  {r.parent_asin}")
        print(f"    {str(r.title)[:90]}")
        print(f"    {r.reason}")


# ---------------------------------------------------------------- commands

def check(todo: pd.DataFrame, source: str) -> float:
    c = client()
    tokens_in = tokens_out = checked = 0
    start = time.time()
    batches = list(chunks(todo, PRODUCTS_PER_REQUEST))
    for i, chunk in enumerate(batches, 1):
        msg = c.messages.create(**build_params(chunk))
        rows = parse_message(msg, chunk)
        append_checks(rows, source)
        checked += len(rows)
        tokens_in += msg.usage.input_tokens
        tokens_out += msg.usage.output_tokens
        print(f"  request {i}/{len(batches)}: {len(rows)}/{len(chunk)} checked ({time.time() - start:,.0f}s)")
    spent = cost(tokens_in, tokens_out, PRICE_STANDARD)
    print(f"\n{checked:,} products checked, {tokens_in:,} input + {tokens_out:,} output tokens, cost ${spent:.3f}")
    if checked < len(todo):
        print(f"{len(todo) - checked} missed; run again to retry just those.")
    return spent / max(checked, 1)


def cmd_pilot(args) -> None:
    products = load_products()
    known = products[products["parent_asin"].isin(PILOT_KNOWN)]
    rest = products[~products["parent_asin"].isin(PILOT_KNOWN)]
    sample = rest.groupby("band", group_keys=False).apply(
        lambda g: g.sample(min(args.per_band, len(g)), random_state=args.seed), include_groups=False)
    sample = products.loc[sample.index]
    todo = pd.concat([known, sample])
    print(f"Pilot: {len(known)} known problem listings + {len(sample)} random ({args.per_band} per band)")
    per_product = check(todo, "pilot")
    remaining = len(products) - len(checked_ids())
    print(f"Estimated cost for the remaining {remaining:,} products: ${per_product * remaining:.2f}")
    finalize(show_ids=set(todo["parent_asin"]))


def cmd_run(args) -> None:
    products = load_products()
    todo = products[~products["parent_asin"].isin(checked_ids())]
    if todo.empty:
        print("Every product is already checked.")
    else:
        print(f"Checking {len(todo):,} products")
        check(todo, "run")
    finalize()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("pilot", help="Check the known problem listings plus a random sample per band")
    sp.add_argument("--per-band", type=int, default=3)
    sp.add_argument("--seed", type=int, default=7)
    sub.add_parser("run", help="Check every product not yet checked (standard API)")
    sub.add_parser("summary", help="Rebuild the output and print the summary")
    args = p.parse_args(argv)
    {"pilot": cmd_pilot, "run": cmd_run, "summary": lambda a: finalize()}[args.cmd](args)


if __name__ == "__main__":
    main()
