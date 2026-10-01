"""
Phase 2: tag every review with Claude Haiku.

Each review gets:
  value_sentiment   positive / negative / neutral / not_mentioned
                    (the reviewer's view of value for money, not overall sentiment)
  mentions_price    true if the review talks about price, cost or value at all
  themes            up to 3 themes from THEMES below, each positive / negative / mixed

Workflow (run from the project folder, environment active, API key set):

  1. Pilot on 500 random reviews (standard API, a couple of minutes, ~$0.25):
       python pipeline\\02_tag_reviews.py pilot --n 500
  2. Export 50 tagged reviews to a CSV and hand-check them in Excel:
       python pipeline\\02_tag_reviews.py export-check --n 50
  3. Tag everything else through the Batch API (half price, usually done within an hour):
       python pipeline\\02_tag_reviews.py batch-submit
       python pipeline\\02_tag_reviews.py batch-collect --wait

Everything is resumable: reviews already tagged are skipped, so re-running a step
never pays twice. Results land in data/processed/:
  tags.jsonl              raw log, one line per tagged review (the source of truth)
  review_tags.parquet     one row per review
  review_themes.parquet   one row per (review, theme) for the tier analysis
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

DATA = Path("data/processed")
TAGS_LOG = DATA / "tags.jsonl"
BATCH_STATE = DATA / "batch_state.json"

MODEL = "claude-haiku-4-5-20251001"
REVIEWS_PER_REQUEST = 25
MAX_REVIEW_CHARS = 1500
# USD per million tokens, Claude Haiku 4.5 (checked 2026-09-30)
PRICE_STANDARD = (1.00, 5.00)
PRICE_BATCH = (0.50, 2.50)

# Edit this list if the pilot shows a gap; keep "other" as the catch-all.
THEMES = [
    "sound_quality",        # clarity, detail, soundstage, overall audio
    "bass",                 # bass amount or quality specifically
    "noise_cancelling",     # ANC or passive isolation
    "comfort_fit",          # comfort, fit, ear/head pain, stays in during sports
    "build_durability",     # materials, breaking, lifespan
    "battery_life",
    "connectivity",         # Bluetooth pairing, dropouts, latency, range
    "microphone_calls",
    "controls_app",         # buttons, touch controls, companion app
    "value_for_money",
    "authenticity_condition",  # counterfeit, used-as-new, wrong item, packaging
    "customer_service",     # seller, returns, warranty
    "appearance_design",    # looks, colors, lights, style
    "other",
]
MAX_THEMES = 5
PROMPT_VERSION = "v3"

SYSTEM = f"""You tag Amazon headphone reviews for a pricing analysis.

Judge only what the reviewer says about THIS product. Some reviewers compare several
headphones they tried; ignore their verdicts on the other products.

For each review return:
- value_sentiment: the reviewer's view of VALUE FOR MONEY for this product.
  positive = worth the price / great deal; negative = overpriced / not worth it / waste of money;
  neutral = mentions price or value without a clear verdict, or the verdict is genuinely split
  (e.g. "decent sound for $15 but they fell apart"); not_mentioned = says nothing about price or value.
  A value verdict needs the reviewer to talk about price, cost, money, deals or whether it was worth it.
  Regret, disappointment or praise alone ("I regret this purchase", "love them") is not_mentioned.
  A glowing review that never mentions price or value is not_mentioned.
- mentions_price: true if the review refers to price, cost, value, deals or money at all.
  If mentions_price is false, value_sentiment must be not_mentioned.
- themes: up to {MAX_THEMES} specific aspects the review actually discusses, most important first.
  Use only these themes: {", ".join(THEMES)}.
  Polarity is the reviewer's satisfaction with that aspect: positive, negative, or mixed
  (mixed includes qualified praise like "pretty good, but a slight hiss").
  Do not infer a theme from generic praise or complaints ("best earbuds ever", "my kid loves them");
  return an empty list when the review names no specific aspect, or is too short or garbled to tell.
  Use "other" only for a specific aspect that fits none of the themes.

Use the star rating to read sarcasm and mixed reviews correctly. Tag every review_id you are given, exactly once."""

TOOL = {
    "name": "record_tags",
    "description": "Record the tags for every review in the request.",
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "review_id": {"type": "string"},
                        "value_sentiment": {"type": "string",
                                            "enum": ["positive", "negative", "neutral", "not_mentioned"]},
                        "mentions_price": {"type": "boolean"},
                        "themes": {
                            "type": "array",
                            "maxItems": MAX_THEMES,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "theme": {"type": "string", "enum": THEMES},
                                    "polarity": {"type": "string",
                                                 "enum": ["positive", "negative", "mixed"]},
                                },
                                "required": ["theme", "polarity"],
                            },
                        },
                    },
                    "required": ["review_id", "value_sentiment", "mentions_price", "themes"],
                },
            }
        },
        "required": ["results"],
    },
}


# ---------------------------------------------------------------- data helpers

def load_reviews() -> pd.DataFrame:
    reviews = pd.read_parquet(DATA / "reviews.parquet")
    products = pd.read_parquet(DATA / "products.parquet")[["parent_asin", "title", "price"]]
    products = products.rename(columns={"title": "product_title"})
    return reviews.merge(products, on="parent_asin", how="left")


def tagged_ids() -> set[str]:
    if not TAGS_LOG.exists():
        return set()
    ids = set()
    with open(TAGS_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["review_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def chunks(df: pd.DataFrame, size: int):
    for i in range(0, len(df), size):
        yield df.iloc[i:i + size]


def build_params(chunk: pd.DataFrame) -> dict:
    parts = []
    for r in chunk.itertuples():
        text = str(r.text)[:MAX_REVIEW_CHARS]
        parts.append(
            f"<review id=\"{r.review_id}\" stars=\"{r.rating}\" "
            f"product=\"{str(r.product_title)[:80]}\" price_usd=\"{r.price:.2f}\">\n"
            f"{str(r.title or '')}\n{text}\n</review>"
        )
    return {
        "model": MODEL,
        "max_tokens": 4096,
        "system": SYSTEM,
        "tools": [TOOL],
        "tool_choice": {"type": "tool", "name": "record_tags"},
        "messages": [{"role": "user", "content": "\n\n".join(parts)}],
    }


def parse_message(message, expected_ids: set[str]) -> list[dict]:
    """Pull valid results for the expected review IDs out of a Claude response."""
    out, seen = [], set()
    for block in message.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        for item in (block.input or {}).get("results", []):
            rid = item.get("review_id")
            if rid not in expected_ids or rid in seen:
                continue
            themes = [t for t in item.get("themes", [])
                      if t.get("theme") in THEMES and t.get("polarity") in ("positive", "negative", "mixed")][:MAX_THEMES]
            vs = item.get("value_sentiment")
            if vs not in ("positive", "negative", "neutral", "not_mentioned"):
                continue
            if not item.get("mentions_price"):
                vs = "not_mentioned"  # a value verdict without any price talk is inconsistent
            out.append({"review_id": rid, "value_sentiment": vs,
                        "mentions_price": bool(item.get("mentions_price")), "themes": themes})
            seen.add(rid)
    return out


def append_tags(rows: list[dict], source: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with open(TAGS_LOG, "a", encoding="utf-8") as f:
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

def finalize() -> None:
    """Rebuild the parquet outputs from the log and print a summary."""
    if not TAGS_LOG.exists():
        print("No tags yet.")
        return
    rows = {}
    with open(TAGS_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                rows[r["review_id"]] = r  # latest wins
            except (json.JSONDecodeError, KeyError):
                pass
    current = set(pd.read_parquet(DATA / "reviews.parquet", columns=["review_id"])["review_id"])
    rows = {k: v for k, v in rows.items() if k in current}  # drop tags for removed reviews
    tags = pd.DataFrame(rows.values())
    tags["themes_json"] = tags["themes"].apply(json.dumps)
    tags.drop(columns=["themes"]).to_parquet(DATA / "review_tags.parquet", index=False)

    themes = pd.DataFrame([{"review_id": r["review_id"], "rank": i + 1, **t}
                           for r in rows.values() for i, t in enumerate(r["themes"])])
    themes.to_parquet(DATA / "review_themes.parquet", index=False)

    total = len(pd.read_parquet(DATA / "reviews.parquet", columns=["review_id"]))
    print(f"\nTagged {len(tags):,} of {total:,} reviews")
    print("\nValue sentiment:")
    print(tags["value_sentiment"].value_counts(normalize=True).round(3).to_string())
    print(f"\nMentions price: {tags['mentions_price'].mean():.1%}")
    if not themes.empty:
        print("\nTop themes (share of tagged reviews):")
        print((themes.groupby("theme")["review_id"].nunique() / len(tags))
              .sort_values(ascending=False).round(3).to_string())


# ---------------------------------------------------------------- commands

def cmd_pilot(args) -> None:
    reviews = load_reviews()
    done = tagged_ids()
    todo = reviews[~reviews["review_id"].isin(done)]
    n = min(args.n, len(todo))
    todo = todo.sample(n, random_state=args.seed)
    if todo.empty:
        print("Nothing left to tag.")
        return
    c = client()
    tokens_in = tokens_out = 0
    tagged = 0
    start = time.time()
    batches = list(chunks(todo, REVIEWS_PER_REQUEST))
    for i, chunk in enumerate(batches, 1):
        msg = c.messages.create(**build_params(chunk))
        rows = parse_message(msg, set(chunk["review_id"]))
        append_tags(rows, "pilot")
        tagged += len(rows)
        tokens_in += msg.usage.input_tokens
        tokens_out += msg.usage.output_tokens
        print(f"  request {i}/{len(batches)}: {len(rows)}/{len(chunk)} tagged "
              f"({time.time() - start:,.0f}s)")
    spent = cost(tokens_in, tokens_out, PRICE_STANDARD)
    per_review = spent / max(tagged, 1)
    remaining = len(reviews) - len(done) - tagged
    print(f"\nPilot: {tagged:,} reviews tagged, {tokens_in:,} input + {tokens_out:,} output tokens")
    print(f"Cost: ${spent:.2f} (${per_review * 1000:.2f} per 1,000 reviews at standard prices)")
    print(f"Estimated cost for the remaining {remaining:,} via the Batch API: "
          f"${per_review / 2 * remaining:.2f}")
    finalize()


def cmd_export_check(args) -> None:
    tags = pd.read_parquet(DATA / "review_tags.parquet")
    reviews = load_reviews()
    df = tags.merge(reviews, on="review_id").sample(min(args.n, len(tags)), random_state=args.seed)
    df["themes"] = df["themes_json"].apply(
        lambda s: "; ".join(f"{t['theme']} ({t['polarity']})" for t in json.loads(s)))
    out = df[["review_id", "rating", "price", "product_title", "title", "text",
              "value_sentiment", "mentions_price", "themes"]].copy()
    out["value_ok (Y/N)"] = ""
    out["themes_ok (Y/N)"] = ""
    out["notes"] = ""
    path = DATA / "hand_check.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel reads it cleanly
    print(f"Wrote {len(out)} reviews to {path.resolve()}")
    print("Fill in the Y/N columns; aim for 85%+ Y on each.")


def cmd_retag_check(args) -> None:
    """Re-tag the hand-checked reviews with the current prompt and compare."""
    path = DATA / "hand_check.csv"
    for enc in ("utf-8-sig", "cp1252"):
        try:
            check = pd.read_csv(path, encoding=enc, dtype=str)
            break
        except UnicodeDecodeError:
            continue
    check = check[check["review_id"].fillna("").str.startswith("r")]
    reviews = load_reviews()
    todo = reviews[reviews["review_id"].isin(check["review_id"])]
    c = client()
    new, tokens_in, tokens_out = {}, 0, 0
    for chunk in chunks(todo, REVIEWS_PER_REQUEST):
        msg = c.messages.create(**build_params(chunk))
        for row in parse_message(msg, set(chunk["review_id"])):
            new[row["review_id"]] = row
        tokens_in += msg.usage.input_tokens
        tokens_out += msg.usage.output_tokens

    def fmt(themes):
        return "; ".join(f"{t['theme']} ({t['polarity']})" for t in themes) or "(none)"

    check["new_value_sentiment"] = check["review_id"].map(lambda r: new.get(r, {}).get("value_sentiment", ""))
    check["new_themes"] = check["review_id"].map(lambda r: fmt(new[r]["themes"]) if r in new else "")
    cols = ["review_id", "rating", "text", "value_sentiment", "new_value_sentiment",
            "themes", "new_themes", "value_ok (Y/N)", "themes_ok (Y/N)", "notes"]
    out = DATA / "hand_check_v2.csv"
    check[cols].to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Re-tagged {len(new)} reviews with prompt {PROMPT_VERSION} "
          f"(${cost(tokens_in, tokens_out, PRICE_STANDARD):.2f}). Rows you marked N:\n")
    flagged = check[(check["value_ok (Y/N)"].str.upper() == "N") | (check["themes_ok (Y/N)"].str.upper() == "N")]
    for r in flagged.itertuples():
        print(f"{r.review_id}  (your note: {str(r.notes)[:100]})")
        print(f"   value:  {r.value_sentiment}  ->  {r.new_value_sentiment}")
        print(f"   themes: {r.themes or '(none)'}")
        print(f"       ->  {r.new_themes}\n")
    changed_rows = check[(check["value_sentiment"] != check["new_value_sentiment"])
                         & ~check["review_id"].isin(flagged["review_id"])]
    if len(changed_rows):
        print("Value changed on rows you marked Y (check these aren't regressions):\n")
        for r in changed_rows.itertuples():
            print(f"{r.review_id}  {r.value_sentiment}  ->  {r.new_value_sentiment}")
            print(f"   \"{' '.join(str(r.text).split())[:220]}\"\n")
    changed = (check["value_sentiment"] != check["new_value_sentiment"]).sum()
    print(f"Value changed on {changed} of {len(check)} rows; full comparison in {out.resolve()}")


def cmd_batch_submit(args) -> None:
    if BATCH_STATE.exists() and not args.force:
        state = json.loads(BATCH_STATE.read_text())
        sys.exit(f"Batch {state['batch_id']} is already submitted. Run batch-collect, "
                 "or add --force to submit another.")
    reviews = load_reviews()
    todo = reviews if args.all else reviews[~reviews["review_id"].isin(tagged_ids())]
    if todo.empty:
        print("Nothing left to tag.")
        return
    requests = [{"custom_id": f"req{i:05d}", "params": build_params(chunk)}
                for i, chunk in enumerate(chunks(todo, REVIEWS_PER_REQUEST))]
    batch = client().messages.batches.create(requests=requests)
    BATCH_STATE.write_text(json.dumps({
        "batch_id": batch.id,
        "submitted": time.strftime("%Y-%m-%d %H:%M"),
        "requests": len(requests),
        "reviews": len(todo),
        "ids_by_request": {r["custom_id"]: list(chunk["review_id"])
                           for r, chunk in zip(requests, chunks(todo, REVIEWS_PER_REQUEST))},
    }))
    print(f"Submitted batch {batch.id}: {len(todo):,} reviews in {len(requests):,} requests.")
    print("Most batches finish within an hour. Then run:  python pipeline\\02_tag_reviews.py batch-collect --wait")


def cmd_batch_collect(args) -> None:
    if not BATCH_STATE.exists():
        sys.exit("No submitted batch found. Run batch-submit first.")
    state = json.loads(BATCH_STATE.read_text())
    c = client()
    while True:
        batch = c.messages.batches.retrieve(state["batch_id"])
        counts = batch.request_counts
        print(f"  {time.strftime('%H:%M')} status {batch.processing_status}: "
              f"{counts.succeeded} succeeded, {counts.errored} errored, {counts.processing} processing")
        if batch.processing_status == "ended":
            break
        if not args.wait:
            print("Not finished yet. Run again later, or add --wait to keep checking.")
            return
        time.sleep(60)

    tokens_in = tokens_out = 0
    tagged = failed = 0
    for entry in c.messages.batches.results(state["batch_id"]):
        expected = set(state["ids_by_request"].get(entry.custom_id, []))
        if entry.result.type != "succeeded":
            failed += len(expected)
            continue
        msg = entry.result.message
        rows = parse_message(msg, expected)
        append_tags(rows, "batch")
        tagged += len(rows)
        failed += len(expected) - len(rows)
        tokens_in += msg.usage.input_tokens
        tokens_out += msg.usage.output_tokens
    BATCH_STATE.rename(BATCH_STATE.with_name(f"batch_done_{state['batch_id']}.json"))
    print(f"\nBatch: {tagged:,} reviews tagged, {failed:,} missed; "
          f"cost ${cost(tokens_in, tokens_out, PRICE_BATCH):.2f}")
    if failed:
        print("Missed reviews stay untagged. Run batch-submit again to retry just those.")
    finalize()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("pilot", help="Tag a random sample with the standard API")
    sp.add_argument("--n", type=int, default=500)
    sp.add_argument("--seed", type=int, default=7)
    se = sub.add_parser("export-check", help="Write a CSV of tagged reviews to hand-check")
    se.add_argument("--n", type=int, default=50)
    se.add_argument("--seed", type=int, default=11)
    sb = sub.add_parser("batch-submit", help="Submit all untagged reviews to the Batch API")
    sb.add_argument("--force", action="store_true")
    sb.add_argument("--all", action="store_true",
                    help="Re-tag every review, including ones already tagged (e.g. after a prompt change)")
    sub.add_parser("retag-check", help="Re-tag the hand-checked reviews with the current prompt and compare")
    sc = sub.add_parser("batch-collect", help="Download batch results")
    sc.add_argument("--wait", action="store_true", help="Keep checking every minute until done")
    sub.add_parser("summary", help="Rebuild outputs and print the tag summary")
    args = p.parse_args(argv)

    {"pilot": cmd_pilot, "export-check": cmd_export_check, "batch-submit": cmd_batch_submit,
     "batch-collect": cmd_batch_collect, "retag-check": cmd_retag_check,
     "summary": lambda a: finalize()}[args.cmd](args)


if __name__ == "__main__":
    main()
