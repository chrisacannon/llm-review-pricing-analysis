"""
Band-level and product-level statistics for the Streamlit app's Analytics tab (Phase 4).

Everything here is plain pandas over data/processed/, with no API calls. Rules that shape
the numbers, all from CLAUDE.md:
  - Products with exclude == True in price_checks.parquet (suspect listings, non-headphones,
    hand-check exclusions) are left out of every statistic and listed separately.
  - Statistics are per band, never pooled: premium and flagship were deliberately oversampled.
  - Value verdict shares are among reviews that give a verdict (positive / negative / neutral),
    i.e. reviews that talk about price; the share mentioning price is shown alongside.

Check the numbers without the app:
    python analytics.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import datasource
from qa import PRICE_BANDS, band_label

DATA = datasource.data_dir()  # data/processed locally; app_data/ when deployed
BAND_ORDER = list(PRICE_BANDS)
VERDICTS = ["negative", "neutral", "positive"]  # reviews that give a value verdict
Z95 = 1.96


def band_of(price: float) -> str:
    for name, (lo, hi) in PRICE_BANDS.items():
        if price >= lo and (hi is None or price < hi):
            return name
    raise ValueError(f"No band for price {price}")


def wilson(k, n, z: float = Z95):
    """Wilson score interval for a proportion k/n; works on scalars or arrays."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z * z / n
        centre = p + z * z / (2 * n)
        margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return (centre - margin) / denom, (centre + margin) / denom


# ---------------------------------------------------------------- loading

def load() -> dict[str, pd.DataFrame]:
    """products (all, with band and exclude), reviews (included products, tagged only), themes."""
    products = pd.read_parquet(DATA / "products.parquet",
                               columns=["parent_asin", "title", "store", "price", "average_rating", "rating_number"])
    checks = pd.read_parquet(DATA / "price_checks.parquet")
    products = products.merge(
        checks[["parent_asin", "verdict", "typical_low", "typical_high", "is_headphone",
                "reason", "exclude", "override_note"]],
        on="parent_asin", how="left")
    products["exclude"] = products["exclude"].fillna(False).astype(bool)
    products["band"] = products["price"].map(band_of)

    reviews = pd.read_parquet(DATA / "reviews.parquet",
                              columns=["review_id", "parent_asin", "rating", "title", "text", "date"])
    tags = pd.read_parquet(DATA / "review_tags.parquet",
                           columns=["review_id", "value_sentiment", "mentions_price", "themes_json"])
    included = products.loc[~products["exclude"], ["parent_asin", "price", "band"]]
    reviews = reviews.merge(tags, on="review_id").merge(included, on="parent_asin")

    themes = pd.read_parquet(DATA / "review_themes.parquet")
    themes = themes.merge(reviews[["review_id", "band"]], on="review_id")
    return {"products": products, "reviews": reviews, "themes": themes}


# ---------------------------------------------------------------- band level

def band_summary(reviews: pd.DataFrame) -> pd.DataFrame:
    """One row per band: sample sizes, share mentioning price, verdict mix, overpriced CI."""
    rows = []
    for band in BAND_ORDER:
        b = reviews[reviews["band"] == band]
        v = b[b["value_sentiment"].isin(VERDICTS)]
        counts = v["value_sentiment"].value_counts()
        n = len(v)
        neg = int(counts.get("negative", 0))
        lo, hi = wilson(neg, n)
        rows.append({
            "band": band, "label": band_label(band),
            "products": b["parent_asin"].nunique(), "reviews": len(b),
            "mention_price": b["mentions_price"].mean(),
            "verdicts": n,
            "negative": neg / n, "neutral": counts.get("neutral", 0) / n, "positive": counts.get("positive", 0) / n,
            "negative_lo": float(lo), "negative_hi": float(hi),
        })
    return pd.DataFrame(rows)


def theme_by_band(reviews: pd.DataFrame, themes: pd.DataFrame) -> pd.DataFrame:
    """One row per (theme, band): prevalence among the band's reviews and net polarity.
    Net polarity = (positive - negative) / mentions, so mixed mentions pull toward zero."""
    per_band = reviews.groupby("band")["review_id"].nunique()
    t = themes[themes["theme"] != "other"]
    g = (t.groupby(["theme", "band", "polarity"])["review_id"].nunique()
         .unstack("polarity", fill_value=0).reset_index())
    for col in ("positive", "negative", "mixed"):
        if col not in g:
            g[col] = 0
    g["mentions"] = g["positive"] + g["negative"] + g["mixed"]
    g["band_reviews"] = g["band"].map(per_band)
    g["prevalence"] = g["mentions"] / g["band_reviews"]
    g["net"] = (g["positive"] - g["negative"]) / g["mentions"]
    return g


# ---------------------------------------------------------------- product level

def product_value(reviews: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Per included product: value verdict counts, overpriced share with CI, band context."""
    v = reviews[reviews["value_sentiment"].isin(VERDICTS)]
    g = v.groupby("parent_asin")["value_sentiment"].agg(
        verdicts="size", negative=lambda s: int((s == "negative").sum())).reset_index()
    g = g.merge(reviews.groupby("parent_asin").size().rename("reviews").reset_index(), on="parent_asin")
    inc = products[~products["exclude"]]
    g = g.merge(inc[["parent_asin", "title", "store", "price", "band", "average_rating"]], on="parent_asin")
    g["share"] = g["negative"] / g["verdicts"]
    g["share_lo"], g["share_hi"] = wilson(g["negative"], g["verdicts"])
    band_rate = v.groupby("band")["value_sentiment"].apply(lambda s: (s == "negative").mean())
    g["band_rate"] = g["band"].map(band_rate)
    g["band_median_price"] = g["band"].map(inc.groupby("band")["price"].median())
    return g


def overpriced_for_band(pv: pd.DataFrame, min_verdicts: int = 8) -> pd.DataFrame:
    """Products priced above their band's median whose overpriced share beats the band average.
    clearly_above: the 95% interval's lower bound is also above the band average, i.e. not
    plausibly noise from a handful of verdicts. Ranked by that lower bound."""
    t = pv[(pv["price"] > pv["band_median_price"]) & (pv["verdicts"] >= min_verdicts)
           & (pv["share"] > pv["band_rate"])].copy()
    t["clearly_above"] = t["share_lo"] > t["band_rate"]
    t["band_order"] = t["band"].map(BAND_ORDER.index)
    return t.sort_values(["share_lo", "share"], ascending=False).drop(columns="band_order")


def product_reviews(reviews: pd.DataFrame, parent_asin: str) -> pd.DataFrame:
    """A product's reviews that give a value verdict, overpriced first."""
    r = reviews[(reviews["parent_asin"] == parent_asin) & reviews["value_sentiment"].isin(VERDICTS)].copy()
    r["order"] = r["value_sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})
    return r.sort_values(["order", "review_id"]).drop(columns="order")


def suspect_listings(products: pd.DataFrame) -> pd.DataFrame:
    """Excluded products with the reason they were left out."""
    x = products[products["exclude"]].copy()

    def why(r):
        if isinstance(r.override_note, str) and r.is_headphone and r.verdict != "suspect":
            return "Hand-check exclusion"
        if not r.is_headphone:
            return "Not headphones"
        return "Suspect price"

    x["why"] = [why(r) for r in x.itertuples()]
    x["why_order"] = x["why"].map({"Suspect price": 0, "Hand-check exclusion": 1, "Not headphones": 2})
    return x.sort_values(["why_order", "price"], ascending=[True, False]).drop(columns="why_order")


# ---------------------------------------------------------------- check from the command line

def main() -> None:
    d = load()
    pd.set_option("display.width", 200)
    s = band_summary(d["reviews"])
    print(s[["label", "products", "reviews", "mention_price", "verdicts", "positive", "neutral",
             "negative", "negative_lo", "negative_hi"]].round(3).to_string(index=False))
    pv = product_value(d["reviews"], d["products"])
    op = overpriced_for_band(pv)
    print(f"\nOverpriced for band (min 8 verdicts): {len(op)} products, {op['clearly_above'].sum()} clearly above")
    print(f"Excluded listings: {len(suspect_listings(d['products']))}")
    tb = theme_by_band(d["reviews"], d["themes"])
    print("\nNet polarity by theme and band:")
    print(tb.pivot(index="theme", columns="band", values="net")[BAND_ORDER].round(2).to_string())


if __name__ == "__main__":
    main()
