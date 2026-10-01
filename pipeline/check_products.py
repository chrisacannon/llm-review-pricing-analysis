"""Quick look at what Phase 1 picked: sample titles per price tier, the most expensive
items, and the most common category paths. Run from the project folder:

    python pipeline\\check_products.py
"""
from collections import Counter

import pandas as pd

p = pd.read_parquet("data/processed/products.parquet")
pd.set_option("display.width", 160, "display.max_colwidth", 90)

print("Sample titles per price tier")
for tier, g in p.groupby("price_quintile"):
    print(f"\n{tier}  (${g.price.min():,.2f} to ${g.price.max():,.2f})")
    for _, r in g.sample(min(8, len(g)), random_state=1).iterrows():
        print(f"  ${r.price:>9,.2f}  {r.title[:90]}")

print("\n\nMost expensive 8")
for _, r in p.nlargest(8, "price").iterrows():
    print(f"  ${r.price:>9,.2f}  {r.title[:90]}")

print("\n\nMost common category text (first 120 characters)")
for cats, n in Counter(p["categories"].str[:120]).most_common(12):
    print(f"  {n:>4}  {cats}")
