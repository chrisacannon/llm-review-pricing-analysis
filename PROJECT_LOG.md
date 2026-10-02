# Project log

What was done, what went wrong, and what changed as a result. Newest entries at the bottom of each phase. Costs are actual Claude API spend.

## Running totals (as of 2026-10-02)

| Item | Value |
| --- | --- |
| Products / reviews | 429 products / 14,039 reviews (14,030 tagged) |
| Excluded from band statistics | 39 products (price check + hand-check): 19 suspect listings, 19 not headphones, 1 multi-pack |
| Claude API spend | ~$5.76 (pilot $0.33, prompt re-checks $0.08, batch $3.19, Q&A tests ~$0.18, supplement batch ~$1.50 estimated, price check $0.42, app Q&A tests $0.04) |
| Tagging accuracy (hand-check, n=50) | value 96%, themes 86% (prompt v1); v3 fixed the main error patterns |

## Key decisions

| Decision | Why |
| --- | --- |
| Real reviews (Amazon Reviews 2023, McAuley Lab) instead of synthetic data | The point is analyzing real customer language; synthetic reviews would be Claude analyzing Claude |
| Headphones, filtered by Amazon category path | Wide price spread ($3 to $1,000) and lots of value talk; category path is cleaner than keyword matching (see Phase 1) |
| Stream the 28 GB Electronics files instead of downloading | Only the ~10 MB sample is kept on disk; a rerun takes ~13 minutes |
| Products sampled evenly across price quintiles | Every price tier needs enough products for the tier comparison |
| Claude Haiku 4.5 via the Batch API for tagging | Tagging is a classification task; Batch API halves the cost |
| Value-for-money sentiment tagged separately from overall sentiment | The project's question is perceived value, not satisfaction |
| Local embeddings (bge-small via fastembed) | No API cost; no PyTorch, so it stays light enough for Streamlit Community Cloud |
| Answers must cite review IDs, and citations are checked in code | Makes the Q&A verifiable and catches made-up citations |

## Phase 1: data (2026-09-30)

**Run 1**: `--keyword "headphone|earbud"`, 789 seconds. 300 products, 8,757 reviews, but the price spread looked wrong (80th percentile $39, max $15,022). Inspection showed only ~124 of 300 were headphones; the rest were cases, ear pads, adapters and cables. Cause: Amazon's category name "Headphones, Earbuds & Accessories" contains the keyword, so every accessory matched.

**Fix**: added `--category-match` (matches the category path only) and `--max-price`. Re-ran with `--category-match "Headphones & Earbuds" --max-price 1000`.

**Run 2**: 299 products, 9,619 reviews, all under Headphones & Earbuds (Earbud 184, Over-Ear 81, On-Ear 23, Open-Ear 5, other 6). Price $2.99 to $999; quintile cutoffs $14.99 / $23.99 / $39.51 / $79.00.

**Remaining noise, accepted** (under 2% of products): a cleaning kit, a Bluetooth beanie, a car radio; plus reseller markups such as a ~$20 Sony MDR-EX155AP listed at $803. The markups are useful for the "overpriced for its tier" analysis rather than a problem to remove.

## Phase 2: tagging (2026-09-30)

**Pilot (prompt v1)**: 500 random reviews, $0.33 at standard prices. Hand-check of 50: value 96% (48/50), themes 86% (43/50); target was 85%.

**Error patterns found in the hand-check**:

| Pattern | Example | Change |
| --- | --- | --- |
| Reviewer compares several products; Claude judged the others | r007703 | v2: judge only this product (still imperfect; rare) |
| Generic praise turned into a specific theme | "best earbuds yet" → sound_quality | v2: allow no themes when nothing specific is said |
| 3-theme cap dropped real topics | r003331 (controls, bass missed) | v2: up to 5 themes |
| Qualified praise tagged positive | "pretty good, slight hiss" | v2: qualified praise → mixed |
| Split value verdict tagged positive | "decent for $15 but they fall apart" | v2: genuinely split → neutral |
| No theme for looks/colors/lights (common for kids' headphones) | "other" at 8% | v2: added appearance_design |

**v2 regression**: "weigh the whole review" made Claude read regret ("I regret this purchase") as a negative value verdict.

**v3 fix**: a value verdict requires explicit talk of price, cost, money or worth; enforced in code (if `mentions_price` is false, value is forced to `not_mentioned`).

**What the re-check revealed**: three reviews approved in the hand-check (r000571, r007190, r002613) had v1 tags of "negative value" while also saying price was never mentioned. The hand-check missed these; the consistency rule caught them. Without it, the share of reviews calling headphones overpriced would have been overstated, and that is the project's headline metric.

**Full run (prompt v3, Batch API)**: 9,608 of 9,619 reviews tagged in ~5 minutes for $3.19 (11 missed, not retried).

| Value verdict | Share |
| --- | --- |
| Not mentioned | 66.7% |
| Positive (good value) | 24.0% |
| Negative (overpriced) | 7.1% (12.4% in the v1 pilot) |
| Neutral | 2.2% |

33.4% of reviews mention price. Top themes: sound quality 53%, comfort/fit 36%, build/durability 26%, connectivity 15%, battery 14%, appearance/design 13% (new in v2), value 13%. "Other" fell from 8.0% to 1.4%.

**First finding**: among reviews that give a value verdict, positive outnumbers negative about 3.4 to 1.

## Phase 3: search index and Q&A (2026-09-30)

Index built on the real data; first question answered with 9 valid citations for $0.02.

**Issue: "top tier" was too broad.** Quintiles split products into equal-count groups, and because the mix skews cheap (median ~$30), the top quintile ran from ~$80 to $999, mixing gaming headsets with Bang & Olufsen. Complaints from that group weren't meaningful as "premium" feedback.

**Fix:** named price bands with dollar thresholds, the way the category is merchandised: budget (under $25), value ($25-49), mid ($50-99), premium ($100-199), flagship ($200+). Quintiles are kept for equal-sample statistics; bands are used for filtering and Q&A.

**Band counts:** budget 127 products / 3,933 reviews; value 84 / 2,902; mid 45 / 1,393; premium 25 / 874; flagship 18 / 517. Flagship is thin, and includes at least one reseller markup (Sony MDR-EX155AP at $803).

**Flagship question re-run:** complaints at $355-$999 centered on Bluetooth reliability, weak noise cancelling, and service or unit condition; sound quality was rarely criticized. The answer flagged its own skew (most complaints were about Bang & Olufsen models) unprompted.

**10-question check:** 128 citations, 0 invalid (every cited ID was among the retrieved reviews), $0.16 total.

**Also fixed:** the first answer was cut off at the response length limit; raised the limit and the app now flags any answer that still gets cut off.

**Q&A check review (2026-10-01):** answers were well grounded, and Claude flagged two data issues on its own:
- **Duplicate reviews**: identical text listed twice for the same product (e.g. r002024/r002025, r001208/r001209, r007803/r007804, r001934/r001935). Fix: duplicate removal in Phase 1 (same product and text, by the same user or 40+ characters long; short generic texts from different people are kept).
- **Listed price vs. price paid**: reviewers of a $449.99 listing said "under $100" and of a $176.45 listing "$20". The dataset's price is a single snapshot, so band labels can be wrong for individual products. To address in Phase 4 with a price plausibility check.

## Band supplements (2026-10-01)

**Why:** with 18 products and 517 reviews, flagship ($200+) was too thin to stand on its own in band comparisons; premium (25 products) and mid (45) were also thin. Bands stay as they are; the goal is every band deep enough to compare.

**How:** three extra Phase 1 pulls (premium $100-199.99, flagship $200-1,000, mid $50-99.99), appended to the existing data (`--append`): existing products skipped, review IDs continued, previous files backed up, duplicates removed from old and new data. Only the new reviews are tagged (prompt v3, Batch API), then the index is rebuilt.

**Sampling note:** flagship is now deliberately oversampled relative to the first pull. Statistics should be reported per band, not pooled across all reviews.

**Results:**

| Pull | Eligible products | Added | Duplicates removed |
| --- | --- | --- | --- |
| Premium $100-199.99 | 7,171 | 40 products / 1,381 reviews | 7 new + 86 in existing data |
| Flagship $200-1,000 | 7,539 | 70 products / 2,481 reviews | 27 |
| Mid $50-99.99 | 6,526 | 20 products / 644 reviews | 5 |

Tagged 4,517 new reviews in one batch (prompt v3); 9 remain untagged. Index rebuilt: 14,039 reviews.

| Band | Products | Reviews | Mention price | Positive / negative among value verdicts |
| --- | --- | --- | --- | --- |
| Budget (under $25) | 127 | 3,898 | 32.9% | 71% / 21% (n=1,282) |
| Value ($25-49) | 84 | 2,873 | 31.0% | 74% / 20% (n=888) |
| Mid ($50-99) | 65 | 2,027 | 35.0% | 72% / 22% (n=708) |
| Premium ($100-199) | 65 | 2,248 | 31.3% | 62% / 32% (n=700) |
| Flagship ($200+) | 88 | 2,993 | 36.1% | 56% / 38% (n=1,077) |

**Emerging headline:** perceived value is flat from budget through mid-range (~72% positive), then drops at $100 and again at $200: flagship buyers who comment on value call it overpriced almost twice as often as budget buyers (38% vs 21%). Not yet adjusted for suspect listings (e.g. Sony MDR-EX155AP at $803, a Brookstone kids' cat-ear headphone at $639.99); the Phase 4 plausibility check comes first.

## Phase 4, step 1: price plausibility check (2026-10-01 to 10-02)

**Why:** listed price is a single snapshot, and reseller markups put products in the wrong band (a ~$20 Sony earbud at $803 counted as flagship). Those products' reviewers paid normal prices, so they diluted the flagship value signal.

**How:** Claude Haiku estimates each product's normal price range from its title, brand and up to 3 price-mentioning reviews, and says whether it's actually headphones (`pipeline/04_check_prices.py`). A rule in code marks a listing **suspect if its price is more than 2× the top of that range**. Suspect listings and non-headphones are excluded from band statistics (`exclude` in `price_checks.parquet`) and shown in the app as suspect listings. Hand-check decisions go in `price_overrides.csv` (committed).

**What changed along the way:**

| Run | Problem | Change |
| --- | --- | --- |
| v1 pilot (20 products) | 8/20 suspect; Claude flagged prices only ~1.1× its own range | 2× rule enforced in code, not left to the prompt |
| v1 pilot | A product was flagged because reviewers called it "overpriced"; removing those would understate the headline metric | v2 prompt: only prices reviewers paid or saw count as evidence, not opinions of worth |
| v1 pilot | Renewed listing judged against new price; DMX terminator flagged as mispriced because it isn't headphones | v2: judge renewed listings at renewed prices; price and product type judged separately |
| v2 pilot | Sony MDR-EX650 flipped from 1.8× to 3.1× between runs: Claude's estimates vary | Borderline listings (1.5–2.5×) printed for a hand-check |
| Full run | 11 of 26 suspects were *too cheap*, mostly generic earbuds priced by guessing from features | Flag overpricing only; a low listed price is usually a real sale price buyers paid |

**Hand-check (Chris):** 10 overrides. Excluded 5 markups between 1.6× and 2× that don't belong in flagship (Bose SoundTrue ×2, Shure E3c, Sony MDR-NC60, Sony XBA-N3). Kept the AKG N400 ($129, near list price). Excluded 2 multi-packs (pack price isn't comparable). Reclassified 2 products as headphones (Sennheiser RS 2000, Chevy Suburban rear-seat headphones).

**Result:** 39 of 429 products excluded: 19 suspect, 19 not headphones, 1 multi-pack. Cost $0.42 (pilots $0.04, full run $0.38).

| Band | Products | Reviews | Overpriced share before → after | n (value verdicts) |
| --- | --- | --- | --- | --- |
| Budget | 116 | 3,664 | 20.7% → 21.3% | 1,229 |
| Value | 81 | 2,734 | 20.0% → 19.8% | 857 |
| Mid | 61 | 1,925 | 21.8% → 22.6% | 674 |
| Premium | 61 | 2,091 | 32.1% → 32.3% | 660 |
| Flagship | 71 | 2,293 | 38.3% → 40.9% | 826 |

**Headline after cleanup:** flagship buyers who comment on value call it overpriced about twice as often as budget-through-mid buyers (41% vs ~21%). Removing the mispriced listings strengthened the flagship signal.

## Phase 4, steps 2–3: Streamlit app (2026-10-02)

**Analytics tab** (`app.py`, numbers in `analytics.py`; no API cost): headline, overpriced share by band with 95% intervals, value verdict mix, a band table with sample sizes, a theme heatmap (net sentiment and prevalence by band), an overpriced-for-band table with a review drill-down, and the excluded listings.

**Overpriced-for-band definition (Chris kept the default):** priced above the band median, at least 8 value verdicts, overpriced share above the band average: 34 products. "Clearly above" (the 95% interval's low end beats the band average): 15. Ranked by that low end.

**What the themes show:** as price rises, sound quality sentiment improves (+0.56 budget → +0.74 flagship) while value-for-money falls (+0.83 → +0.36); connectivity (−0.27) and customer service (−0.64) turn negative at flagship.

**Issue found:** the search index still held reviews of the 39 excluded products; a flagship search for "Sony earbuds with mic" returned 6 of 15 from excluded listings. **Fix:** retrieval leaves excluded products out by default (app and CLI). Claude now sees each review's price band instead of its quintile tier.

**Q&A tab:** example questions, band/topic/value/stars filters, citations that link to and open the cited review (with a hover preview), invalid-citation warnings, and cost per answer. Tested with a stand-in client (it caught a planted fake citation), then live: 2 questions, 29 citations, 0 invalid, $0.036.

**Observed:** retrieval can cluster on one product: 4 of 15 flagship-complaint reviews were the Master & Dynamic MW08 (5 of 15 Master & Dynamic overall). **Fix (Chris approved):** at most 3 reviews per product in each answer (`max_per_product` in `qa.retrieve`, filled from 3× the candidates). On the flagship question this trims the MW08 from 4 to 3; the budget question already had at most 2 per product.
