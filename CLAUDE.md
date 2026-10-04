# CLAUDE.md

Context for Claude Code. Read this, then `PROJECT_LOG.md` (what happened and why) and `README.md` (how to run each phase).

## What this is

A portfolio project for Chris, a pricing and strategy professional moving toward AI/LLM tooling roles. It answers: **for headphones on Amazon, where is price out of line with perceived value, and what do customers say at each price band?** The framing is portfolio-level pricing strategy, not deal-by-deal analysis. Findings should read like pricing analysis, not a generic chatbot demo.

Data: Amazon Reviews 2023 (McAuley Lab, UCSD), Electronics category, filtered to the "Headphones & Earbuds" category path.

## Status

- Phase 1 (data), Phase 2 (Claude Haiku tagging, prompt v3), Phase 3 (local embeddings + cited Q&A): done.
- Band supplements (deeper mid, premium and flagship): done, tagged and indexed.
- Phase 4 step 1, price plausibility check: done (39 products excluded; see the log).
- Phase 4 steps 2–3, Streamlit app Analytics and Q&A tabs (`streamlit run app.py`): done.
- Phase 4 step 4 (guardrails) and deployment prep (option C, numpy search, private HF data repo, attribution): done. Code is public at github.com/chrisacannon/llm-review-pricing-analysis.
- Next: **Chris deploys on share.streamlit.io** (main file `app.py`, Python 3.10, secrets per `.streamlit/secrets.toml.example`; the README's Deploying section has the steps), then Claude checks the live app as a visitor. Then Phase 5: rewrite the README around the findings, align its title with the app's ("Headphone Pricing Intelligence"), and use the repo name in the venv setup lines.

## Environment (Windows, PowerShell)

- Project folder is inside OneDrive; the virtual environment is deliberately **outside** it: `$HOME\venvs\review-pricing-intel`. Activate with `& $HOME\venvs\review-pricing-intel\Scripts\Activate.ps1`.
- API key: `$env:ANTHROPIC_API_KEY`, set per session (may be loaded from a file outside OneDrive). Never write the key into the project folder or any committed file.
- Embedding model cache: `~\.cache\fastembed` (outside OneDrive).
- Chris is comfortable with Excel and Power BI, newer to Python and PowerShell. Give PowerShell commands one per code block; pasting several lines at once has caused problems.

## Layout

| Path | Purpose |
| --- | --- |
| `pipeline/01_load_data.py` | Stream, filter, sample products by price quintile, reservoir-sample reviews, drop duplicates; `--append` adds products without renumbering |
| `pipeline/02_tag_reviews.py` | Claude Haiku tags: value sentiment, price mention, up to 5 themes with polarity; pilot / hand-check / batch / retag-check |
| `pipeline/03_build_index.py` | Local embeddings (BAAI/bge-small-en-v1.5 via fastembed) to `data/processed/review_embeddings.npz`; `qa.py` searches it with numpy (replaced Chroma 2026-10-02) |
| `pipeline/04_check_prices.py` | Claude Haiku price plausibility check: normal price range, is-headphone; suspect rule in code; pilot / run / summary |
| `price_overrides.csv` | Chris's hand-check decisions on the price check (committed; overrides verdict, is_headphone or exclude) |
| `pipeline/check_products.py` | Quick look at sampled product titles by tier |
| `qa.py` | Retrieval, cited answers (Claude Sonnet), price bands, `--bands`, `--eval` |
| `data/processed/` | parquet outputs, `tags.jsonl` (source of truth for tags), summaries, hand-check CSVs, `qa_eval.md` |
| `PROJECT_LOG.md` | Decisions, issues, validation results, costs. **Update it after each phase.** |

## Conventions and rules

- **Review IDs (`r000123`) are permanent.** Tags, citations and the index all key on them. Never renumber; `--append` continues numbering past every ID ever used.
- **Tagging:** `tags.jsonl` is the source of truth. `batch-submit` without `--all` tags only untagged reviews. Use `--all` only after a prompt change, and bump `PROMPT_VERSION`.
- **Value sentiment** is value for money only, not overall satisfaction. If `mentions_price` is false, value is forced to `not_mentioned` in code. This consistency rule matters: without it the "overpriced" share was overstated (see the log).
- **Price bands** (in `qa.py` `PRICE_BANDS`): budget <$25, value $25-49, mid $50-99, premium $100-199, flagship $200+. Chris chose to keep these. Quintiles are kept for equal-count statistics.
- **Band statistics exclude `price_checks.parquet` `exclude == True`** (suspect listings and non-headphones). The suspect rule (>2× above Claude's estimate, overpricing only) lives in code and is recomputed by `summary` without re-running Claude. Hand-check decisions go in `price_overrides.csv`; Chris makes those calls.
- **Sampling:** premium and flagship were deliberately oversampled by the supplemental pulls. Report statistics **per band**, never pooled across all reviews as if representative.
- **Q&A answers** must cite review IDs; citations are checked against the retrieved set. Retrieved reviews are the closest matches, not a random sample; answers must not present counts as frequencies. Retrieval leaves out excluded products and takes at most 3 reviews per product.
- **Costs:** report actual API spend for any step that calls Claude, and add it to the running totals in `PROJECT_LOG.md`. Haiku via Batch API for bulk work; Sonnet for answers.
- `data/` and any `*api-key*` file are git-ignored. Don't commit data or secrets.

## Known data issues

- Listed price is a single snapshot and sometimes wrong: third-party reseller markups (a ~$20 Sony MDR-EX155AP listed at $803), and reviewers citing prices far from the listing ("under $100" for a $449.99 listing). Handled by the price check: clear markups are excluded; listings below 2× and low listed prices stay in.
- Reviews comparing several products can be tagged on the other products' verdicts (rare).
- 19 non-headphone items (stands, cases, cables, beanies, a car radio) are in the sample; the price check flags them and they are excluded from band statistics.

## Phase 4 spec: Streamlit app

Goal from the plan: a deployed Streamlit app with a price-band vs value-sentiment chart, an overpriced-product table, and a Q&A panel.

1. **Price plausibility check** (do first): Claude reviews each product's title, brand and listed price and flags implausible listings (e.g. budget earbud at $803). Flagged products are excluded from band statistics but surfaced in the app as "suspect listings". Pennies of Haiku. Log the flag count.
2. **Analytics tab** (per band, flagged listings excluded):
   - Value verdict mix by band (positive / negative / neutral among reviews that mention price), plus the share mentioning price at all.
   - Theme prevalence and net polarity by band: what buyers praise and complain about as price rises.
   - Overpriced-for-band table: products priced above their band median with a weak value verdict, with review counts and a link to their cited reviews.
   - Show sample sizes (products, reviews) beside every band statistic.
3. **Q&A tab:** question box, band / theme / value / stars filters, the answer with clickable citations that expand to the review text, price and stars. Show cost per answer.
4. **Guardrails for a public demo:** per-session question cap; a few canned example questions with cached answers; optional field for the visitor's own API key; the app's key from Streamlit secrets, never committed.
5. **Deployment (decided 2026-10-02, option C):** public code repo; the app's data (~27 MB: parquet files + review embeddings + example answers) in a **private** Hugging Face dataset repo, downloaded at startup with a token from Streamlit secrets. Review text is never committed to GitHub. Why: the dataset states no license for the review text (code repo is MIT; the lab asks for a citation), and a public commit is hard to undo.

## Working style

- Before spending money on a full run, pilot on a small sample and report cost and results.
- Validate LLM output against a human check before scaling (see the Phase 2 hand-check and v1-v3 prompt history).
- Ask Chris before changes to bands, sampling, or anything that changes the project's headline numbers. He makes the pricing judgment calls.
