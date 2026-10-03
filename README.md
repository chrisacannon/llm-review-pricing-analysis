# Review-Informed Pricing Intelligence

For one Amazon product category: where is price out of line with perceived value, and what do customers say at each price tier?

Python pipeline + Streamlit app using Claude for review tagging and Q&A. Data: [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (McAuley Lab, UCSD).

## Setup

Create the virtual environment outside any synced folder (OneDrive, Dropbox, iCloud): syncing thousands of package files slows everything down and can corrupt the environment. Then run `pip install` from inside the project folder.

Windows (PowerShell):

```powershell
python -m venv $HOME\venvs\review-pricing-intel
& $HOME\venvs\review-pricing-intel\Scripts\Activate.ps1
pip install -r requirements.txt
```

Mac/Linux:

```bash
python -m venv ~/venvs/review-pricing-intel
source ~/venvs/review-pricing-intel/bin/activate
pip install -r requirements.txt
```

Open a new terminal later? `cd` into the project folder and run the activate line again.

## Phase 1: load and sample the data

```bash
# Small category: files download once (~1.2 GB) and are cached for reruns
python pipeline/01_load_data.py --category Musical_Instruments

# Sub-category inside a huge file: stream instead of saving it
python pipeline/01_load_data.py --category Electronics --keyword "headphone|earbud" --stream
```

Outputs in `data/processed/`:

| File | Contents |
| --- | --- |
| `products.parquet` | Sampled products: price, rating, store, price quintile, reviews sampled |
| `reviews.parquet` | Sampled reviews with a stable `review_id` used for citations later |
| `summary.json` | Counts dropped at each filter, price distribution, rating mix |

Useful flags: `--min-ratings` (default 30), `--max-products` (300), `--reviews-per-product` (60), `--min-price`, `--max-price`, `--seed`.

To add more products to an existing sample (e.g. to deepen one price band), use `--append`: products you already have are skipped, review IDs continue where they left off, and the previous files are backed up first. Then tag only the new reviews with `batch-submit` (without `--all`) and rebuild the index.

```powershell
python pipeline\01_load_data.py --category Electronics --category-match "Headphones & Earbuds" --min-price 200 --max-price 1000 --max-products 80 --append --stream
```

Duplicate reviews (same product and text, by the same user or 40+ characters long) are removed on every run.

Products are sampled evenly across price quintiles so every tier is represented. Reviews are reservoir-sampled per product, so they aren't biased toward the start of the file.

## Phase 2: tag reviews with Claude

Each review gets a value-for-money sentiment, a price-mention flag, and up to five themes (sound quality, comfort, battery, connectivity, appearance, etc.) with polarity. Uses Claude Haiku 4.5, 25 reviews per request.

```powershell
python pipeline\02_tag_reviews.py pilot --n 500        # standard API, ~2 min
python pipeline\02_tag_reviews.py export-check --n 50  # hand_check.csv for review in Excel
python pipeline\02_tag_reviews.py batch-submit         # everything else, Batch API (half price)
python pipeline\02_tag_reviews.py batch-collect --wait
```

Resumable: already-tagged reviews are skipped, and missed ones can be resubmitted. Outputs: `tags.jsonl` (raw log), `review_tags.parquet`, `review_themes.parquet`.

## Phase 3: search index and Q&A

Reviews are embedded locally with `BAAI/bge-small-en-v1.5` (via fastembed, no API cost) and saved as one vector per review (`review_embeddings.npz`, ~22 MB); search is a matrix product, with filters for price band, stars, value verdict and theme. Claude Sonnet answers questions from the retrieved reviews only, citing review IDs, and every citation is checked against what was actually retrieved.

```powershell
python pipeline\03_build_index.py     # a few minutes; first run downloads the model (~130 MB)
python qa.py --bands                   # products and reviews per price band
python qa.py "What do buyers of flagship headphones complain about?" --band flagship
python qa.py --eval                    # 10 test questions -> data/processed/qa_eval.md
```

Filters: `--band budget|value|mid|premium|flagship` (under $25, $25-49, $50-99, $100-199, $200+; thresholds in `qa.py`), `--tier 1-5` (equal-count price quintiles), `--min-price`, `--max-price`, `--value positive|negative|neutral|not_mentioned`, `--theme comfort_fit` (repeatable), `--min-rating`, `--max-rating`.

## Phase 4: price check and app

Listed prices are a single snapshot, and some are reseller markups (a ~$20 earbud at $803). Claude Haiku estimates each product's normal price; listings more than twice that, non-headphones and hand-checked cases are excluded from band statistics and shown separately.

```powershell
python pipeline\04_check_prices.py pilot     # known problem listings + a sample per band
python pipeline\04_check_prices.py run       # everything else (~$0.40)
python pipeline\04_check_prices.py summary   # rebuild after editing price_overrides.csv
```

The Streamlit app's Analytics tab shows the value verdict mix by band (with 95% intervals and sample sizes), what buyers praise and complain about as price rises, products overpriced for their band with their reviews, and the excluded listings.

```powershell
streamlit run app.py
python analytics.py                          # the same numbers, printed
```

## Limitations

- Price is a single snapshot from when the data was collected; there is no price history.
- Reviews run through September 2023.
- Price ranges (e.g. "12.99 - 19.99") are dropped as ambiguous.
- Some listings are third-party resellers with marked-up prices (e.g. a ~$20 Sony earbud listed at $803).
- Reviews comparing several products can get value tags based on the other products.

See PROJECT_LOG.md for decisions, data issues and validation results.
