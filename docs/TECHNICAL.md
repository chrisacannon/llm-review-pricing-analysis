# Technical notes

How to run, rebuild and deploy the project. For what it found and how it was run, see the [README](../README.md); for every decision and its reason, the [project log](../PROJECT_LOG.md).

## Pipeline

| Step | What happens | Tooling |
| --- | --- | --- |
| 1. Sample | Stream the 28 GB Electronics file, keep the Headphones & Earbuds category path, sample products across price levels, reservoir-sample reviews, remove duplicates | pandas, Hugging Face |
| 2. Tag | Each review gets a value-for-money verdict, a price-mention flag and up to 5 topics with sentiment; a value verdict without price talk is overridden in code | Claude Haiku 4.5, Batch API |
| 3. Check prices | Estimate each product's normal price; flag listings more than 2x above it (rule applied in code) and non-headphones; hand-checked decisions in `price_overrides.csv` | Claude Haiku 4.5 |
| 4. Index | Embed every review locally; search is a matrix product over the vectors, with filters on the parquet files | BAAI/bge-small-en-v1.5 via fastembed |
| 5. Answer | Retrieve the 15 closest reviews (at most 3 per product, excluded listings left out), answer with citations, verify each citation against what was retrieved | Claude Sonnet 5.5 |
| 6. App | Band analytics with sample sizes and 95% ranges (product-level bootstrap, since reviews cluster within products), topic heatmap, overpriced-for-band table with review drill-down, Q&A with guardrails | Streamlit, Altair |

## Setup

Create the virtual environment outside any synced folder (OneDrive, Dropbox, iCloud), then install from the project folder. Windows (PowerShell):

```powershell
python -m venv $HOME\venvs\llm-review-pricing-analysis
& $HOME\venvs\llm-review-pricing-analysis\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

On Mac/Linux, activate with `source ~/venvs/llm-review-pricing-analysis/bin/activate` and set the key with `export ANTHROPIC_API_KEY=...`.

## Running each step

| Step | Command | Notes |
| --- | --- | --- |
| Sample | `python pipeline/01_load_data.py --category Electronics --category-match "Headphones & Earbuds" --max-price 1000 --stream` | ~13 min; `--append` deepens a band without renumbering review IDs |
| Tag | `python pipeline/02_tag_reviews.py pilot --n 500`, then `export-check`, `batch-submit`, `batch-collect --wait` | Resumable; Batch API at half price |
| Check prices | `python pipeline/04_check_prices.py pilot`, then `run`; `summary` after editing `price_overrides.csv` | ~$0.40 |
| Index | `python pipeline/03_build_index.py` | Free; ~1 hour on a laptop CPU |
| Example answers | `python qa.py --cache-examples` | ~$0.08 |
| App | `streamlit run app.py` | `python analytics.py` prints the same numbers |
| Q&A from the command line | `python qa.py "What do flagship buyers complain about?" --band flagship` | `--eval` runs 10 test questions |

Outputs go to `data/processed/` (git-ignored). Each script's docstring has the details and flags.

## Deploying

The app runs on Streamlit Community Cloud. Its working data (~26 MB: the columns the app reads, review embeddings, cached example answers; no reviewer IDs) is loaded from a separate access-controlled Hugging Face dataset repo rather than this repository, and checked against a manifest of checksums on download.

```powershell
python pipeline\05_app_data.py build                                   # slim copy in app_data/ (git-ignored)
python pipeline\05_app_data.py upload --repo your-username/your-repo   # needs $env:HF_TOKEN with write access
```

Streamlit secrets (see `.streamlit/secrets.toml.example`): `ANTHROPIC_API_KEY`, `HF_DATA_REPO`, and a read-only `HF_TOKEN`. To rehearse locally, put the same values in `.streamlit/secrets.toml` (git-ignored) and point `$env:REVIEW_DATA_DIR` at an empty folder so the app has to download.

Guardrails for the public demo: 5 questions per session on the app's key plus a daily cap, cached example answers, an optional visitor key, a 500-character question limit, and the app's key in its own Anthropic workspace with a monthly spend limit.

## Citation

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```
