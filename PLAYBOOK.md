# AI project playbook

A repeatable way to run data and AI projects (LLM tagging, retrieval and Q&A, analytics apps) from idea to published app. It was distilled while building this project; each step notes where it shows up here, with the details in [`PROJECT_LOG.md`](PROJECT_LOG.md).

The aim is work that reads like business analysis that uses AI well, not a technology demo.

## 1. Scope before code (in conversation)

- Pin down the question and the "so what": who would use the answer, and what decision it informs.
- Write 3-4 concrete questions the finished project must answer. Pair structured analytics with any AI or Q&A piece; together they beat a chatbot demo.
- Use real data over synthetic whenever the point is analyzing human language or behavior.
- Write the plan as a shared doc: data source and limitations, architecture, phases with "done when" criteria, cost estimate, scope cuts, open decisions, risks.
- Check current API prices before estimating costs, and set a spend limit as a backstop.

*In this project:* the question is framed as portfolio-level pricing strategy (where is price out of line with perceived value, by price band). Real Amazon reviews were chosen over synthetic ones, because synthetic reviews would be Claude analyzing Claude.

## 2. Pilot before scaling

- Never run the full dataset, or spend real money or hours, before a small pilot.
- Inspect pilot output by eye. Most data problems show up here.
- Prefer precise filters over keyword matching. Stream huge source files instead of downloading them.

*In this project:* the first data pull matched the keyword "headphone" and turned out to be nearly 60% cases, cables, ear pads and adapters, because Amazon's category name contains the word. Switching to the category path fixed it. Every Claude step (tagging, price check, example answers) ran on a sample first, with cost reported before the full run.

## 3. Validate LLM output before trusting it at scale

- Hand-check about 50 items against clear criteria; target 85%+ per field.
- Read the misses for patterns, revise the prompt, re-tag the same items and compare. Look hardest at rows that changed but were previously approved.
- Add consistency rules in code between fields; they catch errors humans approve.
- Put thresholds and decision rules in code, not only in the prompt.
- Version prompts and record the version with every output.
- For Q&A: require citations, check them in code against what was retrieved, and say that retrieved items are the closest matches, not a random sample.
- Use cheap models and batch APIs for bulk classification; stronger models for answers.

*In this project:* the hand-check found 96% on value verdicts and 86% on topics; prompts went v1 to v3. A consistency rule (no value verdict without price talk) showed the headline overpriced share had been overstated. In the price check, Claude flagged listings only 1.1x its own estimate, so the 2x rule moved into code.

## 4. Design segments the way the business would

- Equal-count bins (quintiles) often don't match how a market is merchandised; use named dollar bands for analysis and keep quantiles for equal-sample statistics.
- Give every compared segment enough depth (aim for 60+ entities and ~2,000 records); top up thin segments with targeted pulls.
- If you oversample segments, report statistics per segment, never pooled.
- Keep IDs permanent so tags, citations and indexes never break.

*In this project:* the top quintile ran from $80 to $999, mixing gaming headsets with Bang & Olufsen, so the analysis uses five dollar bands. Flagship started with 18 products and was deepened to 88 with append pulls that never renumber review IDs.

## 5. Check the statistics before claiming the headline

- Records cluster within entities (many reviews per product), so compute ranges by product, not by review.
- Check that a finding isn't driven by a few products.
- Recompute the headline numbers independently before publishing.
- Check any within-segment claim with the same method.

*In this project:* product-level ranges widened flagship from 38-44% to 35-47%, and the headline held. A within-band price check was run, wasn't significant, and stayed out of the README.

## 6. Keep a project log from day one

- Record running totals (data size, API spend, accuracy), each decision with its reason, each issue and its fix, validation results, actual costs and run times, and emerging findings.
- Separate files by job: README is the public front page, the plan doc holds intent and open decisions, the log holds what actually happened.

*In this project:* [`PROJECT_LOG.md`](PROJECT_LOG.md). It is also the interview story: the hand-check passed, but a consistency rule showed the headline metric was overstated.

## 7. Use the right tool for each stage

- **Chat:** scoping, the plan, judgment calls, interpreting results, reviewing outputs, the write-up.
- **Coding agent (Claude Code):** build-heavy stretches with many edit-run-fix cycles, testing against real data and APIs, git, GitHub and deployment.
- Hand off through files, not memory: README, the log, and a `CLAUDE.md` with purpose, status, environment, conventions, rules that protect the data, known issues, and the next phase's spec.

*In this project:* planning and Phases 1-3 ran from chat; the app, price check and deployment were built in Claude Code, which started from [`CLAUDE.md`](CLAUDE.md) and the log.

## 8. Practical setup

- Keep virtual environments and model caches outside synced folders such as OneDrive.
- Never store API keys in the project folder; set them per session. Add a `.gitignore` for keys, `.env` files, data and caches before anything goes to GitHub.
- Make scripts resumable (skip finished work, back up before overwriting), and have them print costs and progress.

## 9. Explain it back before publishing

- For every method choice, write a plain-language explanation: what it is, why it was chosen, and what the alternative was. If you can't explain it, you don't own it yet.
- Write it as the questions an interviewer or stakeholder would ask, with a short answer and a longer one.
- Get an outside review of the method, not just the code, and update the analysis and the explanations together.

*In this project:* an outside review found the band ranges treated every review as independent, though reviews cluster within products. Switching to product-level ranges widened them (flagship 38–44% became 35–47%) and the headline held. The study guide then put every method choice into interview Q&A, including this one.

## 10. Finish for the portfolio

- Lead the README with 3-4 real findings with numbers, then method, validation and limitations.
- State data limitations plainly.
- Publish the code and a live demo with cost guardrails: per-session caps, cached example answers, an optional bring-your-own key, and a spend-limited key.
- Check the data's terms before publishing it; when they're unclear, keep the code public and the data private.

*In this project:* the [live app](https://llm-review-pricing-analysis.streamlit.app/) caps visitors at 5 questions per session on a spend-limited key; the review data sits in a private repo because the dataset states no license for it.
