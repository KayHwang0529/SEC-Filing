# Pipeline Change Summary

This note records the pipeline changes, where they were made, and the reason for each.
The user-facing operating instructions are in [README.md](./README.md); implementation
details and motivations are summarized here.

## Extraction and filing parsing

| File | Change | Why |
|---|---|---|
| [main.py](./main.py) | Extraction records now include mode, model, prompt version/hash, and dictionary fingerprint. Resume keys use those values per accession and section. | Prevents mock extractions or stale model/prompt/dictionary outputs from being mistaken for current results. Retrying after a partial document now processes only missing sections. |
| [kimi_extractor.py](./kimi_extractor.py) | The actual system and extraction prompt text is hashed. Failed JSON salvage remains inside the retry loop. Extraction metadata records dictionary identity; cached model responses recalculate dictionary scores and text sketches. | Prompt edits invalidate stale cache entries. Malformed responses no longer escape salvage handling and terminate extraction. A dictionary update refreshes baseline scores without requiring another model call. |
| [sec_client.py](./sec_client.py) | Section headers must occur at line starts; extraction returns no section on parse failure rather than substituting full filing text. Added current S&P 500 ticker/sector retrieval. | Inline references such as “see Item 1A” should not truncate the preceding section. Failed parses must not become expensive, incomparable tone observations. The optional larger universe supports a more powered study. |
| [main.py](./main.py) | Added parse-success counts by ticker/form, ticker/form-stratified `--limit`, and `extract --dry-run` token/call estimates. | Makes weak parser coverage visible, gives small hand-check runs broader coverage, and estimates extraction cost before API calls. |

## Features, baselines, and earnings controls

| File | Change | Why |
|---|---|---|
| [features.py](./features.py) | Deltas and firm-relative standardization compare only with prior filings of the same form. Raw sentiment, hedging, guidance, and dictionary/text-change baseline levels are demeaned against the firm's own past same-form filings. | A 10-K should not be compared with the structurally different 10-Q, and persistent firm-level tone should not drive within-quarter tests. |
| [features.py](./features.py) | Full-text fallback and LLM-generated risk-title churn are removed from default tone/risk features. Added prior-year same-form cosine similarity for MD&A and risk sections, using deterministic 512-dimensional signed token sketches. | Missing sections should remain missing. Text similarity provides a deterministic “lazy prices” style comparison that does not depend on LLM-generated phrasing. |
| [kimi_extractor.py](./kimi_extractor.py), [config.py](./config.py) | Added support for the Notre Dame Loughran–McDonald master CSV, with a warning and explicit source metadata when only the small built-in fallback is available. | The baseline comparison should use the full finance dictionary where available and must not silently imply that the miniature fallback is complete. |
| [main.py](./main.py) | Fetch defaults now include 8-K metadata; building fails explicitly if item 2.02 earnings-release filings are unavailable. | Earnings news around filing dates is a required return control rather than an optional suggestion. |
| [prices.py](./prices.py) | Joins each filing to its most recent preceding item 2.02 8-K return (maximum age 45 days), and uses sector ETF-relative returns when sector data exists, otherwise SPY. | Controls for the earnings announcement confound and avoids assuming every stock has market beta exactly one. |

## Study design and price timing

| File | Change | Why |
|---|---|---|
| [config.py](./config.py) | Default `TEST_START` is the 2025-09-05 model-release boundary, with an explicit configurable `MODEL_BOUNDARY_DATE`. | The pinned model card does not provide a verified training cutoff. The documented release boundary is a cautious comparison point, not a claim about the model's training data. |
| [model.py](./model.py) | Set one confirmatory hypothesis (10-K, firm-demeaned sentiment, `car_1_20`); other signal/target results are exploratory. | Avoids treating a broad, post-hoc grid of tests as one confirmatory claim. |
| [model.py](./model.py) | Analysis and preprocessing are form-specific; preprocessing is refit on each training fold. Added annual expanding walk-forward ICs from 2023 and pre/post-release-boundary IC tables. | Avoids 10-K/10-Q standardization leakage and exposes time/regime variation. |
| [model.py](./model.py) | Removed a 50% null significance test for directional hit rate; quintile-spread uncertainty is firm-clustered; Newey–West inference is omitted for fewer than 20 periods. | Makes reported inference less dependent on unjustified exchangeability/base-rate assumptions and avoids presenting very short time-series t-statistics as reliable. |
| [prices.py](./prices.py) | Converts SEC `acceptanceDateTime` from UTC to US/Eastern before mapping to a trading session. Verified against four SEC filing index pages. | The literal `Z` is UTC; treating it as an Eastern wall clock can shift event days, especially for after-close filings. |
| [prices.py](./prices.py) | Price cache checks ticker/ETF coverage, staleness, and the latest event date; it refreshes the recent tail when needed. | Newly fetched filings must not silently have missing returns because an old cache ended too early. |
| [config.py](./config.py) | Added issuer-sector mappings and sector ETF mappings for the built-in universe. | Ensures existing tickers can use sector-relative returns without extra manual setup. |

## Validation and documentation

| File | Change | Why |
|---|---|---|
| [tests/test_pipeline.py](./tests/test_pipeline.py) | Added regression tests for section boundaries/failures, resume identity and partial documents, JSON retry, LM loading/cache refresh, stratified limits, same-form features/scaling, timestamp conversion, sector/earnings returns, price refresh, and walk-forward output. | Covers the core failure cases and the intended behavior of the new paths. |
| [main.py](./main.py) | Added `validation-sample --size 50..100`, exporting notable quotes plus blank human-label fields to `data/results/validation_sample.csv`. | Provides a repeatable starting point for the still-needed manual extraction-quality review. |
| [README.md](./README.md) | Updated commands, required 8-K/dictionary inputs, study design, cutoff caveat, timestamp checks, and power/survivorship caveats. | Keeps setup and interpretation guidance consistent with the new implementation. |

## Validation performed

- `python3 -m unittest discover -s tests -v` — 15 tests passed.
- Pylance diagnostics for changed Python files — no errors found.
- CLI help smoke checks and `git diff --check` — passed.
- SEC index pages were inspected to validate UTC/Eastern timestamp handling.

## Still required for a full study

- Fetch 8-K filings with item 2.02 metadata; the existing filing dataset may not include them.
- Download the Loughran–McDonald master CSV and configure its path before treating the full dictionary as the baseline.
- Run the extraction/build/analyze pipeline on the intended universe. The S&P 500 option uses current constituents and
  therefore has survivorship bias unless historical constituent data is supplied.
- Hand-label the exported 50–100 quote sample and report agreement/quality before making strong extraction claims.
- The model release boundary is not a verified training cutoff; pre/post-boundary comparisons are diagnostic, not proof
  that any result is free of contamination.
