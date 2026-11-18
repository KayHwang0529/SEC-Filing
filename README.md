# SEC Filing Analyzer

Pulls SEC filing metadata, uses **Kimi** to extract structured signals from parsed 10-K/10-Q sections, and tests
whether those signals predict subsequent sector-relative returns. Earnings-release returns are mandatory controls.

```
EDGAR ──► sec_client.py ──► kimi_extractor.py ──► features.py ──► prices.py ──► model.py
(10-K/Q,8-K) (sections)     (JSON per section)     (same-form)      (CARs)        (stats)
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export SEC_USER_AGENT="YourName you@email.com"   # required by SEC
export KIMI_API_KEY=sk-...                       # until set, runs in MOCK mode
```

## Run
```bash
python main.py fetch   --tickers-file tickers.txt --start 2015-01-01   # or --tickers AAPL MSFT ...
python main.py fetch   --sp500 --start 2015-01-01  # current S&P 500 constituents (survivorship-biased)
python main.py extract --dry-run --limit 10       # approximate tokens/calls before spending
python main.py extract --limit 10                 # stratified across tickers and forms
python main.py extract                            # resumable by section, mode, model, and prompt hash
python main.py build                       # features + prices -> data/dataset.csv
python main.py analyze                     # -> data/results/
python main.py analyze --form 10-K         # annual filings only
python main.py validation-sample --size 50 # export quotes for human labeling
```

## Data you need

| Data | Source | Cost | Status |
|---|---|---|---|
| 10-K / 10-Q sections | SEC EDGAR | free | automated; parse failures are excluded and reported |
| Earnings releases | SEC 8-K item 2.02 metadata | free | required control; fetched by default |
| Filing timestamps | EDGAR `acceptanceDateTime` | free | parsed as UTC, converted to US/Eastern for event-day mapping |
| Daily adjusted prices + sector ETFs | yfinance | free | SPY fallback; cached prices refresh when stale |
| Ticker universe | `tickers.txt` (55 large caps) or `fetch --sp500` | — | S&P 500 mode uses current constituents |
| Full Loughran-McDonald dictionary | [University of Notre Dame](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) | free for academic research | place `Loughran-McDonald_MasterDictionary_1993-2025.csv` in `data/` or set `LM_DICTIONARY_PATH` |

The dictionary is free for academic research; contact its maintainers for commercial licensing. If the full dictionary
file is absent, the program warns and uses its small built-in dictionary; do not treat that fallback as the full
Loughran-McDonald baseline. The `--dry-run` token estimate uses characters/4 and is an estimate, not an exact tokenizer
count.

Existing `data/filings.csv` files without 8-K item 2.02 rows cannot build the required earnings control. Re-run
`fetch` (default forms include 10-K, 10-Q, and 8-K) before building.

## Statistical design

- The only confirmatory hypothesis is **10-K, firm-demeaned sentiment, `car_1_20`**. Other signals, targets, and
  model comparisons are exploratory; exploratory p-values are not BH-adjusted.
- Chronological holdout starts at `TEST_START` (default `2025-09-05`) with a `max(HORIZONS)` business-day embargo.
  `Preprocessor` winsorization/scaling is fit on training data **separately by form**, and each expanding walk-forward
  yearly fold refits its own preprocessing.
- `walk_forward_ic.csv` reports per-year correlations from 2023 onward. `cutoff_comparison.csv` compares IC before and after the
  model release boundary. Moonshot's [Kimi-K2-Instruct-0905 model card](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905)
  does **not** document a training-data cutoff; September 5, 2025 is the model's public release boundary, not a
  verified knowledge cutoff. Neither split proves absence of contamination.
- `d_*` and expanding firm-relative features compare only filings of the same form. Preprocessing is also form-specific.
- Raw sentiment/hedging levels are demeaned against the firm's own past filings of that form. Guidance levels and
  LLM-generated risk-title churn are retained for inspection but excluded from the default signal set.
- The deterministic text-change baseline compares 512-dimensional signed token sketches with the nearest same-form
  report-date about one year earlier for MD&A and risk factors. Similarity is a compact hashed approximation.
- Abnormal returns use the issuer's sector ETF when sector metadata is known, otherwise SPY. `car_0_h` includes the
  announcement reaction; `car_1_h` starts after day 0.
- `earnings_car` measures the most recent preceding item 2.02 8-K's one-session sector-adjusted return (up to 45 days
  before the filing) and is included in every multivariate specification.
- Three SEC filing index pages were checked against `acceptanceDateTime`: `2024-11-01T10:01:36Z` displays
  `06:01:36 ET`, `2025-01-31T11:01:27Z` displays `06:01:27 ET`, and `2023-11-02T22:08:27Z` displays
  `18:08:27 ET` (filed the next calendar date). This confirms the `Z` values are UTC and that timezone conversion can
  change the selected trade date. A fourth, `2015-07-31T19:07:52Z`, displays `15:07:52 ET`; treating it as a 19:07 ET
  wall clock would incorrectly shift an otherwise same-day filing to the next session.
- Quintile-spread uncertainty is clustered by firm; hit rate is descriptive, not tested against an assumed 50% base
  rate. Newey-West IC inference is suppressed below 20 periods, and even with more periods should be treated cautiously.

The current 55-firm universe is underpowered for realistic ICs. Use `fetch --sp500` to broaden the sample, but this
uses current constituents and therefore introduces survivorship bias; historical constituents are preferable.
Cross-sectional ICs require at least five firms per quarter. The annual IC files will expose year/regime variation,
but do not remove limited power.

## Caveats
- Parsing rates vary by issuer and form; banks and issuers with non-standard layouts/exhibits may have low rates.
  Extraction prints a ticker/form parse-success table and never substitutes full filing text for a missing section.
- 10-Q risk-factor sections often just say "no material changes"; risk-text similarity should be interpreted by form.
- `validation-sample --size 50` exports notable quotes with blank human-label fields. Human labeling and agreement
  analysis remain a required validation step before making strong claims about extraction quality.
- Run `analyze --form 10-K` and `analyze --form 10-Q` separately when interpreting form-specific signals. The default
  analysis retains both forms but does not calculate cross-form differences or z-scores.
