# SEC Filing Analyzer

Pulls 10-K and 10-Q filings for every ticker in `tickers.txt` from SEC EDGAR, uses **Kimi** to extract structured signals
(sentiment, hedging, guidance direction, risk factors), and tests whether those signals predict
subsequent abnormal returns, using a chronological train/test split with an embargo period.

```
EDGAR ──► sec_client.py ──► kimi_extractor.py ──► features.py ──► prices.py ──► model.py
(filings)   (sections)       (JSON per section)     (QoQ deltas)    (CARs)        (stats)
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
python main.py extract --limit 10          # start small, check data/extractions.jsonl by hand
python main.py extract                     # resumable; Kimi responses are cached on disk
python main.py build                       # features + prices -> data/dataset.csv
python main.py analyze                     # -> data/results/
python main.py analyze --form 10-K         # annual filings only
```

## Data you need

| Data | Source | Cost | Status |
|---|---|---|---|
| 10-K / 10-Q text | SEC EDGAR | free | automated |
| Exact filing timestamps | EDGAR `acceptanceDateTime` | free | automated |
| Daily adjusted prices + SPY | yfinance | free | automated |
| Ticker universe | `tickers.txt` (55 large caps, 11 sectors) | — | edit freely |
| Earnings surprise (control) | EDGAR XBRL + estimates | estimates are paid | optional, recommended |
| Fama-French factors | Ken French Data Library | free | optional upgrade to market-adjusted returns |
| Full Loughran-McDonald dictionary | sraf.nd.edu | free | optional, improves baseline |

**Universe:** use about 100+ firms from one index (e.g. S&P 500 constituents). Cross-sectional
IC needs at least 5 firms per quarter, and statistical power needs many events. Use historical
constituents if you can; today's list carries survivorship bias.

## Statistical design (model.py)
- Chronological split at `TEST_START`, with a `max(HORIZONS)` business-day embargo
- Winsorization and z-scoring fit on train only
- OLS with two-way clustered SEs (firm × quarter)
- Quarterly rank IC with a Newey-West t-stat
- Within-quarter permutation test
- Benjamini-Hochberg FDR across all signal × horizon tests
- Nested out-of-sample comparison: controls → +dictionary baseline → +LLM → both.
  The LLM signal is only useful if it beats the cheap baseline out of sample.
- `car_0_h` includes the announcement reaction; `car_1_h` starts after day 0, so it tests
  tradeable drift.

## Caveats
- **Kimi's training cutoff:** for filings before the model's cutoff, the model may know what
  happened next. Treat the post-cutoff test set as the clean one, or set `TEST_START` after
  the model's cutoff date.
- 10-Q risk-factor sections often just say "no material changes". For risk churn, 10-K vs prior
  10-K is the cleanest comparison.
- 10-Qs are usually filed after the earnings press release, so much of the news is already in the
  price. `car_1_h` (post-filing drift) is the fairer target for 10-Qs.
