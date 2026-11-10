from __future__ import annotations

import numpy as np
import pandas as pd

import config


def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Adjusted closes, columns = tickers (+ benchmark). Cached to data/prices.csv."""
    import yfinance as yf
    syms = sorted(set(t.upper() for t in tickers) | {config.BENCHMARK})
    if config.PRICES_CSV.exists():
        cached = pd.read_csv(config.PRICES_CSV, index_col=0, parse_dates=True)
        if set(syms) <= set(cached.columns) and str(cached.index.min().date()) <= start:
            return cached
    px = yf.download(syms, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(syms[0])
    px = px.dropna(how="all")
    px.to_csv(config.PRICES_CSV)
    return px


def event_day(accepted: str, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Map an EDGAR acceptance timestamp to the first tradeable session.
    EDGAR's acceptanceDateTime carries a 'Z' but in practice the clock is US/Eastern --
    spot-check a few against the "Accepted" time on the filing index page."""
    ts = pd.Timestamp(str(accepted).replace("Z", "").replace(".000", ""))
    day = ts.normalize()
    after_close = ts.hour > 16 or (ts.hour == 16 and ts.minute > 0)
    if after_close:
        day += pd.Timedelta(days=1)
    i = trading_days.searchsorted(day)
    return trading_days[i] if i < len(trading_days) else None


def event_returns(events: pd.DataFrame, px: pd.DataFrame, horizons=config.HORIZONS) -> pd.DataFrame:
    """events needs columns: ticker, accepted. Returns events + car_* columns."""
    logp = np.log(px)
    days = px.index
    bench = config.BENCHMARK
    out = []
    for _, ev in events.iterrows():
        row = {}
        t0 = event_day(ev["accepted"], days) if ev["ticker"] in px.columns else None
        if t0 is not None:
            i0 = days.get_loc(t0)
            row["event_day"] = t0
            s, b = logp[ev["ticker"]], logp[bench]
            for h in horizons:
                if i0 - 1 < 0 or i0 + h >= len(days):
                    continue  # window runs past available price data
                row[f"car_0_{h}"] = (s.iloc[i0 + h] - s.iloc[i0 - 1]) - (b.iloc[i0 + h] - b.iloc[i0 - 1])
                row[f"car_1_{h}"] = (s.iloc[i0 + h] - s.iloc[i0]) - (b.iloc[i0 + h] - b.iloc[i0])
            # Pre-event controls (momentum, volatility) from data strictly before t0
            if i0 >= 61:
                r = s.diff().iloc[i0 - 60: i0]
                row["pre_ret_60"] = s.iloc[i0 - 1] - s.iloc[i0 - 61]
                row["pre_vol_60"] = r.std() * np.sqrt(252)
        out.append(row)
    return pd.concat([events.reset_index(drop=True), pd.DataFrame(out)], axis=1)
