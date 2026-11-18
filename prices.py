from __future__ import annotations

import re

import numpy as np
import pandas as pd

import config


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    required_through: str | None = None,
) -> pd.DataFrame:
    """Adjusted closes for issuers, SPY, and sector ETFs; refresh stale cached prices."""
    import yfinance as yf
    syms = sorted(
        set(t.upper() for t in tickers)
        | {config.BENCHMARK}
        | set(config.SECTOR_TO_ETF.values())
    )
    cached = None
    if config.PRICES_CSV.exists():
        cached = pd.read_csv(config.PRICES_CSV, index_col=0, parse_dates=True)
        cache_start_ok = cached.index.min().date() <= pd.Timestamp(start).date()
        if end:
            cache_end_ok = cached.index.max().date() >= (pd.Timestamp(end) - pd.Timedelta(days=1)).date()
        else:
            cache_end_ok = (
                pd.Timestamp.today().normalize() - cached.index.max().normalize()
            ).days <= 4
        if required_through:
            cache_end_ok = cache_end_ok and (
                cached.index.max().date() >= pd.Timestamp(required_through).date()
            )
        if set(syms) <= set(cached.columns) and cache_start_ok and cache_end_ok:
            return cached
    refresh_start = start
    if cached is not None and set(syms) <= set(cached.columns) and cache_start_ok:
        refresh_start = max(
            pd.Timestamp(start),
            cached.index.max().normalize() - pd.Timedelta(days=7),
        ).strftime("%Y-%m-%d")
    px = yf.download(
        syms, start=refresh_start, end=end, auto_adjust=True, progress=False
    )["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(syms[0])
    px = px.dropna(how="all")
    if cached is not None:
        px = pd.concat([cached, px]).sort_index()
        px = px[~px.index.duplicated(keep="last")]
    px.to_csv(config.PRICES_CSV)
    return px


def event_day(accepted: str, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Map UTC acceptance times to the first tradeable ET session.

    Checked against three SEC index pages; UTC conversion preserves the locally
    displayed Eastern acceptance time, including after-close submissions.
    """
    ts = pd.Timestamp(accepted)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    ts = ts.tz_convert("America/New_York")
    day = ts.tz_localize(None).normalize()
    after_close = ts.hour >= 16
    if after_close:
        day += pd.Timedelta(days=1)
    i = trading_days.searchsorted(day)
    return trading_days[i] if i < len(trading_days) else None


def event_returns(events: pd.DataFrame, px: pd.DataFrame, horizons=config.HORIZONS) -> pd.DataFrame:
    """Return issuer CARs relative to sector ETFs, falling back to SPY."""
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
            s = logp[ev["ticker"]]
            sector = ev.get("sector", "")
            sector_etf = config.SECTOR_TO_ETF.get(sector)
            benchmark_symbol = sector_etf if sector_etf in logp.columns else bench
            b = logp[benchmark_symbol]
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


def _utc_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def attach_earnings_control(
    events: pd.DataFrame,
    earnings_filings: pd.DataFrame,
    px: pd.DataFrame,
    max_age_days: int = 45,
) -> pd.DataFrame:
    """Attach the most recent item 2.02 8-K's one-session abnormal return to each event."""
    logp = np.log(px)
    days = px.index
    earnings = []
    for _, filing in earnings_filings.iterrows():
        if not re.search(r"(?<!\d)2\.02(?!\d)", str(filing.get("items", ""))):
            continue
        ticker = str(filing["ticker"])
        if ticker not in logp.columns:
            continue
        trade_day = event_day(str(filing["accepted"]), days)
        if trade_day is None:
            continue
        i = days.get_loc(trade_day)
        if i < 1:
            continue
        sector = filing.get("sector", config.TICKER_SECTORS.get(ticker, ""))
        etf = config.SECTOR_TO_ETF.get(sector)
        benchmark = etf if etf in logp.columns else config.BENCHMARK
        abnormal_return = (
            logp[ticker].iloc[i] - logp[ticker].iloc[i - 1]
            - logp[benchmark].iloc[i] + logp[benchmark].iloc[i - 1]
        )
        earnings.append({
            "ticker": ticker,
            "accepted_utc": _utc_timestamp(str(filing["accepted"])),
            "earnings_date": trade_day,
            "earnings_car": abnormal_return,
        })

    by_ticker: dict[str, list[dict]] = {}
    for record in earnings:
        by_ticker.setdefault(record["ticker"], []).append(record)
    for records in by_ticker.values():
        records.sort(key=lambda r: r["accepted_utc"])

    result = events.copy()
    result["earnings_date"] = pd.NaT
    result["earnings_car"] = np.nan
    for index, event in result.iterrows():
        accepted = _utc_timestamp(str(event["accepted"]))
        candidates = [
            record for record in by_ticker.get(str(event["ticker"]), [])
            if record["accepted_utc"] <= accepted
            and (accepted - record["accepted_utc"]).days <= max_age_days
        ]
        if candidates:
            match = candidates[-1]
            result.at[index, "earnings_date"] = match["earnings_date"]
            result.at[index, "earnings_car"] = match["earnings_car"]
    return result
