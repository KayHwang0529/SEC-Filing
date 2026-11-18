"""Pipeline CLI.

    python main.py fetch    --tickers-file tickers.txt --start 2015-01-01
    python main.py extract  [--limit 20] [--sections risk_factors mdna]
    python main.py build
    python main.py analyze  [--form 10-K]
    python main.py all      --tickers-file tickers.txt
"""
from __future__ import annotations

import argparse
import json
import math
import random

import pandas as pd

import config
import features
import kimi_extractor
import model
import prices
import sec_client


def cmd_fetch(args):
    rows = []
    for t in args.tickers:
        try:
            fs = sec_client.list_filings(t, forms=tuple(args.forms), start=args.start)
            rows += [
                sec_client.filing_to_record(f)
                | {"sector": args.sectors.get(t, config.TICKER_SECTORS.get(t, ""))}
                for f in fs
            ]
            print(f"{t}: {len(fs)} filings")
        except Exception as e:
            print(f"{t}: FAILED ({e})")
    pd.DataFrame(rows).to_csv(config.FILINGS_CSV, index=False)
    print(f"-> {config.FILINGS_CSV}")


def _extraction_identity() -> tuple[str, str, str, str, str]:
    mode = "mock" if config.MOCK_MODE else "kimi"
    model_name = "dictionary-mock" if config.MOCK_MODE else config.KIMI_MODEL
    return (
        mode,
        model_name,
        config.PROMPT_VERSION,
        kimi_extractor.PROMPT_HASH,
        kimi_extractor.dictionary_fingerprint(),
    )


def _done_keys() -> set[tuple[str, str, str, str, str, str]]:
    if not config.EXTRACTIONS_JSONL.exists():
        return set()
    mode, model_name, prompt_version, prompt_hash, dictionary_hash = _extraction_identity()
    with open(config.EXTRACTIONS_JSONL) as f:
        return {
            (r["accession"], r["section"], mode, model_name, prompt_hash, dictionary_hash)
            for r in map(json.loads, f)
            if r.get("mode") == mode
            and r.get("model") == model_name
            and r.get("prompt_version") == prompt_version
            and r.get("prompt_hash") == prompt_hash
            and r.get("dictionary_hash") == dictionary_hash
        }


def _documents(tickers):
    """Yield (meta, loader) for fetched 10-K / 10-Q filings."""
    if not config.FILINGS_CSV.exists():
        raise SystemExit("No data/filings.csv -- run `python main.py fetch` first.")
    df = pd.read_csv(config.FILINGS_CSV, dtype=str).fillna("")
    df = df[df["form"].isin(["10-K", "10-Q"])]
    if tickers:
        df = df[df["ticker"].isin([t.upper() for t in tickers])]
    for rec in df.to_dict("records"):
        meta = {k: rec.get(k, "") for k in (
            "ticker", "form", "accession", "filing_date", "accepted", "report_date"
        )}
        meta["sector"] = rec.get("sector") or config.TICKER_SECTORS.get(rec["ticker"], "")
        yield meta, lambda rec=rec: sec_client.extract_sections(
            rec["form"], sec_client.html_to_text(sec_client.download(
                sec_client.Filing(**{k: rec[k] for k in sec_client.Filing.__dataclass_fields__}
                                  | {"cik": int(rec["cik"])}))))


def _limit_across_tickers_and_forms(documents, limit: int):
    strata = {}
    for document in documents:
        meta = document[0]
        strata.setdefault((meta["ticker"], meta["form"]), []).append(document)
    selected = []
    while len(selected) < limit and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(selected) < limit:
                selected.append(strata[key].pop(0))
    return selected


def cmd_extract(args):
    if config.MOCK_MODE:
        print("MOCK MODE: APIKEY is the placeholder -> using dictionary heuristics, not Kimi.")
    print(f"Dictionary baseline: {kimi_extractor.lm_dictionary_source()}")
    done = _done_keys()
    mode, model_name, prompt_version, prompt_hash, dictionary_hash = _extraction_identity()
    docs = []
    for meta, load_sections in _documents(args.tickers):
        accession = meta["accession"]
        keys = {
            (accession, section, mode, model_name, prompt_hash, dictionary_hash)
            for section in args.sections or []
        }
        if args.sections:
            if all(key in done for key in keys):
                continue
        elif all(
            (accession, section, mode, model_name, prompt_hash, dictionary_hash) in done
            for section in sec_client.SECTIONS.get(meta["form"], {})
        ):
            continue
        docs.append((meta, load_sections))
    if args.limit:
        docs = _limit_across_tickers_and_forms(docs, args.limit)

    n = 0
    parse_counts = {}
    estimated_chars = estimated_prompt_chars = estimated_tokens = estimated_calls = 0
    output = open(config.EXTRACTIONS_JSONL, "a") if not args.dry_run else None
    try:
        for meta, load_sections in docs:
            try:
                sections = load_sections()
            except Exception as e:
                print(f"  skip {meta['ticker']} {meta['form']} {meta['filing_date']}: {e}")
                continue
            if args.sections:
                sections = {name: text for name, text in sections.items() if name in args.sections}
            counts = parse_counts.setdefault((meta["ticker"], meta["form"]), [0, 0])
            counts[0] += 1
            counts[1] += bool(sections)
            if not sections:
                print(f"  parse failed {meta['ticker']} {meta['form']} {meta['filing_date']}: no requested section")
                n += 1
                continue
            if args.dry_run:
                for section_name, text in sections.items():
                    estimated_chars += len(text)
                    chunks = kimi_extractor._chunks(text, config.KIMI_MAX_CHARS_PER_CALL)
                    for chunk in chunks:
                        prompt = kimi_extractor.EXTRACTION_PROMPT.format(
                            text=chunk,
                            ticker=meta["ticker"],
                            form=meta["form"],
                            section=section_name,
                            date=meta["filing_date"],
                        )
                        estimated_prompt_chars += len(kimi_extractor.SYSTEM_PROMPT) + len(prompt)
                        estimated_calls += 1
                n += 1
                print(f"  [{n}] {meta['ticker']} {meta['form']} {meta['filing_date']}: {list(sections)}")
                continue
            for sec, text in sections.items():
                done_key = (
                    meta["accession"], sec, mode, model_name, prompt_hash, dictionary_hash
                )
                if done_key in done:
                    continue
                x = kimi_extractor.extract(text, meta["ticker"], meta["form"], sec, meta["filing_date"])
                record = meta | {
                    "section": sec,
                    "n_chars": len(text),
                    "mode": mode,
                    "model": model_name,
                    "prompt_version": prompt_version,
                    "prompt_hash": prompt_hash,
                    "dictionary_hash": dictionary_hash,
                    "dictionary_source": kimi_extractor.lm_dictionary_source(),
                    "extraction": x,
                }
                output.write(json.dumps(record) + "\n")
                output.flush()
                done.add(done_key)
            n += 1
            print(f"  [{n}] {meta['ticker']} {meta['form']} {meta['filing_date']}: {list(sections)}")
    finally:
        if output:
            output.close()

    if args.dry_run:
        estimated_tokens = math.ceil(estimated_prompt_chars / 4)
        print(
            f"DRY RUN: {estimated_chars:,} section characters, "
            f"{estimated_prompt_chars:,} prompt/system characters, approximately "
            f"{estimated_tokens:,} input tokens, {estimated_calls:,} Kimi calls. "
            "Token count is a rough chars/4 estimate."
        )
    if parse_counts:
        print("\nSection parse success by ticker and form:")
        print(f"{'Ticker':<8} {'Form':<6} {'Parsed':>8} {'Tried':>8} {'Rate':>8}")
        for (ticker, form), (tried, parsed) in sorted(parse_counts.items()):
            print(f"{ticker:<8} {form:<6} {parsed:>8} {tried:>8} {parsed / tried:>7.1%}")


def cmd_build(args):
    filings = pd.read_csv(config.FILINGS_CSV, dtype=str).fillna("")
    if "items" not in filings:
        raise SystemExit("Filing metadata has no `items` column; rerun `python main.py fetch`.")
    earnings_filings = filings[
        (filings["form"] == "8-K")
        & filings["items"].str.contains(r"(?<!\d)2\.02(?!\d)", regex=True)
    ]
    if earnings_filings.empty:
        raise SystemExit(
            "Earnings-return control requires 8-K item 2.02 metadata; "
            "rerun `python main.py fetch` with the default forms."
        )
    feats = features.build_features(features.load_extractions())
    start = (feats["filing_date"].min() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    latest_filing = (feats["filing_date"].max() + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    px = prices.download_prices(
        feats["ticker"].unique().tolist(), start=start, required_through=latest_filing
    )
    ds = prices.event_returns(feats, px)
    ds = prices.attach_earnings_control(ds, earnings_filings, px)
    feats.to_csv(config.FEATURES_CSV, index=False)
    ds.to_csv(config.DATASET_CSV, index=False)
    coverage = ds["earnings_car"].notna().sum()
    print(
        f"{len(ds)} events, {ds['event_day'].notna().sum()} with prices, "
        f"{coverage} with earnings controls -> {config.DATASET_CSV}"
    )


def cmd_validation_sample(args):
    rows = []
    for record in features.load_extractions():
        extraction = record.get("extraction", {})
        for quote in extraction.get("notable_quotes", []):
            rows.append({
                "ticker": record.get("ticker", ""),
                "form": record.get("form", ""),
                "accession": record.get("accession", ""),
                "filing_date": record.get("filing_date", ""),
                "section": record.get("section", ""),
                "quote": quote,
                "model_sentiment": extraction.get("sentiment"),
                "model_hedging": extraction.get("hedging"),
                "human_tone_label": "",
                "human_sentiment_label": "",
                "quote_supported_by_passage": "",
                "reviewer_notes": "",
            })
    if not rows:
        raise SystemExit("No notable_quotes available; run Kimi extraction before sampling.")
    groups = {}
    for row in rows:
        groups.setdefault((row["ticker"], row["form"]), []).append(row)
    rng = random.Random(0)
    for group in groups.values():
        rng.shuffle(group)
    sample = []
    while len(sample) < args.size and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(sample) < args.size:
                sample.append(groups[key].pop())
    output = config.RESULTS_DIR / "validation_sample.csv"
    pd.DataFrame(sample).to_csv(output, index=False)
    print(f"{len(sample)} quotes for human labeling -> {output}")


def cmd_analyze(args):
    ds = pd.read_csv(config.DATASET_CSV, parse_dates=["event_day", "filing_date"])
    ds = ds.dropna(subset=["event_day"])
    if args.form:
        ds = ds[ds["form"] == args.form]
    print(f"{ds['ticker'].nunique()} tickers, {len(ds)} filings")
    res = model.run_study(ds, n_perm=args.n_perm)
    model.save_results(res)
    uni = res["univariate"]
    if len(uni):
        cols = ["form", "signal", "target", "train_rho", "test_rho", "test_perm_p", "is_primary"]
        print("\nHold-out results (only the pre-specified primary is confirmatory):")
        print(uni.sort_values(["is_primary", "test_perm_p"], ascending=[False, True])[cols]
              .head(15).to_string(index=False))
    if len(res["primary"]):
        print("\nPre-specified primary hypothesis (10-K, car_1_20, firm-demeaned sentiment):")
        print(res["primary"].to_string(index=False))
    if len(res["oos"]) and "r2_oos" in res["oos"]:
        print("\nOut-of-sample R^2 by model:")
        print(
            res["oos"].pivot(index=["form", "target"], columns="model", values="r2_oos")
            .round(4).to_string()
        )


def main():
    p = argparse.ArgumentParser(description="SEC 10-K / 10-Q analyzer")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_tickers(sp, allow_sp500=False):
        sp.add_argument("--tickers", nargs="*", default=None)
        sp.add_argument("--tickers-file", default=None, help="one ticker per line")
        if allow_sp500:
            sp.add_argument("--sp500", action="store_true", help="use current S&P 500 constituents")

    f = sub.add_parser("fetch"); add_tickers(f, allow_sp500=True)
    f.add_argument("--forms", nargs="*", default=["10-K", "10-Q", "8-K"])
    f.add_argument("--start", default="2015-01-01")

    e = sub.add_parser("extract"); add_tickers(e)
    e.add_argument("--limit", type=int, default=0, help="max documents this run (cost control)")
    e.add_argument("--sections", nargs="*", default=None, help="risk_factors and/or mdna")
    e.add_argument("--dry-run", action="store_true", help="estimate tokens/calls without extracting")

    sub.add_parser("build")
    v = sub.add_parser("validation-sample")
    v.add_argument("--size", type=int, default=50, help="number of quotes (50-100 recommended)")
    a = sub.add_parser("analyze")
    a.add_argument("--form", choices=["10-K", "10-Q"], default=None, help="restrict to one form")
    a.add_argument("--n-perm", type=int, default=1000)

    al = sub.add_parser("all"); add_tickers(al, allow_sp500=True)
    al.add_argument("--forms", nargs="*", default=["10-K", "10-Q", "8-K"])
    al.add_argument("--start", default="2015-01-01")
    al.add_argument("--limit", type=int, default=0)
    al.add_argument("--sections", nargs="*", default=None)
    al.add_argument("--dry-run", action="store_true")
    al.add_argument("--form", default=None)
    al.add_argument("--n-perm", type=int, default=1000)

    args = p.parse_args()
    if getattr(args, "tickers_file", None):
        args.tickers = [l.strip() for l in open(args.tickers_file) if l.strip()]

    if args.cmd in ("fetch", "all"):
        if args.sp500:
            args.sectors = sec_client.sp500_constituents()
            args.tickers = list(args.sectors)
        else:
            args.sectors = config.TICKER_SECTORS
        if not args.tickers:
            p.error("--tickers, --tickers-file, or --sp500 required")
        cmd_fetch(args)
    if args.cmd in ("extract", "all"):
        cmd_extract(args)
    if args.cmd in ("build", "all"):
        cmd_build(args)
    if args.cmd == "validation-sample":
        if not 50 <= args.size <= 100:
            p.error("--size must be between 50 and 100")
        cmd_validation_sample(args)
    if args.cmd in ("analyze", "all"):
        cmd_analyze(args)


if __name__ == "__main__":
    main()
