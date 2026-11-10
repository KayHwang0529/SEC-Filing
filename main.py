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
            rows += [sec_client.filing_to_record(f) for f in fs]
            print(f"{t}: {len(fs)} filings")
        except Exception as e:
            print(f"{t}: FAILED ({e})")
    pd.DataFrame(rows).to_csv(config.FILINGS_CSV, index=False)
    print(f"-> {config.FILINGS_CSV}")


def _done_keys() -> set[tuple[str, str]]:
    if not config.EXTRACTIONS_JSONL.exists():
        return set()
    with open(config.EXTRACTIONS_JSONL) as f:
        return {(r["accession"], r["section"]) for r in map(json.loads, f) if r}


def _documents(tickers):
    """Yield (meta, loader) for every fetched 10-K / 10-Q; loader() -> {section: text}."""
    if not config.FILINGS_CSV.exists():
        raise SystemExit("No data/filings.csv -- run `python main.py fetch` first.")
    df = pd.read_csv(config.FILINGS_CSV, dtype=str).fillna("")
    if tickers:
        df = df[df["ticker"].isin([t.upper() for t in tickers])]
    for rec in df.to_dict("records"):
        meta = {k: rec[k] for k in ("ticker", "form", "accession", "filing_date", "accepted")}
        yield meta, lambda rec=rec: sec_client.extract_sections(
            rec["form"], sec_client.html_to_text(sec_client.download(
                sec_client.Filing(**{k: rec[k] for k in sec_client.Filing.__dataclass_fields__}
                                  | {"cik": int(rec["cik"])}))))


def cmd_extract(args):
    if config.MOCK_MODE:
        print("MOCK MODE: APIKEY is the placeholder -> using dictionary heuristics, not Kimi.")
    done = _done_keys()
    done_docs = {acc for acc, _ in done}
    n = 0
    with open(config.EXTRACTIONS_JSONL, "a") as out:
        for meta, load_sections in _documents(args.tickers):
            if args.limit and n >= args.limit:
                break
            acc = meta["accession"]
            if (all((acc, s) in done for s in args.sections) if args.sections else acc in done_docs):
                continue
            try:
                sections = load_sections()
            except Exception as e:
                print(f"  skip {meta['ticker']} {meta['form']} {meta['filing_date']}: {e}")
                continue
            for sec, text in sections.items():
                if args.sections and sec not in args.sections:
                    continue
                if (meta["accession"], sec) in done:
                    continue
                x = kimi_extractor.extract(text, meta["ticker"], meta["form"], sec, meta["filing_date"])
                out.write(json.dumps(meta | {"section": sec, "n_chars": len(text), "extraction": x}) + "\n")
                out.flush()
            n += 1
            print(f"  [{n}] {meta['ticker']} {meta['form']} {meta['filing_date']}: {list(sections)}")


def cmd_build(args):
    feats = features.build_features(features.load_extractions())
    start = (feats["filing_date"].min() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    px = prices.download_prices(feats["ticker"].unique().tolist(), start=start)
    ds = prices.event_returns(feats, px)
    feats.to_csv(config.FEATURES_CSV, index=False)
    ds.to_csv(config.DATASET_CSV, index=False)
    print(f"{len(ds)} events, {ds['event_day'].notna().sum()} with prices -> {config.DATASET_CSV}")


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
        cols = ["signal", "target", "train_rho", "test_rho", "test_perm_p", "test_q_bh"]
        print("\nTop hold-out results (sorted by BH q-value):")
        print(uni.sort_values("test_q_bh")[cols].head(15).to_string(index=False))
    if len(res["oos"]) and "r2_oos" in res["oos"]:
        print("\nOut-of-sample R^2 by model:")
        print(res["oos"].pivot(index="target", columns="model", values="r2_oos").round(4).to_string())


def main():
    p = argparse.ArgumentParser(description="SEC 10-K / 10-Q analyzer")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_tickers(sp):
        sp.add_argument("--tickers", nargs="*", default=None)
        sp.add_argument("--tickers-file", default=None, help="one ticker per line")

    f = sub.add_parser("fetch"); add_tickers(f)
    f.add_argument("--forms", nargs="*", default=["10-K", "10-Q"])
    f.add_argument("--start", default="2015-01-01")

    e = sub.add_parser("extract"); add_tickers(e)
    e.add_argument("--limit", type=int, default=0, help="max documents this run (cost control)")
    e.add_argument("--sections", nargs="*", default=None, help="risk_factors and/or mdna")

    sub.add_parser("build")
    a = sub.add_parser("analyze")
    a.add_argument("--form", choices=["10-K", "10-Q"], default=None, help="restrict to one form")
    a.add_argument("--n-perm", type=int, default=1000)

    al = sub.add_parser("all"); add_tickers(al)
    al.add_argument("--forms", nargs="*", default=["10-K", "10-Q"])
    al.add_argument("--start", default="2015-01-01")
    al.add_argument("--limit", type=int, default=0)
    al.add_argument("--sections", nargs="*", default=None)
    al.add_argument("--form", default=None)
    al.add_argument("--n-perm", type=int, default=1000)

    args = p.parse_args()
    if getattr(args, "tickers_file", None):
        args.tickers = [l.strip() for l in open(args.tickers_file) if l.strip()]

    if args.cmd in ("fetch", "all"):
        if not args.tickers:
            p.error("--tickers or --tickers-file required")
        cmd_fetch(args)
    if args.cmd in ("extract", "all"):
        cmd_extract(args)
    if args.cmd in ("build", "all"):
        cmd_build(args)
    if args.cmd in ("analyze", "all"):
        cmd_analyze(args)


if __name__ == "__main__":
    main()
