from __future__ import annotations

import json
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

import config

TONE_SECTIONS = ["mdna", "full_text"]  # full_text = fallback when section parsing failed
GUIDANCE_CODE = {"raised": 1.0, "initiated": 0.5, "maintained": 0.0, "none": 0.0,
                 "lowered": -1.0, "withdrawn": -1.0}


def load_extractions(path=config.EXTRACTIONS_JSONL) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _doc_rows(records: list[dict]) -> pd.DataFrame:
    """Collapse section-level records -> one row per document."""
    docs: dict[str, dict] = {}
    for r in records:
        key = r["accession"]
        d = docs.setdefault(key, {k: r[k] for k in ("ticker", "form", "accession", "filing_date", "accepted")}
                            | {"_risks": [], "_tone": [], "_guid": []})
        x, sec = r["extraction"], r["section"]
        # Per-section columns (lets you test e.g. MD&A tone vs risk-factor tone separately)
        d[f"{sec}_sentiment"] = x.get("sentiment")
        d[f"{sec}_hedging"] = x.get("hedging")
        d[f"{sec}_lm_net"] = x["dictionary"]["lm_net"]
        if sec in TONE_SECTIONS:
            d["_tone"].append((x["dictionary"]["n_words"], x.get("sentiment", 0), x.get("hedging", 0),
                               x["dictionary"]["lm_net"], x["dictionary"]["lm_uncertainty"]))
        if sec != "mdna":  # risks come from Item 1A (or full_text fallback)
            d["_risks"] += x.get("risk_factors", [])
        d["_guid"].append(x.get("guidance", {}))

    rows = []
    for d in docs.values():
        tone = d.pop("_tone")
        if tone:  # word-count-weighted average over tone sections (normally just MD&A)
            t = np.array(tone, dtype=float)
            w = t[:, 0] / t[:, 0].sum()
            d["sentiment"], d["hedging"], d["lm_net"], d["lm_uncertainty"] = (w @ np.nan_to_num(t[:, 1:])).tolist()
        else:  # no MD&A parsed -> missing, NOT neutral (0 would fake a sentiment shift)
            d["sentiment"] = d["hedging"] = d["lm_net"] = d["lm_uncertainty"] = np.nan
        guid = d.pop("_guid")
        dirs = [g.get("direction", "none") for g in guid]
        codes = [GUIDANCE_CODE.get(x, 0.0) for x in dirs] or [0.0]
        # Any cut/withdrawal dominates (most material); otherwise take the strongest positive.
        d["guidance_score"] = min(codes) if min(codes) < 0 else max(codes)
        d["guidance_provided"] = float(any(g.get("provided") for g in guid))
        risks = d.pop("_risks")
        d["n_risks"] = len(risks)
        d["mean_risk_severity"] = np.mean([r.get("severity", 3) for r in risks]) if risks else np.nan
        d["_risk_titles"] = [r.get("title", "") for r in risks]
        d["_risk_sev"] = [r.get("severity", 3) for r in risks]
        rows.append(d)
    df = pd.DataFrame(rows)
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)


def _risk_diff(cur: list[str], prev: list[str], sev: list[int], thresh: float = 0.6):
    """Count risk factors with no close match in the prior document (fuzzy title match)."""
    if not prev:
        return np.nan, np.nan, np.nan
    def best(t, pool):
        return max((SequenceMatcher(None, t.lower(), p.lower()).ratio() for p in pool), default=0)
    new = [best(t, prev) < thresh for t in cur]
    removed = sum(best(p, cur) < thresh for p in prev)
    new_sev = sum(s for s, is_new in zip(sev, new) if is_new)
    return float(sum(new)), float(removed), float(new_sev)


def build_features(records: list[dict]) -> pd.DataFrame:
    df = _doc_rows(records)

    # Quarter-over-quarter changes: each filing vs the ticker's previous 10-K/10-Q
    g = df.groupby("ticker", group_keys=False)
    for col in ["sentiment", "hedging", "lm_net", "lm_uncertainty", "guidance_score", "n_risks"]:
        df[f"d_{col}"] = g[col].diff()

    # Firm-relative sentiment: z-score against the firm's OWN PAST only (expanding, shifted).
    past = g["sentiment"]
    mu = past.transform(lambda s: s.expanding(min_periods=4).mean().shift(1))
    sd = past.transform(lambda s: s.expanding(min_periods=4).std().shift(1))
    df["sentiment_z_firm"] = (df["sentiment"] - mu) / sd.replace(0, np.nan)

    # Risk factor churn: compare to the previous document of the SAME form
    new_c, rem_c, new_s = [], [], []
    prev_titles: dict[tuple, list[str]] = {}
    for _, row in df.iterrows():
        k = (row["ticker"], row["form"])
        a, b, c = _risk_diff(row["_risk_titles"], prev_titles.get(k, []), row["_risk_sev"])
        new_c.append(a); rem_c.append(b); new_s.append(c)
        if row["_risk_titles"]:
            prev_titles[k] = row["_risk_titles"]
    df["new_risks"], df["removed_risks"], df["new_risk_severity"] = new_c, rem_c, new_s

    return df.drop(columns=["_risk_titles", "_risk_sev"])


SIGNAL_COLUMNS = ["sentiment", "d_sentiment", "sentiment_z_firm", "hedging", "d_hedging",
                  "guidance_score", "d_guidance_score", "new_risks", "new_risk_severity",
                  "d_n_risks"]
BASELINE_COLUMNS = ["lm_net", "d_lm_net", "lm_uncertainty", "d_lm_uncertainty"]
