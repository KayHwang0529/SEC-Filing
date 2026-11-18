from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config

TONE_SECTIONS = ["mdna"]
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
        d = docs.setdefault(
            key,
            {k: r.get(k, "") for k in (
                "ticker", "form", "accession", "filing_date", "accepted", "report_date", "sector"
            )} | {"_risks": [], "_tone": [], "_guid": [], "_section_sketch": {}},
        )
        x, sec = r["extraction"], r["section"]
        # Per-section columns (lets you test e.g. MD&A tone vs risk-factor tone separately)
        d[f"{sec}_sentiment"] = x.get("sentiment")
        d[f"{sec}_hedging"] = x.get("hedging")
        dictionary = x.get("dictionary", {})
        d[f"{sec}_lm_net"] = dictionary.get("lm_net")
        if sec in TONE_SECTIONS:
            d["_tone"].append((
                dictionary.get("n_words", 0), x.get("sentiment", np.nan), x.get("hedging", np.nan),
                dictionary.get("lm_net", np.nan), dictionary.get("lm_uncertainty", np.nan),
            ))
        if sec == "risk_factors":
            d["_risks"] += x.get("risk_factors", [])
        if sec in ("mdna", "risk_factors") and x.get("text_sketch"):
            d["_section_sketch"][sec] = x["text_sketch"]
        d["_guid"].append(x.get("guidance", {}))

    rows = []
    for d in docs.values():
        tone = d.pop("_tone")
        if tone:  # word-count-weighted average over tone sections (normally just MD&A)
            t = np.array(tone, dtype=float)
            w = t[:, 0] / t[:, 0].sum()
            d["sentiment"], d["hedging"], d["lm_net"], d["lm_uncertainty"] = (
                w @ np.nan_to_num(t[:, 1:])
            ).tolist()
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
        rows.append(d)
    df = pd.DataFrame(rows)
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    if "report_date" in df:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        df["report_date"] = df["report_date"].fillna(df["filing_date"])
    else:
        df["report_date"] = df["filing_date"]
    df = df.sort_values(["ticker", "report_date", "filing_date"]).reset_index(drop=True)
    for section in ("mdna", "risk_factors"):
        df[f"{section}_text_similarity"] = _prior_year_similarity(df, section)
    return df


def _cosine(left: list[int], right: list[int]) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else np.nan


def _prior_year_similarity(df: pd.DataFrame, section: str) -> list[float]:
    """Compare section sketches with the closest same-form filing about one year earlier."""
    out = [np.nan] * len(df)
    history: dict[tuple[str, str], list[tuple[pd.Timestamp, list[int]]]] = {}
    for index, row in df.iterrows():
        vector = row["_section_sketch"].get(section)
        if not isinstance(vector, list):
            continue
        key = (row["ticker"], row["form"])
        current_date = row["report_date"]
        prior = [
            (date, old_vector) for date, old_vector in history.get(key, [])
            if 300 <= (current_date - date).days <= 430
        ]
        if prior:
            date, old_vector = min(prior, key=lambda item: abs((current_date - item[0]).days - 365))
            out[index] = _cosine(vector, old_vector)
        history.setdefault(key, []).append((current_date, vector))
    return out


def build_features(records: list[dict]) -> pd.DataFrame:
    if not records:
        raise ValueError("No extraction records found; run `python main.py extract` first.")
    df = _doc_rows(records)

    # Compare each filing only with the ticker's previous filing of the same form.
    g = df.groupby(["ticker", "form"], group_keys=False)
    for col in ["sentiment", "hedging", "lm_net", "lm_uncertainty", "guidance_score", "n_risks"]:
        df[f"d_{col}"] = g[col].diff()

    # Demean levels using only the firm's past filings of that form.
    for col in ("sentiment", "hedging", "guidance_score"):
        prior_mean = g[col].transform(lambda s: s.expanding(min_periods=1).mean().shift(1))
        df[f"{col}_firm_demeaned"] = df[col] - prior_mean

    past = g["sentiment"]
    mu = past.transform(lambda s: s.expanding(min_periods=4).mean().shift(1))
    sd = past.transform(lambda s: s.expanding(min_periods=4).std().shift(1))
    df["sentiment_z_firm"] = (df["sentiment"] - mu) / sd.replace(0, np.nan)

    df["risk_text_change"] = 1 - df["risk_factors_text_similarity"]
    df["mdna_text_change"] = 1 - df["mdna_text_similarity"]
    for col in ("lm_net", "lm_uncertainty", "risk_text_change", "mdna_text_change"):
        prior_mean = g[col].transform(lambda s: s.expanding(min_periods=1).mean().shift(1))
        df[f"{col}_firm_demeaned"] = df[col] - prior_mean
    return df.drop(columns=["_section_sketch"])


SIGNAL_COLUMNS = [
    "sentiment_firm_demeaned", "d_sentiment", "sentiment_z_firm",
    "hedging_firm_demeaned", "d_hedging",
]
BASELINE_COLUMNS = [
    "lm_net_firm_demeaned", "d_lm_net",
    "lm_uncertainty_firm_demeaned", "d_lm_uncertainty",
    "risk_text_change_firm_demeaned", "mdna_text_change_firm_demeaned",
]
