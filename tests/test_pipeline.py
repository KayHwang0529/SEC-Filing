import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
import features
import kimi_extractor
import main
import model
import prices
import sec_client


class SectionParsingTests(unittest.TestCase):
    def test_inline_cross_reference_does_not_end_section(self):
        text = (
            "ITEM 1. BUSINESS\n"
            + "Business description " * 25
            + "\nSee Item 1A. Risk Factors for discussion.\n"
            + "Additional business information " * 25
            + "\nITEM 1A. RISK FACTORS\n"
            + "Risk discussion " * 25
            + "\nITEM 1B. UNRESOLVED STAFF COMMENTS\n"
        )
        section = sec_client.extract_section(
            text,
            r"item\s*1\.?\s*[-–—:]?\s*business",
            r"item\s*1a\.?\s*[-–—:]?\s*risk factors",
            min_len=20,
        )
        self.assertIn("Additional business information", section)
        self.assertNotIn("Risk discussion", section)

    def test_parse_failure_is_empty_not_full_text(self):
        self.assertEqual(sec_client.extract_sections("10-K", "unstructured filing content"), {})


class ExtractionTests(unittest.TestCase):
    def test_done_keys_are_scoped_to_mode_model_and_prompt(self):
        record = {
            "accession": "a1",
            "section": "mdna",
            "mode": "mock",
            "model": "dictionary-mock",
            "prompt_version": config.PROMPT_VERSION,
            "prompt_hash": kimi_extractor.PROMPT_HASH,
            "dictionary_hash": kimi_extractor.dictionary_fingerprint(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extractions.jsonl"
            path.write_text(json.dumps(record) + "\n")
            with patch.object(config, "EXTRACTIONS_JSONL", path), patch.object(
                config, "MOCK_MODE", True
            ):
                self.assertIn((
                    "a1", "mdna", "mock", "dictionary-mock",
                    kimi_extractor.PROMPT_HASH, kimi_extractor.dictionary_fingerprint(),
                ), main._done_keys())
            with patch.object(config, "EXTRACTIONS_JSONL", path), patch.object(
                config, "MOCK_MODE", False
            ):
                self.assertEqual(main._done_keys(), set())
            for field in ("model", "prompt_version", "prompt_hash", "dictionary_hash"):
                stale = record | {field: "stale"}
                path.write_text(json.dumps(stale) + "\n")
                with patch.object(config, "EXTRACTIONS_JSONL", path), patch.object(
                    config, "MOCK_MODE", True
                ):
                    self.assertEqual(main._done_keys(), set(), field)

    def test_partial_document_resume_processes_only_missing_section(self):
        mode, model_name, prompt_version, prompt_hash, dictionary_hash = (
            main._extraction_identity()
        )
        existing = {
            "accession": "a1",
            "section": "mdna",
            "mode": mode,
            "model": model_name,
            "prompt_version": prompt_version,
            "prompt_hash": prompt_hash,
            "dictionary_hash": dictionary_hash,
        }
        meta = {
            "ticker": "AAA",
            "form": "10-K",
            "accession": "a1",
            "filing_date": "2024-01-01",
            "accepted": "2024-01-01T15:00:00.000Z",
            "report_date": "2023-12-31",
            "sector": "Industrials",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extractions.jsonl"
            path.write_text(json.dumps(existing) + "\n")
            with patch.object(config, "EXTRACTIONS_JSONL", path), patch.object(
                config, "MOCK_MODE", True
            ), patch.object(
                main, "_documents",
                return_value=[(meta, lambda: {"mdna": "mdna", "risk_factors": "risks"})],
            ), patch.object(
                kimi_extractor, "extract", return_value={"dictionary": {}, "text_sketch": []}
            ) as extract:
                main.cmd_extract(Namespace(
                    tickers=None, sections=None, limit=0, dry_run=False
                ))
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["section"], "risk_factors")
        extract.assert_called_once_with("risks", "AAA", "10-K", "risk_factors", "2024-01-01")

    def test_invalid_salvaged_json_retries(self):
        class FakeCompletions:
            def __init__(self):
                self.responses = ["text {not json}", '{"sentiment": 0.2}']

            def create(self, **kwargs):
                content = self.responses.pop(0)
                return type("Response", (), {
                    "choices": [type("Choice", (), {
                        "message": type("Message", (), {"content": content})()
                    })()]
                })()

        fake_client = type("Client", (), {
            "chat": type("Chat", (), {"completions": FakeCompletions()})()
        })()
        with patch.object(kimi_extractor, "_get_client", return_value=fake_client), patch.object(
            kimi_extractor.time, "sleep"
        ):
            self.assertEqual(kimi_extractor._call_kimi("prompt"), {"sentiment": 0.2})

    def test_full_dictionary_loader_uses_configured_csv(self):
        table = (
            "Word,Positive,Negative,Uncertainty\n"
            "better,1,0,0\n"
            "risk,0,0,1\n"
            "decline,0,1,0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dictionary.csv"
            path.write_text(table)
            with patch.object(config, "LM_DICTIONARY_PATH", path), patch.object(
                kimi_extractor, "_lm_words", None
            ):
                scores = kimi_extractor.dictionary_scores("better risk decline")
        self.assertEqual(scores["lm_net"], 0)
        self.assertAlmostEqual(scores["lm_uncertainty"], 1 / 3)

    def test_dictionary_baseline_is_recomputed_when_extraction_cache_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dictionary = root / "dictionary.csv"
            cache = root / "cache"
            cache.mkdir()
            dictionary.write_text("Word,Positive,Negative,Uncertainty\nword,1,0,0\n")
            with patch.object(config, "LM_DICTIONARY_PATH", dictionary), patch.object(
                config, "CACHE_DIR", cache
            ), patch.object(config, "MOCK_MODE", True), patch.object(
                kimi_extractor, "_lm_words", None
            ):
                first = kimi_extractor.extract("word", "AAA", "10-K", "mdna", "2024-01-01")
                dictionary.write_text("Word,Positive,Negative,Uncertainty\nword,0,1,0\n")
                kimi_extractor._lm_words = None
                second = kimi_extractor.extract("word", "AAA", "10-K", "mdna", "2024-01-01")
        self.assertEqual(first["dictionary"]["lm_net"], 1)
        self.assertEqual(second["dictionary"]["lm_net"], -1)

    def test_limited_docs_are_round_robin_sampled_by_ticker_and_form(self):
        docs = [
            ({"ticker": "AAA", "form": "10-K"}, "a1"),
            ({"ticker": "AAA", "form": "10-K"}, "a2"),
            ({"ticker": "AAA", "form": "10-Q"}, "a3"),
            ({"ticker": "BBB", "form": "10-K"}, "b1"),
        ]
        selected = main._limit_across_tickers_and_forms(docs, 3)
        self.assertEqual({doc[1] for doc in selected}, {"a1", "a3", "b1"})


class FeatureTests(unittest.TestCase):
    def _record(self, ticker, form, accession, filing_date, sentiment, sketch):
        return {
            "ticker": ticker,
            "form": form,
            "accession": accession,
            "filing_date": filing_date,
            "report_date": filing_date,
            "accepted": f"{filing_date}T15:00:00.000Z",
            "section": "mdna",
            "extraction": {
                "sentiment": sentiment,
                "hedging": 0.2,
                "guidance": {"provided": False, "direction": "none"},
                "dictionary": {
                    "lm_net": sentiment,
                    "lm_uncertainty": 0.1,
                    "n_words": 100,
                },
                "text_sketch": sketch,
            },
        }

    def test_changes_compare_with_same_form_and_cosine_uses_prior_year(self):
        records = [
            self._record("AAA", "10-Q", "q1", "2020-03-31", 0.1, [1, 0]),
            self._record("AAA", "10-K", "k1", "2020-12-31", 0.8, [0, 1]),
            self._record("AAA", "10-Q", "q2", "2021-03-31", 0.4, [1, 0]),
        ]
        df = features.build_features(records).set_index("accession")
        self.assertAlmostEqual(df.loc["q2", "d_sentiment"], 0.3)
        self.assertAlmostEqual(df.loc["q2", "mdna_text_similarity"], 1.0)
        self.assertTrue(np.isnan(df.loc["q1", "mdna_text_similarity"]))

    def test_preprocessor_fits_separately_by_form(self):
        train = pd.DataFrame({
            "form": ["10-K", "10-K", "10-Q", "10-Q"],
            "x": [10.0, 20.0, 100.0, 200.0],
        })
        test = pd.DataFrame({"form": ["10-K", "10-Q"], "x": [20.0, 200.0]})
        _, scaled = model._preprocess_by_form(train, test, ["x"])
        self.assertAlmostEqual(scaled.iloc[0]["x"], 0.70710678)
        self.assertAlmostEqual(scaled.iloc[1]["x"], 0.70710678)


class PriceAndStudyTests(unittest.TestCase):
    def test_acceptance_timestamp_is_converted_from_utc_to_eastern(self):
        days = pd.date_range("2024-01-03", "2024-01-05", freq="B")
        self.assertEqual(
            prices.event_day("2024-01-03T20:59:00.000Z", days),
            pd.Timestamp("2024-01-03"),
        )
        self.assertEqual(
            prices.event_day("2024-01-03T21:00:00.000Z", days),
            pd.Timestamp("2024-01-04"),
        )
        summer_days = pd.date_range("2015-07-31", "2015-08-03", freq="B")
        self.assertEqual(
            prices.event_day("2015-07-31T19:07:52.000Z", summer_days),
            pd.Timestamp("2015-07-31"),
        )

    def test_earnings_return_attaches_from_prior_202_item_8k(self):
        days = pd.date_range("2024-01-02", "2024-01-08", freq="B")
        px = pd.DataFrame({
            "AAA": [100, 102, 103, 105, 106],
            "SPY": [100, 100, 101, 101, 101],
            "XLI": [100, 100, 101, 101, 101],
        }, index=days)
        events = pd.DataFrame([{
            "ticker": "AAA",
            "sector": "Industrials",
            "accepted": "2024-01-05T15:00:00.000Z",
        }])
        filings = pd.DataFrame([{
            "ticker": "AAA",
            "sector": "Industrials",
            "accepted": "2024-01-03T22:00:00.000Z",
            "items": "2.02 Results of Operations",
        }])
        result = prices.attach_earnings_control(events, filings, px)
        self.assertEqual(result.loc[0, "earnings_date"], pd.Timestamp("2024-01-04"))
        self.assertAlmostEqual(result.loc[0, "earnings_car"], np.log(103 / 102) - np.log(101 / 100))

    def test_event_return_uses_sector_etf(self):
        days = pd.date_range("2024-01-02", "2024-01-05", freq="B")
        px = pd.DataFrame({
            "AAA": [100.0, 100.0, 110.0, 110.0],
            "SPY": [100.0, 100.0, 100.0, 100.0],
            "XLI": [100.0, 100.0, 105.0, 105.0],
        }, index=days)
        events = pd.DataFrame([{
            "ticker": "AAA",
            "sector": "Industrials",
            "accepted": "2024-01-03T15:00:00.000Z",
        }])
        result = prices.event_returns(events, px, horizons=[1])
        self.assertAlmostEqual(result.loc[0, "car_1_1"], np.log(1.1) - np.log(1.05))

    def test_price_cache_refreshes_when_latest_filing_is_past_cache_end(self):
        today = pd.Timestamp.today().normalize()
        yesterday = today - pd.offsets.BDay(1)
        symbols = {"AAA", config.BENCHMARK} | set(config.SECTOR_TO_ETF.values())
        cached = pd.DataFrame(
            100.0,
            index=pd.DatetimeIndex([yesterday]),
            columns=sorted(symbols),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            cached.to_csv(path)
            calls = []

            def download(tickers, start, end, auto_adjust, progress):
                calls.append((tickers, start))
                return {"Close": pd.DataFrame(
                    101.0,
                    index=pd.DatetimeIndex([today]),
                    columns=sorted(symbols),
                )}

            with patch.object(config, "PRICES_CSV", path), patch.dict(
                sys.modules, {"yfinance": SimpleNamespace(download=download)}
            ):
                result = prices.download_prices(
                    ["AAA"],
                    start=(today - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                    required_through=today.strftime("%Y-%m-%d"),
                )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.index.max(), today)

    def test_run_study_emits_annual_walk_forward_ic(self):
        rows = []
        for date in pd.date_range("2018-03-15", "2026-03-15", freq="QS"):
            for firm in range(12):
                signal = float((firm + date.quarter) % 12)
                rows.append({
                    "event_day": date,
                    "ticker": f"T{firm}",
                    "form": "10-K" if date.quarter == 1 else "10-Q",
                    "sig": signal,
                    "earnings_car": signal / 100,
                    "pre_ret_60": 0.01 * signal,
                    "pre_vol_60": 0.1 + signal / 100,
                    "car_1_1": signal / 1000 + firm / 10000,
                })
        result = model.run_study(
            pd.DataFrame(rows),
            signals=["sig"],
            baselines=[],
            horizons=[1],
            windows=("car_1",),
            n_perm=2,
        )
        self.assertEqual(
            set(result["yearly_ic"]["year"].unique()), {2023, 2024, 2025, 2026}
        )


if __name__ == "__main__":
    unittest.main()
