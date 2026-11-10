"""SEC EDGAR client: ticker->CIK, filing index, document download, section extraction.

All endpoints here are free and need no key, only a User-Agent header.
  - https://www.sec.gov/files/company_tickers.json
  - https://data.sec.gov/submissions/CIK##########.json
  - https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import requests
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

import config

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)  # iXBRL filings are XHTML

_session = requests.Session()
_session.headers.update({"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"})
_last_request = 0.0


def _get(url: str) -> requests.Response:
    """Rate-limited GET (SEC allows 10 req/s)."""
    global _last_request
    wait = config.SEC_MIN_INTERVAL_S - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    for attempt in range(4):
        resp = _session.get(url, timeout=30)
        _last_request = time.time()
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


# ----------------------------------------------------------------------------- lookup
_ticker_map: dict[str, int] | None = None

PREDECESSOR_CIKS: dict[str, list[int]] = {
    "DIS": [1001039],  # TWDC Enterprises 18 Corp. (pre-2019 Disney)
    "XOM": [34088],    # Exxon Mobil Corp (pre-2026 holding-company reorg)
}


def ticker_to_cik(ticker: str) -> int:
    global _ticker_map
    if _ticker_map is None:
        data = _get("https://www.sec.gov/files/company_tickers.json").json()
        _ticker_map = {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
    return _ticker_map[ticker.upper()]


@dataclass
class Filing:
    ticker: str
    cik: int
    form: str
    accession: str
    filing_date: str
    report_date: str
    accepted: str          # acceptanceDateTime, e.g. 2024-02-02T16:31:05.000Z (actually ET, see prices.py)
    primary_doc: str
    items: str             # unused for 10-K/10-Q (EDGAR item codes)

    @property
    def folder_url(self) -> str:
        return f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{self.accession.replace('-', '')}"

    @property
    def url(self) -> str:
        return f"{self.folder_url}/{self.primary_doc}"


def _rows_from_block(block: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "accession": block["accessionNumber"],
        "filing_date": block["filingDate"],
        "report_date": block["reportDate"],
        "accepted": block["acceptanceDateTime"],
        "form": block["form"],
        "primary_doc": block["primaryDocument"],
        "items": block.get("items", [""] * len(block["form"])),
    })


def list_filings(ticker: str, forms=("10-K", "10-Q"), start: str = "2015-01-01") -> list[Filing]:
    """List a ticker's 10-K / 10-Q filings since `start` (amendments like 10-K/A excluded)."""
    frames = []
    for cik in [ticker_to_cik(ticker)] + PREDECESSOR_CIKS.get(ticker.upper(), []):
        sub = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
        blocks = [sub["filings"]["recent"]]
        for f in sub["filings"].get("files", []):  # older filings are paginated
            if f.get("filingTo", "9999") >= start:
                blocks.append(_get(f"https://data.sec.gov/submissions/{f['name']}").json())
        frames += [_rows_from_block(b).assign(cik=cik) for b in blocks]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("accession")

    df = df[df["form"].isin(forms) & (df["filing_date"] >= start)].sort_values("filing_date")
    return [Filing(ticker=ticker.upper(), **row) for row in df.to_dict("records")]


# ----------------------------------------------------------------------------- download
def _cache_path(filing: Filing, name: str) -> Path:
    return config.RAW_DIR / filing.ticker / f"{filing.accession}_{name}"


def download(filing: Filing) -> str:
    """Return raw HTML of the filing's main document. Cached under data/raw/{TICKER}/."""
    path = _cache_path(filing, filing.primary_doc)
    if path.exists():
        return path.read_text(errors="ignore")
    html = _get(filing.url).text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return html


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):  # hidden iXBRL header
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


# ----------------------------------------------------------------------------- sections

SECTIONS = {
    "10-K": {
        "risk_factors": (r"item\s*1a\.?\s*[-–—:]?\s*risk\s+factors", r"item\s*1b\.?|item\s*1c\.?|item\s*2\.?\s*[-–—:]?\s*properties"),
        "mdna": (r"item\s*7\.?\s*[-–—:]?\s*management.{0,5}s\s+discussion", r"item\s*7a\.?|item\s*8\.?\s*[-–—:]?\s*financial\s+statements"),
    },
    "10-Q": {
        "mdna": (r"item\s*2\.?\s*[-–—:]?\s*management.{0,5}s\s+discussion", r"item\s*3\.?\s*[-–—:]?\s*quantitative"),
        "risk_factors": (r"item\s*1a\.?\s*[-–—:]?\s*risk\s+factors", r"item\s*2\.?\s*[-–—:]?\s*unregistered|item\s*3\.?\s*[-–—:]?\s*defaults|item\s*5\.?|item\s*6\.?\s*[-–—:]?\s*exhibits"),
    },
}


def extract_section(text: str, start_pat: str, end_pat: str, min_len: int = 500) -> str:
    best = ""
    for m in re.finditer(start_pat, text, re.I):
        end = re.compile(end_pat, re.I).search(text, m.end())
        span = text[m.start(): end.start() if end else len(text)]
        if len(span) > len(best):
            best = span
    return best if len(best) >= min_len else ""


def extract_sections(form: str, text: str) -> dict[str, str]:
    """Return {section_name: text}. Falls back to the full document if parsing fails."""
    out = {}
    for name, (s, e) in SECTIONS.get(form, {}).items():
        sec = extract_section(text, s, e)
        if sec:
            out[name] = sec
    return out or {"full_text": text}


def filing_to_record(f: Filing) -> dict:
    return asdict(f) | {"url": f.url}
