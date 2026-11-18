"""Central configuration. Override any value with an environment variable."""
import os
from pathlib import Path

# --- Kimi (Moonshot AI) -------------------------------------------------------

APIKEY = os.getenv("KIMI_API_KEY", "APIKEY")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")  
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2-0905-preview")
KIMI_TEMPERATURE = float(os.getenv("KIMI_TEMPERATURE", "0.1"))  # low = more reproducible extraction
KIMI_MAX_CHARS_PER_CALL = int(os.getenv("KIMI_MAX_CHARS_PER_CALL", "300000"))  # ~75k tokens
PROMPT_VERSION = "v2"

MOCK_MODE = APIKEY in ("", "APIKEY")

# --- SEC EDGAR ----------------------------------------------------------------
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "SECFilingAnalyzer contact@example.com")
SEC_MIN_INTERVAL_S = 0.12  # SEC fair-access limit is 10 req/s

# --- Paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                # cached EDGAR HTML
CACHE_DIR = DATA_DIR / "llm_cache"        # cached Kimi responses
FILINGS_CSV = DATA_DIR / "filings.csv"
EXTRACTIONS_JSONL = DATA_DIR / "extractions.jsonl"
FEATURES_CSV = DATA_DIR / "features.csv"
PRICES_CSV = DATA_DIR / "prices.csv"
DATASET_CSV = DATA_DIR / "dataset.csv"
RESULTS_DIR = DATA_DIR / "results"
LM_DICTIONARY_PATH = Path(os.getenv(
    "LM_DICTIONARY_PATH", str(DATA_DIR / "Loughran-McDonald_MasterDictionary_1993-2025.csv")
))

for _d in (DATA_DIR, RAW_DIR, CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Study design -------------------------------------------------------------
BENCHMARK = "SPY"
HORIZONS = [1, 5, 20, 60]        # trading days after the event
# Moonshot's Kimi-K2-Instruct-0905 model card does not state a training cutoff.
# Use the public release date as a conservative boundary, not as a claimed cutoff.
MODEL_BOUNDARY_DATE = os.getenv("MODEL_BOUNDARY_DATE", "2025-09-05")
TEST_START = os.getenv("TEST_START", MODEL_BOUNDARY_DATE)
EMBARGO_DAYS = max(HORIZONS)     # gap between train and test so return windows don't overlap

SECTOR_TO_ETF = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

TICKER_SECTORS = {
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "NVDA": "Information Technology", "ORCL": "Information Technology",
    "CSCO": "Information Technology", "INTC": "Information Technology",
    "IBM": "Information Technology", "ADBE": "Information Technology",
    "CRM": "Information Technology",
    "GOOGL": "Communication Services", "META": "Communication Services",
    "VZ": "Communication Services", "T": "Communication Services",
    "CMCSA": "Communication Services", "DIS": "Communication Services",
    "AMZN": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "TGT": "Consumer Discretionary",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "C": "Financials", "GS": "Financials", "MS": "Financials",
    "AXP": "Financials",
    "JNJ": "Health Care", "PFE": "Health Care", "MRK": "Health Care",
    "ABBV": "Health Care", "UNH": "Health Care", "LLY": "Health Care",
    "AMGN": "Health Care", "BMY": "Health Care",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "WMT": "Consumer Staples",
    "COST": "Consumer Staples",
    "CAT": "Industrials", "DE": "Industrials", "BA": "Industrials",
    "HON": "Industrials", "GE": "Industrials", "MMM": "Industrials",
    "UPS": "Industrials", "LMT": "Industrials",
    "NEE": "Utilities", "DUK": "Utilities",
}
