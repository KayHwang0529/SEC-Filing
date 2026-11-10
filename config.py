"""Central configuration. Override any value with an environment variable."""
import os
from pathlib import Path

# --- Kimi (Moonshot AI) -------------------------------------------------------

APIKEY = os.getenv("KIMI_API_KEY", "APIKEY")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")  
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2-0905-preview")
KIMI_TEMPERATURE = float(os.getenv("KIMI_TEMPERATURE", "0.1"))  # low = more reproducible extraction
KIMI_MAX_CHARS_PER_CALL = int(os.getenv("KIMI_MAX_CHARS_PER_CALL", "300000"))  # ~75k tokens
PROMPT_VERSION = "v1" 

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

for _d in (DATA_DIR, RAW_DIR, CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Study design -------------------------------------------------------------
BENCHMARK = "SPY"
HORIZONS = [1, 5, 20, 60]        # trading days after the event
TEST_START = os.getenv("TEST_START", "2023-01-01")  # out-of-sample period begins here
EMBARGO_DAYS = max(HORIZONS)     # gap between train and test so return windows don't overlap
