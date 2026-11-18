
from __future__ import annotations

import hashlib
import json
import re
import time
import warnings
from collections import Counter

import config

SYSTEM_PROMPT = """You are a financial analyst extracting structured data from SEC 10-K and \
10-Q filings. Be literal and conservative: only report what the text states. \
Respond with a single JSON object and nothing else."""

EXTRACTION_PROMPT = """Company: {ticker}  |  Document: {form}  |  Section: {section}  |  Date: {date}

Return JSON with exactly these keys:
{{
  "sentiment": <float -1.0 (very negative) .. 1.0 (very positive), management's overall tone>,
  "tone": <"optimistic" | "neutral" | "cautious" | "negative">,
  "hedging": <float 0..1, how much uncertain/hedged language ("may", "could", "uncertain")>,
  "guidance": {{
    "provided": <bool>,
    "direction": <"raised" | "maintained" | "lowered" | "initiated" | "withdrawn" | "none">,
    "items": [{{"metric": str, "period": str, "low": number|null, "high": number|null,
               "unit": str, "direction": <same options as above>}}]
  }},
  "risk_factors": [{{"title": <short canonical title, <=12 words>,
                     "category": <"macro"|"competition"|"regulatory"|"legal"|"supply_chain"|
                                  "cyber"|"financial"|"operational"|"geopolitical"|"other">,
                     "severity": <int 1..5>}}],
  "key_topics": [<up to 8 short strings>],
  "notable_quotes": [<up to 3 short verbatim quotes that drive the sentiment score>]
}}

Use "none"/empty lists when the section has no guidance or risk factors.

TEXT:
\"\"\"
{text}
\"\"\""""

_client = None
PROMPT_HASH = hashlib.sha256(f"{SYSTEM_PROMPT}\n{EXTRACTION_PROMPT}".encode()).hexdigest()


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=config.APIKEY, base_url=config.KIMI_BASE_URL)
    return _client


def _cache_key(text: str, meta: dict) -> str:
    h = hashlib.sha256()
    model = "dictionary-mock" if config.MOCK_MODE else config.KIMI_MODEL
    h.update(f"{model}|{config.PROMPT_VERSION}|{PROMPT_HASH}|{config.MOCK_MODE}|".encode())
    h.update(json.dumps(meta, sort_keys=True).encode())
    h.update(text.encode())
    return h.hexdigest()


def _call_kimi(prompt: str, retries: int = 4) -> dict:
    client = _get_client()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=config.KIMI_MODEL,
                temperature=config.KIMI_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Occasionally the model wraps JSON in prose; salvage the outer object.
            m = re.search(r"\{.*\}", content or "", re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError as salvage_error:
                    e = salvage_error
            if attempt == retries - 1:
                raise e
            time.sleep(2 ** attempt * 2)
            print(f"  Kimi retry {attempt + 1}: {e}")
        except Exception as e:  # rate limit / transient network
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 2)
            print(f"  Kimi retry {attempt + 1}: {e}")
    raise RuntimeError("Kimi extraction failed")


# ----------------------------------------------------------------------------- mock
# These small sets are only an offline fallback; a full LM dictionary can be
# supplied at config.LM_DICTIONARY_PATH.
LM_POS = set("achieve attain benefit gain gains growth improve improved improvement increase "
             "increased opportunity profitable record strong stronger exceed exceeded favorable".split())
LM_NEG = set("adverse decline declined decrease loss losses impairment litigation weak weaker "
             "challenging difficult unfavorable restructuring default volatility headwinds".split())
LM_UNC = set("may could might uncertain uncertainty approximately risk possible depend "
             "believe anticipate fluctuate".split())

_lm_words: dict[str, set[str]] | None = None
_lm_source_reported = False


def _load_lm_words() -> dict[str, set[str]]:
    global _lm_words, _lm_source_reported
    if _lm_words is not None:
        return _lm_words
    path = config.LM_DICTIONARY_PATH
    if path.exists():
        import pandas as pd

        table = pd.read_csv(path)
        columns = {str(c).strip().lower(): c for c in table.columns}
        word_column = columns.get("word")
        categories = ("positive", "negative", "uncertainty")
        if word_column is None or any(c not in columns for c in categories):
            raise ValueError(
                f"{path} must contain Word, Positive, Negative, and Uncertainty columns"
            )
        words = table[word_column].astype(str).str.lower()
        _lm_words = {
            category: set(words[table[columns[category]].fillna(0).astype(float) > 0])
            for category in categories
        }
    else:
        _lm_words = {"positive": LM_POS, "negative": LM_NEG, "uncertainty": LM_UNC}
        if not _lm_source_reported:
            warnings.warn(
                f"Full Loughran-McDonald dictionary not found at {path}; "
                "using the small built-in fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            _lm_source_reported = True
    return _lm_words


def lm_dictionary_source() -> str:
    return str(config.LM_DICTIONARY_PATH) if config.LM_DICTIONARY_PATH.exists() else "built-in miniature fallback"


def dictionary_fingerprint() -> str:
    if config.LM_DICTIONARY_PATH.exists():
        return hashlib.sha256(config.LM_DICTIONARY_PATH.read_bytes()).hexdigest()
    fallback = "\n".join(
        ",".join(sorted(words)) for words in (LM_POS, LM_NEG, LM_UNC)
    )
    return hashlib.sha256(fallback.encode()).hexdigest()


def dictionary_scores(text: str) -> dict:
    words = re.findall(r"[a-z]+", text.lower())
    n = max(len(words), 1)
    categories = _load_lm_words()
    pos = sum(w in categories["positive"] for w in words)
    neg = sum(w in categories["negative"] for w in words)
    unc = sum(w in categories["uncertainty"] for w in words)
    return {"lm_net": (pos - neg) / max(pos + neg, 1), "lm_uncertainty": unc / n, "n_words": n}


def text_sketch(text: str, dimensions: int = 512) -> list[int]:
    """Create a stable signed-hash bag-of-words vector for text-change comparisons."""
    counts = Counter(re.findall(r"[a-z]+", text.lower()))
    vector = [0] * dimensions
    for word, count in counts.items():
        digest = hashlib.sha256(word.encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % dimensions
        sign = 1 if digest[8] & 1 else -1
        vector[bucket] += sign * count
    return vector


def _mock_extract(text: str) -> dict:
    d = dictionary_scores(text)
    low = text.lower()
    direction = ("raised" if re.search(r"rais(e|ed|ing) (our |its )?(full[- ]year )?(guidance|outlook)", low)
                 else "lowered" if re.search(r"(lower|reduc|cut)(ed|ing)? (our |its )?(guidance|outlook)", low)
                 else "none")
    risks = re.findall(r"\n([A-Z][^\n.]{20,120}(?:could|may)[^\n.]{5,120}\.)", text)[:15]
    return {
        "sentiment": round(d["lm_net"], 4),
        "tone": "neutral",
        "hedging": round(min(d["lm_uncertainty"] * 20, 1.0), 4),
        "guidance": {"provided": direction != "none", "direction": direction, "items": []},
        "risk_factors": [{"title": r.strip()[:80], "category": "other", "severity": 3} for r in risks],
        "key_topics": [],
        "notable_quotes": [],
        "_mock": True,
    }


# ----------------------------------------------------------------------------- public
def _chunks(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    paras, out, cur = text.split("\n\n"), [], ""
    for p in paras:
        if len(cur) + len(p) > size and cur:
            out.append(cur)
            cur = ""
        cur += p + "\n\n"
    return out + ([cur] if cur else [])


def _merge(parts: list[dict], weights: list[int]) -> dict:
    """Combine chunk-level extractions: length-weighted means, union of lists."""
    if len(parts) == 1:
        return parts[0]
    w = [x / sum(weights) for x in weights]
    order = ["withdrawn", "lowered", "raised", "initiated", "maintained", "none"]  # most material wins
    dirs = [p.get("guidance", {}).get("direction", "none") for p in parts]
    seen, risks = set(), []
    for p in parts:
        for r in p.get("risk_factors", []):
            k = r.get("title", "").lower()
            if k and k not in seen:
                seen.add(k)
                risks.append(r)
    return {
        "sentiment": sum(wi * float(p.get("sentiment", 0)) for wi, p in zip(w, parts)),
        "tone": parts[max(range(len(parts)), key=lambda i: weights[i])].get("tone", "neutral"),
        "hedging": sum(wi * float(p.get("hedging", 0)) for wi, p in zip(w, parts)),
        "guidance": {
            "provided": any(p.get("guidance", {}).get("provided") for p in parts),
            "direction": min(dirs, key=lambda d: order.index(d) if d in order else len(order)),
            "items": [i for p in parts for i in p.get("guidance", {}).get("items", [])],
        },
        "risk_factors": risks,
        "key_topics": list(dict.fromkeys(t for p in parts for t in p.get("key_topics", [])))[:12],
        "notable_quotes": [q for p in parts for q in p.get("notable_quotes", [])][:5],
    }


def extract(text: str, ticker: str, form: str, section: str, date: str) -> dict:
    """Extract structured signals from one document section (cached)."""
    meta = {"ticker": ticker, "form": form, "section": section, "date": date}
    path = config.CACHE_DIR / f"{_cache_key(text, meta)}.json"
    if path.exists():
        result = json.loads(path.read_text())
        result["dictionary"] = dictionary_scores(text)
        result["text_sketch"] = text_sketch(text)
        result["dictionary_hash"] = dictionary_fingerprint()
        result["dictionary_source"] = lm_dictionary_source()
        path.write_text(json.dumps(result))
        return result

    if config.MOCK_MODE:
        result = _mock_extract(text)
    else:
        chunks = _chunks(text, config.KIMI_MAX_CHARS_PER_CALL)
        parts = [_call_kimi(EXTRACTION_PROMPT.format(text=c, **meta)) for c in chunks]
        result = _merge(parts, [len(c) for c in chunks])

    result["dictionary"] = dictionary_scores(text)  # always keep the baseline alongside
    result["text_sketch"] = text_sketch(text)
    result["extraction_mode"] = "mock" if config.MOCK_MODE else "kimi"
    result["model"] = "dictionary-mock" if config.MOCK_MODE else config.KIMI_MODEL
    result["prompt_version"] = config.PROMPT_VERSION
    result["prompt_hash"] = PROMPT_HASH
    result["dictionary_hash"] = dictionary_fingerprint()
    result["dictionary_source"] = lm_dictionary_source()
    path.write_text(json.dumps(result))
    return result
