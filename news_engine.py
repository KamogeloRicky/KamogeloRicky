"""
Robust news_engine for RickyFX bot.

Improvements (kept backwards compatible):
- Reads NEWSAPI key from common environment variable names.
- Adds retries and timeouts with simple exponential backoff.
- Adds a Google News RSS fallback when APIs fail or return nothing.
- Returns a consistent dict including "status" ("ok", "no_news", "api_error") and "errors" (list)
  so callers (analysis.py) can disambiguate "no relevant news" vs "API unavailable".
- Does not remove any existing keys; preserves pair, count, sentiment_score, headlines, timestamp.
- Lightweight prints for visibility; safe for headless use.

New (additive, non-breaking):
- Economic calendar support via CSV (ANALYSIS_NEWS_CSV or BACKTEST_NEWS_CSV), with caching and UTC normalization.
- 30-minute blackout helpers:
    - get_upcoming_events(window_min=30, currencies=None)
    - is_blackout(now=None, pair=None, currencies=None, window_min=30) -> bool
    - blackout_penalty(now=None, pair=None, currencies=None, window_min=30) -> float (0.0–0.2)
- Currency/instrument extraction with index-awareness (e.g., US30/DJI, NAS100/NDX, SPX500/S&P500, UK100/FTSE100, DE40/DAX, JP225/Nikkei, HK50/HSI).
- Smarter query builder that uses index aliases and finance-only qualifiers.
- Headline relevance filter to drop non-financial/sports items like "UK 100m title" athletics results.
"""

from __future__ import annotations

import os
import requests
import datetime as dt
import time
import xml.etree.ElementTree as ET
import html
import csv
import re
from typing import List, Dict, Any, Optional, Tuple

# -------------------- API Keys (more robust env lookups) --------------------
NEWSAPI_KEY = (
    os.getenv("NEWSAPI_KEY")
    or os.getenv("NEWS_API_KEY")
    or os.getenv("NEWSAPI")
    or os.getenv("NEWS_API")
    or os.getenv("dd7b030786b543fxa2d452e5430d5c22")  # legacy: keep compatibility with prior env name usage
    or None
)
FOREXNEWSAPI_KEY = os.getenv("FOREXNEWSAPI_KEY") or os.getenv("FOREX_NEWSAPI_KEY") or None

# -------------------- Endpoints --------------------
NEWS_URL = "https://newsapi.org/v2/everything"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

# -------------------- Calendar / Blackout config --------------------
NEWS_CSV_PATH = os.getenv("ANALYSIS_NEWS_CSV", os.getenv("BACKTEST_NEWS_CSV", "")) or ""
NEWS_TTL_SEC = int(os.getenv("NEWS_TTL_SEC", "300"))          # cache CSV for 5 minutes
NEWS_BLACKOUT_MIN = int(os.getenv("NEWS_BLACKOUT_MIN", "30"))  # 30-minute blackout window by default
SESSION_TZ = os.getenv("ANALYSIS_SESSION_TZ", "Africa/Johannesburg")

# Optional domain allowlist to improve relevance (comma-separated)
NEWS_DOMAINS = os.getenv("NEWS_DOMAINS", "")  # e.g. "reuters.com,bloomberg.com,ft.com,cnbc.com,wsj.com"
EXCLUDE_SPORTS = os.getenv("NEWS_EXCLUDE_SPORTS", "1") != "0"

# in-memory cache for calendar CSV
_CAL_CACHE: Dict[str, Any] = {
    "path": None,
    "mtime": None,
    "loaded_at": 0.0,
    "ttl": NEWS_TTL_SEC,
    "events": [],  # list of dicts {time(UTC), local_time, impact, currencies[], title, source, id}
}

# -------------------- Time helpers --------------------
def _now_utc() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)


def _to_utc(ts: Any) -> Optional[dt.datetime]:
    try:
        # Accept ISO string or datetime; always return aware UTC datetime
        if isinstance(ts, dt.datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=dt.timezone.utc)
            return ts.astimezone(dt.timezone.utc)
        # Try to parse common formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return dt.datetime.strptime(str(ts), fmt).replace(tzinfo=dt.timezone.utc)
            except Exception:
                pass
        # Fallback: try fromisoformat (may need to strip Z)
        s = str(ts).rstrip("Z")
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            return d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(dt.timezone.utc)
    except Exception:
        return None


def _fmt_utc(ts: dt.datetime) -> str:
    try:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        return ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""

# -------------------- Pair/Instrument helpers --------------------
_SUFFIX_RE = re.compile(r"[A-Za-z]{1}$")

def _strip_suffix(symbol: str) -> str:
    """
    Remove common 1-letter broker suffixes like 'c', 'm', 'r' (case-insensitive).
    E.g., EURUSDc -> EURUSD, ETHUSDm -> ETHUSD, US30m -> US30
    """
    s = symbol.strip()
    if len(s) >= 2 and s[-1].isalpha():
        return s[:-1]
    return s


def _extract_currencies(pair: str) -> List[str]:
    """
    Extract relevant calendar currencies from a symbol.
    EURUSD, GBPUSD -> ["EUR","USD"]
    XAUUSD -> ["USD"] (map metals to USD influence)
    ETHUSD, BTCUSD -> ["USD"] (crypto macro sensitivity)
    USDJPY -> ["USD","JPY"]
    For stock indices, return a representative currency (e.g., US indices -> ["USD"], DE40 -> ["EUR"], JP225 -> ["JPY"])
    """
    p = _strip_suffix(pair).upper().replace("_", "")
    # Indices mapping
    if _is_index_symbol(p):
        if p.startswith(("US", "SPX", "SP", "NAS", "DJ", "DJI", "DOW")):
            return ["USD"]
        if p.startswith(("UK", "FTSE")):
            return ["GBP"]
        if p.startswith(("DE", "GER", "DAX")):
            return ["EUR"]
        if p.startswith(("FR", "CAC")):
            return ["EUR"]
        if p.startswith(("JP", "NIK", "N225", "NI225")):
            return ["JPY"]
        if p.startswith(("HK", "HSI", "HANGSENG")):
            return ["HKD"]
        if p.startswith(("AU", "ASX", "XJO")):
            return ["AUD"]
        return ["USD"]  # default
    # Metals and common CFDs
    if p.startswith(("XAU", "XAG", "XPT", "XPD")):
        return ["USD"]
    if p.startswith(("BTC", "ETH")):
        return ["USD"]
    # Forex 6-letter code
    if len(p) >= 6 and p[:3].isalpha() and p[3:6].isalpha():
        return [p[:3], p[3:6]]
    if p.endswith("USD"):
        return ["USD"]
    return []


def _is_index_symbol(sym: str) -> bool:
    s = sym.upper()
    aliases = (
        "US30", "DJ30", "DJI", "DOW", "DOWJONES",
        "NAS100", "NDX", "NASDAQ100", "NASDQ100",
        "SPX500", "SP500", "SPX", "US500", "S&P500", "SNP500",
        "UK100", "FTSE100", "UKX",
        "DE40", "GER40", "DAX", "DE30", "GER30",
        "FR40", "CAC40",
        "JP225", "NIKKEI", "N225", "NI225",
        "HK50", "HSI", "HANGSENG",
        "CHINA50", "CN50",
        "ES35", "ESP35", "IBEX35",
        "IT40", "FTSEMIB",
        "AU200", "ASX200", "XJO",
        "EU50", "STOXX50", "SX5E",
    )
    if s in aliases:
        return True
    # Heuristic: two letters + digits or digits suffix like DE40, FR40, JP225
    return bool(re.match(r"^[A-Z]{2,3}\d{2,3}$", s))


def _index_aliases(sym: str) -> List[str]:
    s = sym.upper()
    m = {
        "US30": ["US30", "Dow Jones", "DJI", "Dow", "DJIA"],
        "DJI": ["DJI", "Dow Jones", "US30", "DJIA"],
        "DOW": ["Dow Jones", "DJI", "US30", "DJIA"],
        "NAS100": ["Nasdaq 100", "NASDAQ 100", "NDX", "NAS100"],
        "NDX": ["NDX", "Nasdaq 100", "NASDAQ 100", "NAS100"],
        "SPX500": ["S&P 500", "SPX", "SP500", "US500", "S&P500"],
        "SP500": ["S&P 500", "SP500", "SPX", "US500"],
        "SPX": ["S&P 500", "SPX", "US500"],
        "UK100": ["FTSE 100", "UK100", "FTSE100", "UKX"],
        "FTSE100": ["FTSE 100", "UK100", "UKX"],
        "DE40": ["DAX", "GER40", "DE40", "DAX 40"],
        "GER40": ["DAX", "GER40", "DE40", "DAX 40"],
        "FR40": ["CAC 40", "FR40"],
        "CAC40": ["CAC 40", "FR40"],
        "JP225": ["Nikkei 225", "N225", "JP225", "Nikkei"],
        "N225": ["Nikkei 225", "N225", "JP225", "Nikkei"],
        "HK50": ["Hang Seng", "HSI", "HK50"],
        "HSI": ["Hang Seng", "HSI", "HK50"],
        "CHINA50": ["China A50", "CN50", "CHINA50"],
        "CN50": ["China A50", "CN50"],
        "ES35": ["IBEX 35", "ESP35", "ES35", "IBEX"],
        "ESP35": ["IBEX 35", "ESP35", "ES35", "IBEX"],
        "IT40": ["FTSE MIB", "IT40", "MIB"],
        "AU200": ["ASX 200", "AU200", "XJO"],
        "EU50": ["Euro Stoxx 50", "EURO STOXX 50", "EU50", "STOXX 50", "SX5E"],
    }
    # Normalize keys
    for k, v in list(m.items()):
        m[k] = v
    # Try direct
    if s in m:
        return m[s]
    # Heuristic mapping like DE40/FR40 etc.
    if s.startswith("DE") and s.endswith("40"):
        return m["DE40"]
    if s.startswith("FR") and s.endswith("40"):
        return m["FR40"]
    if s.startswith("JP") and s.endswith("225"):
        return m["JP225"]
    if s.startswith("HK") and s.endswith("50"):
        return m["HK50"]
    if s.startswith("UK") and s.endswith("100"):
        return m["UK100"]
    if s.startswith("US") and s.endswith(("30", "500")):
        # US30 or US500
        return m["US30"] if s.endswith("30") else m["SPX500"]
    return [s]


# -------------------- HTTP helper --------------------
def _safe_request(url: str, params: dict = None, headers: dict = None, timeout: int = 8, retries: int = 2, backoff: float = 1.0) -> Optional[requests.Response]:
    """Simple request wrapper with retries and exponential backoff. Returns Response or None."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            return resp
        except Exception as e:
            print(f"news_engine: request error (attempt {attempt+1}/{retries+1}): {e}")
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
            else:
                return None
    return None

# -------------------- Relevance filter (drop sports/non-financial) --------------------
_FINANCE_WHITELIST = [
    # markets
    "stock", "stocks", "equity", "equities", "index", "indices", "futures", "etf",
    "market", "markets", "selloff", "rally", "volatile", "volatility", "bear", "bull",
    # macro
    "inflation", "cpi", "ppi", "pmi", "gdp", "retail sales", "unemployment", "jobs",
    "rate", "rates", "interest", "yields", "bond", "treasury", "fomc", "fed", "ecb", "boj", "boe",
    # fx/commodities
    "currency", "forex", "fx", "usd", "eur", "gbp", "jpy", "aud", "cad", "nzd", "cnh",
    "oil", "wti", "brent", "gold", "xau", "silver", "xag",
    # company/earnings
    "earnings", "guidance", "profit", "revenue", "eps",
    # index names
    "dow", "nasdaq", "s&p", "s & p", "sp500", "spx", "ftse", "dax", "cac", "nikkei", "hang seng", "hsi", "stoxx",
]

_SPORTS_BLACKLIST = [
    "athletics", "sprinter", "marathon", "olympic", "olympics", "world cup", "championship",
    "football", "soccer", "rugby", "cricket", "tennis", "golf", "f1", "formula 1", "motogp",
    "100m", "200m", "400m", "relay", "medal", "lap", "race", "goal", "matchday", "fixture",
]

_ENTERTAINMENT_BLACKLIST = [
    "celebrity", "hollywood", "bollywood", "music", "album", "movie", "film", "box office", "tv series",
]

def _is_finance_relevant(title: str, description: Optional[str] = None, source: Optional[str] = None) -> bool:
    """
    Basic content filter: require at least one finance whitelist term and reject obvious sports/entertainment.
    """
    t = (title or "").lower()
    d = (description or "").lower()

    if EXCLUDE_SPORTS:
        for bad in _SPORTS_BLACKLIST + _ENTERTAINMENT_BLACKLIST:
            if bad in t or bad in d:
                return False

    joined = f"{t} {d}".strip()
    return any(w in joined for w in _FINANCE_WHITELIST)

# -------------------- Simple sentiment --------------------
def analyze_sentiment(news_items: List[Dict[str, Any]]) -> str:
    """
    Lightweight sentiment estimation from news headlines.
    Kept compatible with older returned values ("Bullish"/"Bearish"/"Neutral").
    """
    positive_keywords = ["rise", "gain", "strong", "bullish", "growth", "positive", "support", "beats", "surge", "improves"]
    negative_keywords = ["fall", "drop", "weak", "bearish", "loss", "negative", "resistance", "misses", "plunge", "slumps"]

    score = 0
    for item in news_items:
        text = (item.get("headline") or "").lower()
        for w in positive_keywords:
            if w in text:
                score += 1
        for w in negative_keywords:
            if w in text:
                score -= 1

    if score > 0:
        return "Bullish"
    elif score < 0:
        return "Bearish"
    else:
        return "Neutral"

# -------------------- News providers --------------------
def _instrument_aliases(pair: str) -> List[str]:
    """
    Return search aliases for the instrument, especially for indices.
    """
    base = _strip_suffix(pair).upper().replace("_", "")
    if _is_index_symbol(base):
        return _index_aliases(base)
    # Crypto
    if base.startswith("ETH"):
        return ["Ethereum", "ETH", "crypto", base]
    if base.startswith("BTC"):
        return ["Bitcoin", "BTC", "crypto", base]
    # Metals
    if base.startswith("XAU"):
        return ["gold", "XAUUSD", "XAU", base]
    if base.startswith("XAG"):
        return ["silver", "XAGUSD", "XAG", base]
    # Forex currencies and raw base
    aliases = [base]
    if len(base) >= 6 and base[:3].isalpha() and base[3:6].isalpha():
        a, b = base[:3], base[3:6]
        currency_alias = {
            "USD": ["USD", "U.S. dollar", "US dollar", "Dollar"],
            "EUR": ["EUR", "Eurozone", "Euro"],
            "GBP": ["GBP", "British pound", "Sterling", "Pound"],
            "JPY": ["JPY", "Yen"],
            "CHF": ["CHF", "Swiss franc"],
            "AUD": ["AUD", "Australian dollar"],
            "NZD": ["NZD", "New Zealand dollar"],
            "CAD": ["CAD", "Canadian dollar", "Loonie"],
            "CNH": ["CNH", "Chinese yuan", "Renminbi", "CNY"],
        }
        aliases += currency_alias.get(a, [a]) + currency_alias.get(b, [b])
    return list(dict.fromkeys(aliases))  # dedupe, preserve order


def _build_query_for_pair(pair: str) -> str:
    """
    Build a richer query for NewsAPI/Google based on the pair/index, including synonyms and finance qualifiers.
    Also excludes obvious sports terms by adding minus-words where supported (Google News).
    """
    terms = _instrument_aliases(pair)
    # Finance qualifiers to improve relevance
    qualifiers = ["market", "index", "stocks", "futures", "equity", "CPI", "GDP", "rates", "yields"]
    query = " OR ".join(terms + qualifiers)

    # For Google News, we can add simple negative keywords to de-noise sports content
    negatives = ""
    if EXCLUDE_SPORTS:
        negatives = " -athletics -football -soccer -rugby -cricket -tennis -golf -Olympics -100m -200m -marathon"

    # Optional domain allowlist (works for NewsAPI via 'domains', for Google we'll append site filters inline)
    if NEWS_DOMAINS:
        # For Google, embed site: filters
        sites = [s.strip() for s in NEWS_DOMAINS.split(",") if s.strip()]
        if sites:
            site_clause = " OR ".join([f"site:{s}" for s in sites])
            query = f"({query}) ({site_clause}){negatives}"
        else:
            query = f"{query}{negatives}"
    else:
        query = f"{query}{negatives}"

    return query

# -------------------- Providers --------------------
def _fetch_newsapi(pair: str, limit: int = 5) -> Dict[str, Any]:
    """Try NewsAPI.org and return a structured dict."""
    raw_query = _build_query_for_pair(pair)
    params = {
        "q": raw_query,
        "language": "en",
        "pageSize": max(1, int(limit)),
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
        # Focus on titles to reduce noise
        "searchIn": "title",
    }
    if NEWS_DOMAINS:
        params["domains"] = NEWS_DOMAINS

    resp = _safe_request(NEWS_URL, params=params, timeout=8, retries=2)
    if resp is None:
        raise RuntimeError("NewsAPI: request failed")
    if resp.status_code != 200:
        try:
            body = resp.json()
            msg = body.get("message") or str(body)
        except Exception:
            msg = resp.text
        raise RuntimeError(f"NewsAPI returned {resp.status_code}: {msg}")

    data = resp.json()
    articles = data.get("articles", [])
    headlines = []
    for a in articles:
        title = (a.get("title") or a.get("description") or "").strip()
        desc = (a.get("description") or "").strip()
        src = (a.get("source") or {}).get("name", "NewsAPI")
        published = (a.get("publishedAt") or "")[:19].replace("T", " ")
        if not _is_finance_relevant(title, desc, src):
            continue
        headlines.append({
            "headline": title,
            "impact": "Medium",
            "source": src,
            "time": published,
            "raw": a
        })
    return {
        "pair": pair,
        "count": len(headlines),
        "sentiment_score": analyze_sentiment(headlines) if headlines else "Neutral",
        "headlines": headlines,
        "timestamp": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _fetch_forexnewsapi(pair: str, limit: int = 5) -> Dict[str, Any]:
    """Try a hypothetical ForexNewsAPI (keeps original shape)."""
    url = f"https://forexnewsapi.com/api/v1"
    params = {"pair": pair, "items": max(1, int(limit)), "token": FOREXNEWSAPI_KEY}
    resp = _safe_request(url, params=params, timeout=8, retries=2)
    if resp is None:
        raise RuntimeError("ForexNewsAPI: request failed")
    if resp.status_code != 200:
        raise RuntimeError(f"ForexNewsAPI returned {resp.status_code}")
    data = resp.json()
    items = data.get("news", []) if isinstance(data, dict) else []
    headlines = []
    for item in items:
        title = (item.get("title") or "").strip()
        desc = (item.get("summary") or item.get("description") or "").strip()
        if not _is_finance_relevant(title, desc, item.get("source", "ForexNewsAPI")):
            continue
        headlines.append({
            "headline": title,
            "impact": item.get("impact", "Medium"),
            "source": item.get("source", "ForexNewsAPI"),
            "time": item.get("published_at", ""),
            "raw": item
        })
    return {
        "pair": pair,
        "count": len(headlines),
        "sentiment_score": analyze_sentiment(headlines) if headlines else "Neutral",
        "headlines": headlines,
        "timestamp": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _fetch_google_news_rss(pair: str, limit: int = 8) -> Optional[Dict[str, Any]]:
    """Google News RSS fallback (no API key required) with index-aware, finance-filtered query."""
    query = _build_query_for_pair(pair)
    url = GOOGLE_NEWS_RSS.format(query=requests.utils.quote(query))
    resp = _safe_request(url, timeout=8, retries=2)
    if resp is None or resp.status_code != 200:
        return None
    try:
        root = ET.fromstring(resp.text)
        channel = root.find("channel")
        items = channel.findall("item") if channel is not None else root.findall("item")
        headlines = []
        for it in items[: max(1, int(limit))]:
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            pub = (it.findtext("pubDate") or "").strip()
            title = html.unescape(title)
            if not _is_finance_relevant(title, None, "GoogleNews"):
                continue
            headlines.append({
                "headline": title,
                "impact": "Medium",
                "source": "GoogleNews",
                "time": pub,
                "link": link
            })
        return {
            "pair": pair,
            "count": len(headlines),
            "sentiment_score": analyze_sentiment(headlines) if headlines else "Neutral",
            "headlines": headlines,
            "timestamp": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        print(f"news_engine: Google RSS parse error: {e}")
        return None

# -------------------- Public news fetch (backwards compatible) --------------------
def get_recent_news(pair: str, limit: int = 5) -> Dict[str, Any]:
    """
    Public function to fetch recent news for a pair or index.
    Returns a dict with keys:
      - pair, count, sentiment_score, headlines, timestamp
    Adds:
      - status: "ok" / "no_news" / "api_error"
      - errors: list of diagnostic strings (may be empty)
    Filters out non-financial/sports items (e.g., 'UK 100m' athletics results) to keep headlines relevant.
    """
    errors: List[str] = []
    # 1) Try ForexNewsAPI if key present (experimental)
    if FOREXNEWSAPI_KEY:
        try:
            res = _fetch_forexnewsapi(pair, limit)
            res["status"] = "ok" if res.get("count", 0) > 0 else "no_news"
            res["errors"] = errors
            return res
        except Exception as e:
            errors.append(f"ForexNewsAPI_error: {e}")

    # 2) Try NewsAPI if key present
    if NEWSAPI_KEY:
        try:
            res = _fetch_newsapi(pair, limit)
            res["status"] = "ok" if res.get("count", 0) > 0 else "no_news"
            res["errors"] = errors
            return res
        except Exception as e:
            errors.append(f"NewsAPI_error: {e}")

    # 3) Fallback to Google News RSS
    try:
        rss = _fetch_google_news_rss(pair, limit=limit if limit else 8)
        if rss:
            rss["status"] = "ok" if rss.get("count", 0) > 0 else "no_news"
            rss["errors"] = errors
            return rss
    except Exception as e:
        errors.append(f"GoogleRSS_error: {e}")

    # 4) Final fallback: explicit informative message
    now_ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "pair": pair,
        "count": 0,
        "sentiment_score": "Neutral",
        "headlines": [
            {
                "headline": "No relevant news found or all news sources failed.",
                "impact": "Low",
                "source": "System",
                "time": now_ts,
            }
        ],
        "timestamp": now_ts,
        "status": "api_error",
        "errors": errors
    }

# -------------------- Economic calendar (CSV) + Blackout --------------------
def _load_calendar_csv(path: str) -> List[Dict[str, Any]]:
    """
    Load calendar CSV into normalized events:
    Each event:
      - time (UTC aware datetime)
      - local_time (string for display, same format as timestamp)
      - impact: High/Medium/Low
      - currencies: list[str]
      - title: str
      - source: "csv"
      - id: stable string
    CSV expected columns (case-insensitive, any alias):
      - time | timestamp | date
      - impact | importance
      - currency | currencies | symbol
      - title | event | headline
    """
    events: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return events

    def _col(df_cols: List[str], candidates: List[str]) -> Optional[str]:
        lower = {c.lower(): c for c in df_cols}
        for cand in candidates:
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            t_col = _col(cols, ["time", "timestamp", "date", "datetime"])
            i_col = _col(cols, ["impact", "importance", "priority"])
            c_col = _col(cols, ["currency", "currencies", "symbol", "country"])
            h_col = _col(cols, ["title", "event", "headline"])
            for row in reader:
                ts_raw = row.get(t_col) if t_col else None
                ts = _to_utc(ts_raw) if ts_raw else None
                if ts is None:
                    continue
                impact = str(row.get(i_col, "Medium") if i_col else "Medium").capitalize()
                if impact not in ("High", "Medium", "Low"):
                    impact = "Medium"
                cur_raw = row.get(c_col, "") if c_col else ""
                currencies = [c.strip().upper() for c in str(cur_raw).replace("/", ",").replace(";", ",").split(",") if c.strip()]
                title = str(row.get(h_col, "") if h_col else "").strip()
                eid = f"{_fmt_utc(ts)}|{impact}|{','.join(currencies)}|{title}"
                events.append({
                    "time": ts,
                    "local_time": _fmt_utc(ts),
                    "impact": impact,
                    "currencies": currencies,
                    "title": title,
                    "source": "csv",
                    "id": eid
                })
    except Exception as e:
        print(f"news_engine: error reading calendar CSV: {e}")
        return []
    # Sort and dedupe
    seen = set()
    out = []
    for e in sorted(events, key=lambda x: x["time"]):
        if e["id"] in seen:
            continue
        seen.add(e["id"])
        out.append(e)
    return out


def _calendar_events() -> List[Dict[str, Any]]:
    """
    Return cached calendar events, reloading as needed based on TTL or mtime change.
    """
    path = NEWS_CSV_PATH
    if not path:
        return []
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        return []
    now = time.time()
    if (_CAL_CACHE["path"] != path
            or _CAL_CACHE["mtime"] != mtime
            or (now - float(_CAL_CACHE["loaded_at"])) > float(_CAL_CACHE["ttl"])):
        events = _load_calendar_csv(path)
        _CAL_CACHE.update({
            "path": path,
            "mtime": mtime,
            "loaded_at": now,
            "events": events,
        })
    return _CAL_CACHE.get("events", [])


def get_upcoming_events(window_min: int = NEWS_BLACKOUT_MIN, currencies: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Return events in [now - window_min, now + window_min] filtered by currencies (if provided).
    """
    evs = _calendar_events()
    if not evs:
        return []
    now = _now_utc()
    lo = now - dt.timedelta(minutes=int(window_min))
    hi = now + dt.timedelta(minutes=int(window_min))
    cur_set = set([c.upper() for c in (currencies or [])])
    out: List[Dict[str, Any]] = []
    for e in evs:
        t = e.get("time")
        if t is None:
            continue
        if not (lo <= t <= hi):
            continue
        if cur_set:
            if not any(c.upper() in cur_set for c in e.get("currencies", [])):
                continue
        out.append(e)
    return out


def is_blackout(now: Optional[dt.datetime] = None,
                pair: Optional[str] = None,
                currencies: Optional[List[str]] = None,
                window_min: int = NEWS_BLACKOUT_MIN) -> bool:
    """
    True if a HIGH-impact event for the pair's currencies is within +/- window_min minutes of 'now'.
    If 'currencies' is provided, 'pair' is ignored for filtering.
    """
    if currencies is None:
        currencies = _extract_currencies(pair or "")
    base_now = now or _now_utc()
    if base_now.tzinfo is None:
        base_now = base_now.replace(tzinfo=dt.timezone.utc)
    lo = base_now - dt.timedelta(minutes=int(window_min))
    hi = base_now + dt.timedelta(minutes=int(window_min))

    evs = _calendar_events()
    if not evs:
        return False

    cur_set = set([c.upper() for c in (currencies or [])])
    for e in evs:
        t = e.get("time")
        if t is None:
            continue
        if not (lo <= t <= hi):
            continue
        if e.get("impact") != "High":
            continue
        if cur_set:
            if not any(c.upper() in cur_set for c in e.get("currencies", [])):
                continue
        return True
    return False


def blackout_penalty(now: Optional[dt.datetime] = None,
                     pair: Optional[str] = None,
                     currencies: Optional[List[str]] = None,
                     window_min: int = NEWS_BLACKOUT_MIN) -> float:
    """
    Return a numeric penalty in [0.0, 0.2]. 0.2 if inside the blackout window, else 0.0.
    """
    return 0.2 if is_blackout(now=now, pair=pair, currencies=currencies, window_min=window_min) else 0.0

# -------------------- Compatibility helper for context engines --------------------
def get_news_events() -> List[Dict[str, Any]]:
    """
    Return a list of events compatible with context_engine/contextual_analysis expectations:
      [{ "time": ISO string, "impact": "High"|"Medium"|"Low", ... }, ...]

    This pulls from the CSV calendar if configured. If none, returns [].
    """
    evs = _calendar_events()
    out: List[Dict[str, Any]] = []
    for e in evs:
        t = e.get("time")
        out.append({
            "time": _fmt_utc(t) if isinstance(t, dt.datetime) else str(t),
            "impact": e.get("impact", "Medium"),
            "currencies": e.get("currencies", []),
            "title": e.get("title", ""),
            "source": e.get("source", "csv"),
            "id": e.get("id"),
        })
    return out
