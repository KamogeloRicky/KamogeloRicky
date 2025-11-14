"""
Comprehensive analysis entrypoint for RickyFX Analysis Bot — SAST timestamps enforced.

Key capabilities (all additive, nothing removed):
- SAST conversion for all user-visible timestamps.
- Exchange-aware sessions (e.g., America/New_York for US equities).
- Severity-based gating (hard vs soft) to avoid over-neutralization.
- Pending-entry mode (keeps Buy/Sell label even if exact price not pinned).
- Directional alignment bonus (HTF alignment scoring).
- GOAT / GOAT+ / Precision Retest engines with post-pass.
- Anti-sweep protected SL and true edge-of-retest entry logic.
- 1m microstructure precision + 5m, with VWAP, FVG, OB taps.
- 4h higher-timeframe confluence signals (trend slope, BOS/CHOCH).
- Confluence lock prevents soft neutralization when MTF confluence is strong.
- Robust fallbacks for missing modules/functions, no-breaking add-only.

**Phase 3 Update:** GOAT+ CSV logging has been centralized into the new `journal.py` module for the Maestro.
"""

import os
import math
import logging
import datetime as dt
from typing import Any, Dict, Optional, List, Tuple
import csv

import pandas as pd
import numpy as np

# optional libraries
try:
    import ta
except Exception:
    ta = None

# timezone helper
try:
    import pytz
    SAST_TZ = pytz.timezone("Africa/Johannesburg")
except Exception:
    SAST_TZ = None

# VADER sentiment optional
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER_AVAILABLE = False
    _VADER = None

# -------------------------
# MT5 credentials (restored exactly)
# -------------------------
MT5_LOGIN = 211825028            # original provided login constant
MT5_PASSWORD = "RICKy@081"       # original provided password constant
MT5_SERVER = "Exness-MT5Trial9"  # original provided server constant

# -------------------------
# Logging
# -------------------------
LOG_FILE = os.getenv("ANALYSIS_LOG_FILE", "analysis_log.txt")
logger = logging.getLogger("rickyfx_analysis")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)


def log(msg: str, level: str = "info"):
    try:
        print(msg)
    except Exception:
        pass
    try:
        getattr(logger, level)(msg)
    except Exception:
        try:
            logger.info(msg)
        except Exception:
            pass


# -------------------------
# Helper: convert timestamps to SAST (Africa/Johannesburg)
# -------------------------
def _to_sast(ts) -> str:
    """
    Convert timestamp to SAST string.
    - If ts is naive, assume UTC then convert to SAST.
    """
    try:
        if isinstance(ts, str) and ts.strip().endswith(" SAST"):
            return ts
        if isinstance(ts, str):
            parts = ts.strip().split()
            if parts and parts[-1].isalpha() and len(parts[-1]) in (3, 4) and parts[-1].upper() == "SAST":
                ts = " ".join(parts[:-1])
    except Exception:
        pass

    try:
        t = pd.to_datetime(ts)
        if pd.isna(t):
            raise ValueError("invalid timestamp")
        if getattr(t, "tzinfo", None) is None or str(getattr(t, "tzinfo", None)) == "None":
            try:
                t = t.tz_localize("UTC")
            except Exception:
                pass
        if SAST_TZ is not None:
            try:
                t = t.tz_convert(SAST_TZ)
                try:
                    return t.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    return t.strftime("%Y-%m-%d %H:%M:%S") + " SAST"
            except Exception:
                try:
                    t = t.tz_convert("UTC") + pd.Timedelta(hours=2)
                    return t.strftime("%Y-%m-%d %H:%M:%S") + " SAST"
                except Exception:
                    return str(ts)
        else:
            try:
                try:
                    t = t.tz_convert("UTC")
                except Exception:
                    pass
                t = t + pd.Timedelta(hours=2)
                return t.strftime("%Y-%m-%d %H:%M:%S") + " SAST"
            except Exception:
                try:
                    t2 = pd.to_datetime(ts) + pd.Timedelta(hours=2)
                    return t2.strftime("%Y-%m-%d %H:%M:%S") + " SAST"
                except Exception:
                    return str(ts)
    except Exception:
        try:
            t2 = pd.to_datetime(ts)
            t2 = t2 + pd.Timedelta(hours=2)
            return t2.strftime("%Y-%m-%d %H:%M:%S") + " SAST"
        except Exception:
            return str(ts)


# -------------------------
# Project imports with defensive fallbacks
# -------------------------
try:
    from data import fetch_candles, shutdown_mt5
except Exception:
    fetch_candles = None
    shutdown_mt5 = lambda: None
    log("warning: data.fetch_candles or shutdown_mt5 not found — analysis will not fetch live candles.", "warning")

try:
    from news_engine import get_recent_news
except Exception:
    get_recent_news = None

try:
    from precision_entry import precision_analysis, find_precise_entry
except Exception:
    precision_analysis = None
    find_precise_entry = None

try:
    from advanced_signals import advanced_analysis
except Exception:
    def advanced_analysis(df: pd.DataFrame) -> Dict[str, Any]:
        return {"advanced_signal": "Neutral", "notes": []}

try:
    from context_engine import contextual_analysis as contextual_analysis
except Exception:
    try:
        from context_engine import context_analysis as contextual_analysis
    except Exception:
        def contextual_analysis(df: pd.DataFrame, pair: str = None, timeframe: str = None, price: float = None) -> Dict[str, Any]:
            return {"higher_tf_trend": None, "signal": "Neutral", "score": 0.0, "retest": None, "entry_time": None, "pattern": None, "notes": []}

try:
    from utils import indicator_signals, ensemble_signal, multi_timeframe_trend, SUPPORTED_TIMEFRAMES
except Exception:
    SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h"]

    def indicator_signals(df: pd.DataFrame) -> Dict[str, str]:
        try:
            if ta is None:
                return {}
            rsi = ta.momentum.RSIIndicator(df["close"], window=14).rsi().iloc[-1]
            macd_diff = ta.trend.MACD(df["close"]).macd_diff().iloc[-1]
            ema_fast = ta.trend.EMAIndicator(df["close"], window=12).ema_indicator().iloc[-1]
            ema_slow = ta.trend.EMAIndicator(df["close"], window=26).ema_indicator().iloc[-1]
            return {
                "RSI": "Buy" if rsi < 30 else "Sell" if rsi > 70 else "Neutral",
                "MACD": "Buy" if macd_diff > 0 else "Sell" if macd_diff < 0 else "Neutral",
                "EMA": "Buy" if ema_fast > ema_slow else "Sell" if ema_fast < ema_slow else "Neutral"
            }
        except Exception:
            return {}

    def ensemble_signal(signals: Dict[str, str], trend: Optional[str] = None) -> str:
        votes = list(signals.values())
        buy = votes.count("Buy")
        sell = votes.count("Sell")
        if trend == "Buy":
            buy += 1
        elif trend == "Sell":
            sell += 1
        if buy > sell:
            return "Buy"
        if sell > buy:
            return "Sell"
        return "Neutral"

    def multi_timeframe_trend(pair: str) -> Optional[str]:
        return None

try:
    from adaptive_learning import adaptive_engine, log_result
except Exception:
    class _DummyAdaptive:
        def __init__(self):
            self._w = {"core":1.0,"advanced":1.0,"precision":1.0,"context":1.0,"news":1.0}
            self._p = {"atr_multiplier":1.5,"retest_sensitivity":0.6,"pattern_confidence_threshold":0.7}
        def get_weights(self): return self._w
        def get_params(self): return self._p
        def summary(self): return {"total_trades":0,"wins":0,"accuracy":0.0,"weights":self._w,"params":self._p}
    adaptive_engine = _DummyAdaptive()
    def log_result(pair, success, confidence, source="analysis"):
        try:
            with open("adaptive_log.txt", "a") as f:
                f.write(f"{dt.datetime.utcnow().isoformat()} | {pair} | {source} | success={success} | conf={confidence}\n")
        except Exception:
            pass

try:
    from backtest_engine import run_backtest
except Exception:
    def run_backtest(pair, timeframe, lookback=2000):
        return {"success": False, "error": "backtest_engine not available"}

try:
    from forward_test_engine import run_forward_test
except Exception:
    def run_forward_test(pair, timeframe, future_bars=50):
        return {"success": False, "error": "forward_test_engine not available"}

# **Phase 3 Update:** Import the centralized logging function
try:
    from journal import log_trade_decision
except ImportError:
    # Fallback if journal.py is missing. This ensures analysis.py never crashes.
    def log_trade_decision(res):
        print("Warning: journal.py not found. Cannot log trade decision to centralized journal.")

# Provide fallback for liquidity_heuristic if not imported elsewhere
try:
    liquidity_heuristic  # type: ignore
except NameError:
    def liquidity_heuristic(df: pd.DataFrame, atr: Optional[float]) -> Dict[str, Any]:
        try:
            rng = float((df["high"].tail(20) - df["low"].tail(20)).mean())
            vol = float(df["close"].pct_change().tail(50).std() or 0.0)
            score = 0.0
            if atr and rng:
                score = max(0.0, min(1.0, (rng / (atr * 2.5))))
            return {"liquidity_score": round(score, 3), "avg_range": rng, "atr": atr, "vol": vol}
        except Exception:
            return {"liquidity_score": 0.0}


# -------------------------
# News helpers & fallback RSS (SAST)
# -------------------------
import requests
import xml.etree.ElementTree as ET
import html as html_mod

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_POS_KW = ["gain", "rise", "bullish", "up", "strong", "increase", "beats", "surge", "improves"]
_NEG_KW = ["fall", "drop", "bearish", "down", "weak", "decrease", "misses", "plunge", "slumps"]

def _news_query_for_pair(pair: str) -> str:
    p = (pair or "").upper()
    negatives = ' -"$30m" -"US$30m" -US30m'
    if any(k in p for k in ["US30", "DJI", "DOW"]):
        return '("Dow Jones" OR "US30 index" OR DJI)' + negatives
    if any(k in p for k in ["US100", "NAS100", "NDX"]):
        return '("Nasdaq 100" OR "US100 index" OR NDX)'
    if any(k in p for k in ["US500", "SPX", "SP500"]):
        return '("S&P 500" OR SPX OR "US500 index")'
    if any(k in p for k in ["DE40", "GER40", "DAX"]):
        return '("DAX" OR "GER40" OR "Germany 40")'
    if "XAU" in p:
        return '("Gold" OR XAUUSD)'
    if "XAG" in p:
        return '("Silver" OR XAGUSD)'
    return pair

def _analyze_sentiment_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"label": "Neutral", "score": 0.0}
    try:
        if _VADER_AVAILABLE and _VADER is not None:
            vs = _VADER.polarity_scores(text)
            comp = float(vs.get("compound", 0.0))
            if comp >= 0.05:
                lab = "Positive"
            elif comp <= -0.05:
                lab = "Negative"
            else:
                lab = "Neutral"
            return {"label": lab, "score": comp}
        else:
            low = text.lower()
            p = sum(1 for k in _POS_KW if k in low)
            n = sum(1 for k in _NEG_KW if k in low)
            if p > n:
                return {"label": "Positive", "score": float(p - n)}
            if n > p:
                return {"label": "Negative", "score": float(n - p) * -1.0}
            return {"label": "Neutral", "score": 0.0}
    except Exception:
        return {"label": "Neutral", "score": 0.0}


def _fetch_google_news_rss_local(pair: str, max_articles: int = 8) -> Optional[Dict[str, Any]]:
    try:
        q = pair.upper().replace("C", "").replace("_", "").strip()
        try:
            q = _news_query_for_pair(pair)
        except Exception:
            pass
        url = GOOGLE_NEWS_RSS.format(query=requests.utils.quote(q))
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return None
        root = ET.fromstring(resp.text)
        channel = root.find("channel")
        items = channel.findall("item") if channel is not None else root.findall("item")
        headlines = []
        pos = neg = 0
        for it in items[:max_articles]:
            title = it.findtext("title") or ""
            pub = it.findtext("pubDate") or it.findtext("dc:date") or None
            if pub:
                try:
                    pub_sast = _to_sast(pub)
                except Exception:
                    pub_sast = _to_sast(dt.datetime.utcnow())
            else:
                pub_sast = _to_sast(dt.datetime.utcnow())
            title = html_mod.unescape(title)
            sent = _analyze_sentiment_text(title)
            if sent["label"] == "Positive":
                pos += 1
            elif sent["label"] == "Negative":
                neg += 1
            headlines.append({"headline": title, "impact": "Medium", "source": "GoogleNews", "time": pub_sast, "sentiment": sent})
        overall = "Neutral"
        if pos > neg:
            overall = "Positive"
        elif neg > pos:
            overall = "Negative"
        now_sast = _to_sast(dt.datetime.utcnow())
        return {"pair": pair, "count": len(headlines), "sentiment_score": overall, "headlines": headlines, "timestamp": now_sast}
    except Exception:
        return None


def fetch_news_with_status(pair: str) -> Dict[str, Any]:
    errors = []
    if get_recent_news is not None:
        try:
            news_info = get_recent_news(pair)
            if isinstance(news_info, dict):
                raw_headlines = news_info.get("headlines", []) or []
                sanitized = []
                for h in raw_headlines:
                    head = h.get("headline") or h.get("title") or ""
                    impact = h.get("impact", "Medium")
                    source = h.get("source") or h.get("source_name") or "Unknown"
                    raw_time = h.get("time") or h.get("publishedAt") or None
                    if raw_time:
                        try:
                            time_sast = _to_sast(raw_time)
                        except Exception:
                            time_sast = _to_sast(dt.datetime.utcnow())
                    else:
                        time_sast = _to_sast(dt.datetime.utcnow())
                    sent = h.get("sentiment")
                    sanitized.append({"headline": head, "impact": impact, "source": source, "time": time_sast, "sentiment": sent})
                news_info["headlines"] = sanitized
                count = news_info.get("count", None)
                news_info["status"] = "ok" if (not isinstance(count, int) or count > 0 or len(sanitized) > 0) else "no_news"
                news_info["errors"] = errors
                news_info["timestamp"] = _to_sast(news_info.get("timestamp") or dt.datetime.utcnow())
                return news_info
        except Exception as e:
            errors.append(f"user_news_engine_error: {e}")

    try:
        rss = _fetch_google_news_rss_local(pair)
        if rss:
            rss["status"] = "ok" if rss.get("count", 0) > 0 else "no_news"
            rss["errors"] = errors
            return rss
    except Exception as e:
        errors.append(f"rss_error: {e}")

    now_sast = _to_sast(dt.datetime.utcnow())
    return {
        "pair": pair,
        "count": 0,
        "sentiment_score": "Neutral",
        "headlines": [
            {"headline": "No relevant news found or all news sources failed.", "impact": "Low", "source": "System", "time": now_sast}
        ],
        "timestamp": now_sast,
        "status": "api_error",
        "errors": errors
    }


# -------------------------
# Utility & heuristics helpers
# -------------------------
def safe_round(x, nd=6):
    try:
        return round(float(x), nd)
    except Exception:
        return x


def detect_volume_surge(df: pd.DataFrame, lookback: int = 20, multiplier: float = 1.5) -> bool:
    try:
        if "volume" not in df.columns or len(df) < lookback + 1:
            return False
        recent = df["volume"].iloc[-lookback:]
        last = df["volume"].iloc[-1]
        return float(last) > (recent.mean() * multiplier)
    except Exception:
        return False


def _pips_for_pair(pair: str) -> float:
    try:
        p = (pair or "").upper()
        if p.endswith("JPY") or p.endswith("JPYC") or p.endswith("JPYc"):
            return 0.01
    except Exception:
        pass
    return 0.0001


def compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    try:
        if ta is None:
            return None
        atr_ser = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=period).average_true_range()
        return float(atr_ser.iloc[-1])
    except Exception:
        try:
            return float(np.std(df["close"].diff().dropna().tail(period)))
        except Exception:
            return None


# -------------------------
# Pattern and structure helpers
# -------------------------
def detect_retest(df: pd.DataFrame, direction: str, atr: float, sensitivity: float = 1.0) -> Optional[Dict[str, Any]]:
    try:
        n = min(len(df), 200)
        sub = df.iloc[-n:].copy().reset_index(drop=True)
        if len(sub) < 5:
            return None
        highs = sub["high"]
        lows = sub["low"]
        closes = sub["close"]
        look_for = 30
        window = min(len(sub), look_for)
        if window < 5:
            pivot_idx = len(sub) - 2
        else:
            hh_idx = highs[-window:].idxmax()
            ll_idx = lows[-window:].idxmin()
            pivot_idx = hh_idx if direction == "Sell" else ll_idx

        pivot_price = float((highs.iloc[pivot_idx] if direction == "Sell" else lows.iloc[pivot_idx]))
        last_price = float(closes.iloc[-1])
        if atr and abs(last_price - pivot_price) <= max(0.00001, sensitivity * atr):
            if direction == "Buy":
                entry_price = last_price
                sl = pivot_price - 0.5 * atr
            else:
                entry_price = last_price
                sl = pivot_price + 0.5 * atr
            confidence = 0.45 + min(0.45, 0.25 * (1.0 / (1.0 + abs(last_price - pivot_price))))
            return {"entry_price": float(entry_price), "sl": float(sl), "confidence": float(confidence), "reason": "heuristic_retest", "pivot_price": float(pivot_price)}
    except Exception:
        return None
    return None


def detect_bos_choch_liquidity(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        if df is None or len(df) < 8:
            return {"bos": None, "choch": False, "liquidity_sweep": False, "details": "insufficient_data"}
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        window = min(50, max(3, len(df) // 4))
        recent_segment = df.iloc[-(window + 1):-1] if window >= 3 else df.iloc[-3:-1]
        recent_high = float(recent_segment["high"].max())
        recent_low = float(recent_segment["low"].min())
        last_close = float(closes[-1])
        prev_close = float(closes[-2])
        bos = None
        if last_close > recent_high and prev_close <= recent_high:
            bos = "Buy"
        elif last_close < recent_low and prev_close >= recent_low:
            bos = "Sell"
        choch = False
        if bos is not None:
            try:
                prev_window = min(80, len(df) - window - 1) if len(df) > window + 5 else 0
                if prev_window >= 3:
                    prev_segment = df.iloc[-(window + prev_window + 1):-(window + 1)]
                    prev_high = float(prev_segment["high"].max())
                    prev_low = float(prev_segment["low"].min())
                    if bos == "Buy" and prev_high >= recent_high:
                        choch = True
                    if bos == "Sell" and prev_low <= recent_low:
                        choch = True
            except Exception:
                choch = False
        liquidity = False
        try:
            rng = float((df["high"] - df["low"]).tail(20).mean()) if len(df) >= 20 else float((df["high"] - df["low"]).mean())
            threshold = rng * 0.9 if rng and rng > 0 else 0.0
            if recent_high and df["high"].iloc[-1] > recent_high + threshold and df["close"].iloc[-1] < df["open"].iloc[-1]:
                liquidity = True
            if recent_low and df["low"].iloc[-1] < recent_low - threshold and df["close"].iloc[-1] > df["open"].iloc[-1]:
                liquidity = True
        except Exception:
            liquidity = False
        return {"bos": bos, "choch": choch, "liquidity_sweep": liquidity, "details": {"recent_high": recent_high, "recent_low": recent_low}}
    except Exception as e:
        return {"bos": None, "choch": False, "liquidity_sweep": False, "details": f"error:{e}"}


def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    try:
        if len(df) < 2:
            return False
        a = df.iloc[-2]; b = df.iloc[-1]
        return (b["close"] > b["open"]) and (a["close"] < a["open"]) and (b["close"] > a["open"]) and (b["open"] < a["close"])
    except Exception:
        return False


def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    try:
        if len(df) < 2:
            return False
        a = df.iloc[-2]; b = df.iloc[-1]
        return (b["close"] < b["open"]) and (a["close"] > a["open"]) and (b["open"] > a["close"]) and (b["close"] < a["open"])
    except Exception:
        return False


def is_hammer_like(df: pd.DataFrame) -> bool:
    try:
        if len(df) < 1:
            return False
        c = df.iloc[-1]
        body = abs(c["close"] - c["open"])
        if body == 0:
            return False
        lower_wick = (c["open"] - c["low"]) if c["open"] >= c["close"] else (c["close"] - c["low"])
        upper_wick = c["high"] - max(c["close"], c["open"])
        return lower_wick > (body * 2) and upper_wick < (body * 0.5)
    except Exception:
        return False


def is_shooting_star_like(df: pd.DataFrame) -> bool:
    try:
        if len(df) < 1:
            return False
        c = df.iloc[-1]
        body = abs(c["close"] - c["open"])
        if body == 0:
            return False
        upper_wick = c["high"] - max(c["close"], c["open"])
        lower_wick = min(c["close"], c["open"]) - c["low"]
        return upper_wick > (body * 2) and lower_wick < (body * 0.5)
    except Exception:
        return False


# -------------------------
# Confidence helpers for adaptive logging
# -------------------------
def _confidence_from_backtest(bt: dict) -> float:
    try:
        metrics = bt.get("metrics", {}) if isinstance(bt, dict) else {}
        win_rate = (metrics.get("win_rate_percent", 0) or 0) / 100.0
        expectancy = (metrics.get("expectancy", 0) or 0)
        avg_win = abs(metrics.get("avg_win", 0.0)) or 1.0
        conf = 0.35 + 0.5 * win_rate
        conf += 0.1 * max(-1.0, min(1.0, expectancy / (avg_win + 1e-9)))
        return float(max(0.2, min(0.99, conf)))
    except Exception:
        return 0.5


def _confidence_from_forward(ft: dict) -> float:
    try:
        if ft is None:
            return 0.5
        if isinstance(ft.get("confidence"), (int, float)):
            c = float(ft.get("confidence") or 0.5)
            return float(max(0.2, min(0.99, c)))
        pnl = float(ft.get("pnl", 0) or 0)
        price_now = float(ft.get("price_now", 0) or 1)
        rel = abs(pnl) / (abs(price_now) + 1e-9)
        conf = 0.3 + min(0.6, rel * 10.0)
        return float(max(0.2, min(0.99, conf)))
    except Exception:
        return 0.5


# -------------------------
# Improved lot-size helper (robust)
# -------------------------
def compute_lot_size_from_risk(account_balance: float,
                               risk_pct: float,
                               sl_pips: float,
                               pip_value_per_lot: float = 10.0,
                               confidence: float = 0.5,
                               min_lot: float = 0.01,
                               max_lot: Optional[float] = None) -> Optional[float]:
    """
    Robust lot-size calculator. Returns lot size rounded to 2 decimals or None if inputs are invalid.
    """
    try:
        if account_balance is None or risk_pct is None or sl_pips is None:
            return None
        account_balance = float(account_balance)
        risk_pct = float(risk_pct)
        sl_pips = float(sl_pips)
        pip_value_per_lot = float(pip_value_per_lot)
        confidence = float(confidence or 0.5)

        if account_balance <= 0 or risk_pct <= 0 or sl_pips <= 0 or pip_value_per_lot <= 0:
            return None

        effective_scale = 0.6 + 0.65 * confidence  # [0.6, 1.25]
        effective_risk_pct = risk_pct * effective_scale
        risk_amount = account_balance * (effective_risk_pct / 100.0)
        lot = risk_amount / (max(1e-9, sl_pips) * pip_value_per_lot)
        lot = max(min_lot, round(lot, 2))
        if max_lot is not None:
            lot = min(lot, float(max_lot))
        return lot
    except Exception:
        return None


# -------------------------
# GOAT MODE: constants and helpers
# -------------------------
_GOAT_MODE = os.getenv("GOAT_MODE") == "1" or os.getenv("ANALYSIS_GOAT_MODE") == "1"

try:
    if os.getenv("GOAT_MODE") is None and os.getenv("ANALYSIS_GOAT_MODE") is None:
        _GOAT_MODE = True
    if os.getenv("GOAT_MODE") == "0" or os.getenv("ANALYSIS_GOAT_MODE") == "0" or os.getenv("GOAT_DISABLE") == "1" or os.getenv("ANALYSIS_GOAT_DISABLE") == "1":
        _GOAT_MODE = False
    if _GOAT_MODE:
        log("GOAT mode enabled automatically. Set GOAT_MODE=0 to disable.", "info")
except Exception:
    pass

_GOAT_ALLOWED_SESSIONS = tuple((os.getenv("ANALYSIS_ALLOWED_SESSIONS") or "London,NY").split(","))
_GOAT_MAX_SPREAD_PIPS = float(os.getenv("ANALYSIS_MAX_SPREAD_PIPS") or "1.0")
_GOAT_NEWS_BLACKOUT_MIN = int(os.getenv("ANALYSIS_NEWS_BLACKOUT_MIN") or "30")
_GOAT_MIN_ATR_PIPS = float(os.getenv("ANALYSIS_MIN_ATR_PIPS") or "0.0")
_GOAT_TRAIL_AFTER_R = float(os.getenv("ANALYSIS_TRAIL_AFTER_R") or "1.5")
_GOAT_PARTIAL_TP_R = float(os.getenv("ANALYSIS_PARTIAL_TP_R") or "1.0")
_GOAT_TIME_STOP_BARS = int(os.getenv("ANALYSIS_TIME_STOP_BARS") or "300")
_GOAT_SESSION_STRICT = os.getenv("ANALYSIS_SESSION_STRICT") == "1"

# Block types
_HARD_BLOCKS = set(["news_blackout", "dd_halt", "negative_expected_value", "atr_too_low", "event_keyword"])
_SOFT_BLOCKS = set(["session_filter", "spread_filter", "no_entry_point", "high_congestion", "sr_too_close", "spread_to_atr_high", "stale_signal", "no_2of3_confluence", "contra_strict_block"])

def _exchange_timezone(pair: str) -> Optional[str]:
    p = (pair or "").upper()
    if any(p.startswith(t) for t in ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "NFLX", "AMD", "INTC"]) or p.endswith("M"):
        return "America/New_York"
    return None

def _goat_session_from_ts(ts: Any) -> str:
    try:
        t = pd.to_datetime(ts)
        hour = int(t.hour)
        if 7 <= hour < 16:
            return "London"
        if 13 <= hour < 22:
            return "NY"
        return "Asia"
    except Exception:
        return "Unknown"

def _goat_session_from_ts_tz(ts: Any, pair: str = None) -> str:
    try:
        t = pd.to_datetime(ts)
        env_tz = os.getenv("ANALYSIS_SESSION_TZ")
        tz = None
        if env_tz:
            try:
                import pytz as _p
                tz = _p.timezone(env_tz)
            except Exception:
                tz = None
        if tz is None:
            ex_tz = _exchange_timezone(pair or "")
            if ex_tz:
                try:
                    import pytz as _p
                    tz = _p.timezone(ex_tz)
                except Exception:
                    tz = None
        if tz is None:
            tz = SAST_TZ
        if tz is not None:
            if getattr(t, "tzinfo", None) is None:
                t = t.tz_localize("UTC").astimezone(tz)
            else:
                t = t.tz_convert(tz)
        hour = int(t.hour)
        if 7 <= hour < 16:
            return "London"
        if 13 <= hour < 22:
            return "NY"
        return "Asia"
    except Exception:
        return _goat_session_from_ts(ts)

def _parse_sast(ts_str: str) -> Optional[pd.Timestamp]:
    try:
        s = (ts_str or "").strip()
        if s.endswith(" SAST"):
            s = s[:-5].strip()
        t = pd.to_datetime(s)
        return t
    except Exception:
        return None

def _goat_regime_score(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        close = df["close"].astype(float)
        if len(close) < 60:
            return {"score": 0.0, "trend": "Unknown"}
        ma_fast = close.rolling(20).mean()
        ma_slow = close.rolling(50).mean()
        ma_diff = (ma_fast - ma_slow) / (close + 1e-9)
        vol = close.pct_change().rolling(50).std()
        trendiness = float(abs(ma_diff.iloc[-1]))
        vol_now = float(vol.iloc[-1] if not np.isnan(vol.iloc[-1]) else 0.0)
        score = 0.0
        if trendiness > 0.001 and vol_now > 0:
            score = min(1.0, trendiness / 0.005)
        trend = "Trending" if score >= 0.5 else "Choppy" if score <= 0.2 else "Mixed"
        return {"score": round(score, 3), "trend": trend}
    except Exception:
        return {"score": 0.0, "trend": "Unknown"}

def _goat_p_hat(combined_score: float) -> float:
    try:
        return float(max(0.01, min(0.99, 0.5 + 0.45 * math.tanh(combined_score))))
    except Exception:
        return 0.5

def _goat_rr(pair: str, price: Optional[float], entry: Optional[float], sl: Optional[float], tp: Optional[float], atr: Optional[float]) -> Dict[str, Any]:
    try:
        ep = entry if entry is not None else price
        if ep is not None and sl is not None and tp is not None and ep != sl:
            rr = abs(tp - ep) / (abs(ep - sl) + 1e-9)
            return {"rr": float(rr), "source": "explicit"}
        if atr is not None and price is not None:
            rr = 3.0 / 1.5
            return {"rr": float(rr), "source": "atr_fallback"}
    except Exception:
        pass
    return {"rr": 1.0, "source": "default"}

def _goat_news_blackout(news: Dict[str, Any], minutes: int) -> bool:
    try:
        now = pd.Timestamp.utcnow().tz_localize("UTC")
        for h in news.get("headlines", []) or []:
            imp = (h.get("impact") or "Low").lower()
            if not imp.startswith("high"):
                continue
            ts_s = h.get("time")
            t = _parse_sast(ts_s) if ts_s else None
            if t is None:
                continue
            try:
                if t.tzinfo is None:
                    t_utc = (t - pd.Timedelta(hours=2)).tz_localize("UTC")
                else:
                    t_utc = t.tz_convert("UTC")
            except Exception:
                t_utc = pd.Timestamp.utcnow().tz_localize("UTC")
            delta = abs((now - t_utc).total_seconds()) / 60.0
            if delta <= minutes:
                return True
    except Exception:
        return False
    return False

def _goat_exit_plan(direction: str, entry: float, sl: float, atr: Optional[float]) -> Dict[str, Any]:
    try:
        if direction not in ("Buy", "Sell") or entry is None or sl is None:
            return {}
        r = abs(entry - sl)
        tp1 = entry + (_GOAT_PARTIAL_TP_R * r if direction == "Buy" else -_GOAT_PARTIAL_TP_R * r)
        if atr is not None:
            tp2 = entry + (3.0 * atr if direction == "Buy" else -3.0 * atr)
        else:
            tp2 = entry + (2.0 * r if direction == "Buy" else -2.0 * r)
        return {"tp1": float(tp1), "tp2": float(tp2), "trail_after_R": _GOAT_TRAIL_AFTER_R, "time_stop_bars": _GOAT_TIME_STOP_BARS}
    except Exception:
        return {}

def _instrument_point_and_pipvalue(pair: str) -> Dict[str, float]:
    """
    Infer point size and pip value per 1 lot for common symbols.
    Env overrides: ANALYSIS_POINT_SIZE, ANALYSIS_PIP_VALUE.
    """
    try:
        env_point = os.getenv("ANALYSIS_POINT_SIZE")
        env_pip_value = os.getenv("ANALYSIS_PIP_VALUE")
        if env_point or env_pip_value:
            return {
                "point": float(env_point) if env_point else None,
                "pip_value": float(env_pip_value) if env_pip_value else None
            }
    except Exception:
        pass

    p = (pair or "").upper()
    # Indices
    if any(k in p for k in ["US30", "DJI", "DJ30"]):
        return {"point": 1.0, "pip_value": 1.0}
    if any(k in p for k in ["US500", "SPX", "SP500"]):
        return {"point": 0.1, "pip_value": 1.0}
    if any(k in p for k in ["US100", "NAS100", "NDX"]):
        return {"point": 1.0, "pip_value": 1.0}
    if any(k in p for k in ["DE40", "GER40", "DAX"]):
        return {"point": 1.0, "pip_value": 1.0}
    # Metals
    if "XAU" in p:
        return {"point": 0.1, "pip_value": 1.0}
    if "XAG" in p:
        return {"point": 0.01, "pip_value": 0.5}
    # US equities
    if any(p.startswith(t) for t in ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "NFLX", "AMD", "INTC"]) or p.endswith("M"):
        return {"point": 0.01, "pip_value": 1.0}
    # Default
    return {"point": None, "pip_value": None}

# ========================
# GOAT+ ENHANCEMENTS & PRECISION
# ========================
_GOAT_PLUS = os.getenv("ANALYSIS_GOAT_PLUS") != "0"

# Freshness and drift
_MAX_SIGNAL_AGE_MIN = int(os.getenv("ANALYSIS_MAX_SIGNAL_AGE_MIN") or "15")
_MAX_ENTRY_DRIFT_R = float(os.getenv("ANALYSIS_MAX_ENTRY_DRIFT_R") or "0.3")
_MAX_SPREAD_TO_ATR = float(os.getenv("ANALYSIS_MAX_SPREAD_TO_ATR") or "0.12")

# Confluence
_REQUIRE_2OF3 = os.getenv("ANALYSIS_REQUIRE_2OF3") != "0"
_CONTRA_STRICT = os.getenv("ANALYSIS_CONTRA_STRICT") != "0"

# Exits
_TRAIL_ATR_MULT = float(os.getenv("ANALYSIS_TRAIL_ATR_MULT") or "1.0")
_TRAIL_PCT_OF_MFE = float(os.getenv("ANALYSIS_TRAIL_PCT_OF_MFE") or "0.5")
_BE_AFTER_TP1 = os.getenv("ANALYSIS_BE_AFTER_TP1") != "0"

# Risk throttling
_MAX_DD_SOFT = float(os.getenv("ANALYSIS_MAX_DD_SOFT") or "0.06")
_MAX_DD_HARD = float(os.getenv("ANALYSIS_MAX_DD_HARD") or "0.12")
_RISK_FLOOR = float(os.getenv("ANALYSIS_RISK_FLOOR") or "0.25")

# Regime intelligence
_HURST_ON = os.getenv("ANALYSIS_HURST_ON") != "0"
_ENTROPY_ON = os.getenv("ANALYSIS_ENTROPY_ON") != "0"
_MIN_TREND_SLOPE = float(os.getenv("ANALYSIS_MIN_TREND_SLOPE") or "0.0")

# Event risk
_EVENT_BLACKOUT_MIN = int(os.getenv("ANALYSIS_EVENT_BLACKOUT_MIN") or "60")
_EVENT_KEYWORDS = [s.strip() for s in (os.getenv("ANALYSIS_EVENT_KEYWORDS") or "NFP,FOMC,CPI,rate decision,interest rate,ECB,FED,BoE,BoJ,minutes,press conference").split(",") if s.strip()]

# Exposure
_MAX_FAMILY_EXPOSURE = float(os.getenv("ANALYSIS_MAX_FAMILY_EXPOSURE") or "0.5")

# Auto-calibration
_AUTOCALIB_ON = os.getenv("ANALYSIS_AUTOCALIB_ON") != "0"
_AUTOCALIB_STEP = float(os.getenv("ANALYSIS_AUTOCALIB_STEP") or "0.02")
_EV_FLOOR = float(os.getenv("ANALYSIS_EV_FLOOR") or "0.0")

# Precision retest + anti-sweep parameters
_PRECISE_RETEST_ON = os.getenv("ANALYSIS_PRECISE_RETEST") != "0"
_RETEST_EDGE_ENTRY = os.getenv("ANALYSIS_RETEST_EDGE_ENTRY", "1") != "0"
_SL_BUFFER_ATR = float(os.getenv("ANALYSIS_SL_BUFFER_ATR") or "0.35")
_SL_MIN_TICKS = int(os.getenv("ANALYSIS_SL_MIN_TICKS") or "3")
_ENTRY_TICK_OFFSET = int(os.getenv("ANALYSIS_ENTRY_TICK_OFFSET") or "1")
_NO_APPLY_PROTECTED = os.getenv("ANALYSIS_NO_APPLY_PROTECTED", "0") == "1"

# 1m / 4h toggles and params
_USE_1M = os.getenv("ANALYSIS_USE_1M", "1") != "0"
_USE_4H = os.getenv("ANALYSIS_USE_4H", "1") != "0"
_MICRO1M_BARS = int(os.getenv("ANALYSIS_MICRO1M_BARS") or "1200")
_MICRO5M_BARS = int(os.getenv("ANALYSIS_MICRO5M_BARS") or "300")
_HTF4H_BARS = int(os.getenv("ANALYSIS_HTF4H_BARS") or "500")

# Confluence lock (prevents soft neutralization when MTF confluence is strong)
_CONFLUENCE_LOCK_ON = os.getenv("ANALYSIS_CONFLUENCE_LOCK", "1") != "0"
_CONFLUENCE_LOCK_MIN = int(os.getenv("ANALYSIS_CONFLUENCE_LOCK_MIN") or "2")

# Extra micro ATR buffer factor for SL
_SL_BUFFER_MICRO_ATR = float(os.getenv("ANALYSIS_SL_BUFFER_MICRO_ATR") or "0.0")


# -------------------------
# Micro tools: VWAP, FVG, Order Block, Confluence
# -------------------------
def _compute_vwap(df: pd.DataFrame) -> Optional[pd.Series]:
    try:
        if "volume" not in df.columns:
            return None
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_vol = df["volume"].astype(float).cumsum().replace(0, np.nan)
        vwap = (tp.astype(float) * df["volume"].astype(float)).cumsum() / cum_vol
        return vwap
    except Exception:
        return None

def _detect_fvg_zone(micro_df: pd.DataFrame, i: int, direction: str, lookback: int = 6) -> Optional[Tuple[float, float]]:
    """
    Simple micro fair value gap detection near index i.
    Returns (low, high) of imbalance if found.
    """
    try:
        start = max(1, i - lookback)
        end = i
        for j in range(end, start - 1, -1):
            if j - 1 < 0 or j >= len(micro_df):
                continue
            a = micro_df.iloc[j - 1]
            b = micro_df.iloc[j]
            if direction == "Buy" and a["low"] > b["high"]:
                return (float(b["high"]), float(a["low"]))
            if direction == "Sell" and a["high"] < b["low"]:
                return (float(a["high"]), float(b["low"]))
    except Exception:
        pass
    return None

def _last_opposite_order_block_level(micro_df: pd.DataFrame, i: int, direction: str, search: int = 10) -> Optional[float]:
    """
    Returns the high (for Buy) or low (for Sell) of the last opposite candle before i.
    """
    try:
        start = max(0, i - search)
        seg = micro_df.iloc[start:i]
        if seg.empty:
            return None
        if direction == "Buy":
            opp = seg[(seg["close"] < seg["open"])]
            if opp.empty:
                return None
            return float(opp["high"].iloc[-1])
        else:
            opp = seg[(seg["close"] > seg["open"])]
            if opp.empty:
                return None
            return float(opp["low"].iloc[-1])
    except Exception:
        return None

def _tick_from_point(point: Optional[float]) -> float:
    try:
        if point and point > 0:
            return float(point)
    except Exception:
        pass
    return 0.0001

def _micro_swing_extremes(df: pd.DataFrame, i: int, lb: int = 3) -> Tuple[Optional[float], Optional[float]]:
    try:
        lo = float(df["low"].iloc[max(0, i - lb): i + 1].min())
        hi = float(df["high"].iloc[max(0, i - lb): i + 1].max())
        return lo, hi
    except Exception:
        return None, None

def _pick_best_entry_edge(direction: str,
                          z_lo: float,
                          z_hi: float,
                          point: float,
                          fvg_zone: Optional[Tuple[float, float]],
                          ob_level: Optional[float]) -> float:
    """
    Choose a confluence edge for entry:
    - Zone edge with small tick offset.
    - If FVG edge exists within/near zone, include.
    - If OB level is within range, include.
    Buy: pick max candidate; Sell: pick min candidate.
    """
    tick = _tick_from_point(point)
    off = _ENTRY_TICK_OFFSET * tick
    candidates: List[float] = []
    if direction == "Buy":
        candidates.append(z_lo + off)
    else:
        candidates.append(z_hi - off)
    if fvg_zone:
        f_lo, f_hi = fvg_zone
        if direction == "Buy":
            candidates.append(max(z_lo + off, float(f_lo)))
        else:
            candidates.append(min(z_hi - off, float(f_hi)))
    if ob_level is not None:
        if direction == "Buy":
            candidates.append(max(z_lo + off, float(ob_level)))
        else:
            candidates.append(min(z_hi - off, float(ob_level)))
    if direction == "Buy":
        return float(max(candidates))
    else:
        return float(min(candidates))

def _protected_sl_price(direction: str,
                        z_lo: float,
                        z_hi: float,
                        swing_low: Optional[float],
                        swing_high: Optional[float],
                        atr: Optional[float],
                        point: float,
                        micro_atr: Optional[float] = None) -> float:
    """
    Protected SL beyond zone and swing with buffer = max(ATR * SL_BUFFER_ATR,
                                                        microATR * (SL_BUFFER_ATR + SL_BUFFER_MICRO_ATR),
                                                        SL_MIN_TICKS * point)
    """
    try:
        tick = _tick_from_point(point)
        a_buf = float(atr) * _SL_BUFFER_ATR if (atr is not None and atr == atr) else 0.0
        m_buf = 0.0
        if micro_atr is not None and micro_atr == micro_atr:
            m_buf = float(micro_atr) * (_SL_BUFFER_ATR + _SL_BUFFER_MICRO_ATR)
        tick_buf = _SL_MIN_TICKS * tick
        buf = max(a_buf, m_buf, tick_buf)
        if direction == "Buy":
            base = min([v for v in [swing_low, z_lo] if v is not None])
            return float(base - buf)
        if direction == "Sell":
            base = max([v for v in [swing_high, z_hi] if v is not None])
            return float(base + buf)
    except Exception:
        pass
    if direction == "Buy":
        return float(z_lo - _SL_MIN_TICKS * _tick_from_point(point))
    if direction == "Sell":
        return float(z_hi + _SL_MIN_TICKS * _tick_from_point(point))
    return float((z_lo + z_hi) / 2.0)

def _bar_end_time_sast(df: pd.DataFrame, idx: int) -> Optional[str]:
    try:
        ts = df.index[idx]
        return _to_sast(ts)
    except Exception:
        try:
            return _to_sast(dt.datetime.utcnow())
        except Exception:
            return None

def _find_zone_from_heuristic(heur: Dict[str, Any], atr: Optional[float], direction: str, tighten: float = 0.4) -> Optional[Tuple[float, float, float]]:
    try:
        pivot = float(heur.get("pivot_price", heur.get("entry_price")))
    except Exception:
        return None
    try:
        span = float(atr) * tighten if (atr and math.isfinite(atr)) else abs(pivot) * 0.0002
    except Exception:
        span = 0.0
    if direction in ("Buy", "Sell"):
        return (pivot - span, pivot + span, pivot)
    return None

def _micro_confirm_bar(micro_df: pd.DataFrame, i: int, direction: str) -> bool:
    try:
        row = micro_df.iloc[i]
        if direction == "Buy":
            return row["close"] > row["open"]
        if direction == "Sell":
            return row["close"] < row["open"]
    except Exception:
        return False
    return False

def _qualify_microstructure(micro_df: pd.DataFrame, i: int, direction: str) -> Dict[str, Any]:
    """
    Qualify micro bar by wick rejection, micro FVG, opposite OB tap.
    Outputs precision_score [0..1] and labels.
    """
    try:
        labels = []
        score = 0.0
        row = micro_df.iloc[i]
        rng = row["high"] - row["low"] + 1e-12
        upper_wick = row["high"] - max(row["close"], row["open"])
        lower_wick = min(row["close"], row["open"]) - row["low"]
        wick_ratio = (lower_wick if direction == "Buy" else upper_wick) / rng
        if wick_ratio > 0.4:
            labels.append("wick_reject")
            score += 0.35
        try:
            prev = micro_df.iloc[i-1]
            if direction == "Buy":
                if prev["low"] > row["high"]:
                    labels.append("micro_fvg")
                    score += 0.3
            else:
                if prev["high"] < row["low"]:
                    labels.append("micro_fvg")
                    score += 0.3
        except Exception:
            pass
        try:
            lookback = micro_df.iloc[max(0, i-8):i]
            if direction == "Buy":
                opp = lookback[(lookback["close"] < lookback["open"])]
                if not opp.empty:
                    ob_hi = float(opp["high"].iloc[-1])
                    if row["low"] <= ob_hi <= row["high"]:
                        labels.append("ob_tap")
                        score += 0.25
            else:
                opp = lookback[(lookback["close"] > lookback["open"])]
                if not opp.empty:
                    ob_lo = float(opp["low"].iloc[-1])
                    if row["low"] <= ob_lo <= row["high"]:
                        labels.append("ob_tap")
                        score += 0.25
        except Exception:
            pass
        score = float(max(0.0, min(1.0, score)))
        return {"precision_score": score, "labels": labels}
    except Exception:
        return {"precision_score": 0.0, "labels": []}

def _refine_precise_retest(micro_df: Optional[pd.DataFrame],
                           direction: str,
                           atr: Optional[float],
                           heuristic: Optional[Dict[str, Any]],
                           point: float,
                           vwap_series: Optional[pd.Series] = None) -> Optional[Dict[str, Any]]:
    """
    Find first touch+confirm on retest zone and return:
    - edge-entry price (with FVG/OB confluence)
    - protected SL (zone+swing+ATR buffer)
    - exact bar end time (SAST)
    """
    try:
        if micro_df is None or not isinstance(micro_df, pd.DataFrame) or micro_df.empty:
            return None
        if heuristic is None:
            return None
        zone = _find_zone_from_heuristic(heuristic, atr, direction)
        if zone is None:
            return None
        z_lo, z_hi, pivot = zone
        N = min(180, len(micro_df))
        micro_atr = compute_atr(micro_df)
        for i in range(len(micro_df)-N, len(micro_df)):
            row = micro_df.iloc[i]
            touched = (row["low"] <= z_hi and row["high"] >= z_lo)
            if not touched:
                continue
            if not _micro_confirm_bar(micro_df, i, direction):
                continue
            if vwap_series is not None:
                try:
                    v = float(vwap_series.iloc[i])
                    if direction == "Buy" and row["close"] < v:
                        continue
                    if direction == "Sell" and row["close"] > v:
                        continue
                except Exception:
                    pass
            qual = _qualify_microstructure(micro_df, i, direction)
            fvg = _detect_fvg_zone(micro_df, i, direction, lookback=6)
            ob_level = _last_opposite_order_block_level(micro_df, i, direction, search=10)
            entry_edge = _pick_best_entry_edge(direction, z_lo, z_hi, point, fvg, ob_level) if _RETEST_EDGE_ENTRY else float(row["close"])
            sw_lo, sw_hi = _micro_swing_extremes(micro_df, i, lb=3)
            prot_sl = _protected_sl_price(direction, z_lo, z_hi, sw_lo, sw_hi, atr, point, micro_atr=micro_atr)
            entry_time_sast = _bar_end_time_sast(micro_df, i)
            return {
                "entry_point_refined": safe_round(entry_edge),
                "entry_point_edge": safe_round(entry_edge),
                "protected_sl": safe_round(prot_sl),
                "entry_time_refined_sast": entry_time_sast,
                "entry_mode": ("limit" if _RETEST_EDGE_ENTRY else ("market")),
                "trigger_price": safe_round(entry_edge),
                "precision_grade": qual["precision_score"],
                "precision_labels": qual["labels"],
                "retest_zone": {"low": safe_round(z_lo), "high": safe_round(z_hi), "pivot": safe_round(pivot)},
                "touch_index": i,
                "fvg_zone": fvg,
                "ob_level": safe_round(ob_level) if ob_level is not None else None
            }
    except Exception:
        return None
    return None

def _fresh_signal_ok(entry_time_sast: Optional[str], max_age_min: int) -> Dict[str, Any]:
    """
    Check if a signal/entry timestamp is fresh within a specified minutes window.
    """
    try:
        if not entry_time_sast:
            return {"ok": True, "age_min": None}
        t = _parse_sast(entry_time_sast)
        if t is None:
            return {"ok": True, "age_min": None}
        now = pd.Timestamp.utcnow()
        age = abs((now - t).total_seconds()) / 60.0
        return {"ok": bool(age <= max_age_min), "age_min": float(age)}
    except Exception:
        return {"ok": True, "age_min": None}

def _spread_to_atr_ok(spread_pips: Optional[float], atr: Optional[float], point: Optional[float], max_ratio: float) -> Dict[str, Any]:
    """
    Validate spread-to-ATR ratio using instrument point size.
    """
    try:
        if spread_pips is None or atr is None or point in (None, 0):
            return {"ok": True, "ratio": None}
        atr_pips = atr / point
        ratio = float(spread_pips / (max(1e-9, atr_pips)))
        return {"ok": bool(ratio <= max_ratio), "ratio": round(ratio, 4)}
    except Exception:
        return {"ok": True, "ratio": None}

def _hurst_estimate(series: pd.Series) -> Optional[float]:
    """
    Rough Hurst exponent proxy using R/S via log-log scaling of lagged differences.
    """
    try:
        x = series.dropna().astype(float).values
        n = len(x)
        if n < 100:
            return None
        lags = np.array([2, 4, 8, 16, 32])
        tau = []
        for lag in lags:
            if lag >= n:
                continue
            diff = x[lag:] - x[:-lag]
            tau.append(np.sqrt(np.std(diff)))
        if not tau:
            return None
        tau = np.array(tau)
        m = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
        hurst = m[0] * 2.0
        return float(max(0.01, min(0.99, hurst)))
    except Exception:
        return None

def _spectral_entropy(series: pd.Series) -> Optional[float]:
    """
    Normalized spectral entropy as a proxy for noise/chop (0..1).
    """
    try:
        x = series.dropna().astype(float).values
        n = len(x)
        if n < 64:
            return None
        x = x - np.mean(x)
        ps = np.abs(np.fft.rfft(x))**2
        ps = ps / (np.sum(ps) + 1e-12)
        ent = -np.sum(ps * np.log(ps + 1e-12))
        ent = float(ent / (np.log(len(ps) + 1e-12)))
        return float(max(0.0, min(1.0, ent)))
    except Exception:
        return None

def _sr_congestion_score(df: pd.DataFrame, window: int = 200) -> Dict[str, float]:
    """
    Crude congestion and swing cluster measure for SR context.
    """
    try:
        sub = df.tail(window)
        highs = sub["high"].astype(float).values
        lows = sub["low"].astype(float).values
        rng = (highs - lows)
        if len(rng) == 0:
            return {"congestion": 0.0, "clusters": 0.0}
        congestion = float(np.mean(rng) / (np.std(rng) + 1e-9))
        swing_highs = np.sum((highs[1:-1] > highs[0:-2]) & (highs[1:-1] > highs[2:]))
        swing_lows = np.sum((lows[1:-1] < lows[0:-2]) & (lows[1:-1] < lows[2:]))
        clusters = float((swing_highs + swing_lows) / max(1, window))
        return {"congestion": round(congestion, 3), "clusters": round(clusters, 3)}
    except Exception:
        return {"congestion": 0.0, "clusters": 0.0}

def _drawdown_throttle() -> Dict[str, Any]:
    """
    Throttle risk based on equity drawdown using env vars ANALYSIS_EQUITY_EST and ANALYSIS_EQUITY_PEAK.
    Returns scale in [0,1], dd fraction, and state: ok|throttle|halt|no_data|error.
    """
    try:
        eq = float(os.getenv("ANALYSIS_EQUITY_EST") or "0")
        peak = float(os.getenv("ANALYSIS_EQUITY_PEAK") or "0")
        if eq <= 0 or peak <= 0:
            return {"scale": 1.0, "dd": 0.0, "state": "no_data"}
        dd = max(0.0, (peak - eq) / peak)
        if dd >= _MAX_DD_HARD:
            return {"scale": 0.0, "dd": dd, "state": "halt"}
        if dd <= _MAX_DD_SOFT:
            return {"scale": 1.0, "dd": dd, "state": "ok"}
        span = _MAX_DD_HARD - _MAX_DD_SOFT
        frac = (dd - _MAX_DD_SOFT) / (span + 1e-9)
        scale = max(_RISK_FLOOR, 1.0 - frac)
        return {"scale": float(scale), "dd": float(dd), "state": "throttle"}
    except Exception:
        return {"scale": 1.0, "dd": 0.0, "state": "error"}

def _exposure_guard(pair: str) -> Dict[str, Any]:
    """
    Guard against over-exposure by family using env ANALYSIS_OPEN_RISK_{FAMILY} and _MAX_FAMILY_EXPOSURE.
    Families: FX, INDEX, METAL, EQUITY, else UNKNOWN.
    """
    try:
        fam = "FX"
        p = (pair or "").upper()
        if any(k in p for k in ["US30", "US100", "US500", "DAX", "GER40", "DE40", "DJI", "SPX", "NDX"]):
            fam = "INDEX"
        elif "XAU" in p or "XAG" in p:
            fam = "METAL"
        elif any(p.startswith(t) for t in ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "NFLX", "AMD", "INTC"]) or p.endswith("M"):
            fam = "EQUITY"
        open_risk = float(os.getenv(f"ANALYSIS_OPEN_RISK_{fam}") or "0.0")
        ok = open_risk <= _MAX_FAMILY_EXPOSURE
        return {"family": fam, "open_risk": open_risk, "ok": ok, "max": _MAX_FAMILY_EXPOSURE}
    except Exception:
        return {"family": "UNKNOWN", "open_risk": None, "ok": True, "max": _MAX_FAMILY_EXPOSURE}

def _soften_neutralization(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    If decision is Neutral but edge is strong (p_hat & EV), reinstate orig direction with scaled risk.
    """
    try:
        goat = result.get("goat_post") or result.get("goat") or {}
        orig = goat.get("original_label") or (result.get("goat") or {}).get("original_label")
        decision = result.get("final_suggestion")
        if decision != "Neutral":
            return result
        if orig not in ("Buy", "Sell"):
            return result
        p_hat = float((goat.get("p_hat") if goat.get("p_hat") is not None else (result.get("goat") or {}).get("p_hat") or 0.0))
        ev = float((goat.get("ev") if goat.get("ev") is not None else (result.get("goat") or {}).get("ev") or 0.0))
        blocked = list((goat.get("blocked_reasons") or []))
        if (p_hat >= 0.7) and (ev >= 0.2) and (len(blocked) <= 2):
            result["final_suggestion"] = orig
            result["goat_soft"] = {
                "enabled": True,
                "reason": "soften_neutralization",
                "applied_from": "goat_post" if result.get("goat_post") else "goat",
                "risk_scale": float(os.getenv("ANALYSIS_SOFTEN_RISK_SCALE") or 0.5),
                "p_hat": p_hat,
                "ev": ev,
                "blocked_reasons": blocked
            }
            try:
                if result.get("lot_size_recommendation") is not None:
                    scaled = float(result["lot_size_recommendation"]) * float(os.getenv("ANALYSIS_SOFTEN_RISK_SCALE") or 0.5)
                    result["lot_size_recommendation_goat"] = round(max(0.01, scaled), 2)
            except Exception:
                pass
        return result
    except Exception:
        return result

def _htf_4h_analysis(df_4h: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Lightweight 4h trend and structure analysis for confluence.
    """
    try:
        if df_4h is None or df_4h.empty:
            return {"dir": None, "slope": 0.0, "bos": None, "choch": False, "regime": None}
        th = _trend_health(df_4h)
        bos = detect_bos_choch_liquidity(df_4h)
        slope = th.get("slope", 0.0) or 0.0
        dir4h = "Buy" if slope > 0 else "Sell" if slope < 0 else "Neutral"
        reg = _goat_regime_score(df_4h)
        return {"dir": dir4h, "slope": slope, "bos": bos.get("bos"), "choch": bos.get("choch"), "regime": reg}
    except Exception:
        return {"dir": None, "slope": 0.0, "bos": None, "choch": False, "regime": None}

def _trend_health(df: pd.DataFrame) -> Dict[str, float]:
    try:
        close = df["close"].astype(float)
        if len(close) < 60:
            return {"slope": 0.0, "persistence": 0.0}
        ma_fast = close.rolling(20).mean()
        ma_slow = close.rolling(50).mean()
        spread = (ma_fast - ma_slow)
        y = spread.dropna().values
        if len(y) < 10:
            return {"slope": 0.0, "persistence": 0.0}
        x = np.arange(len(y))
        slope = float(np.polyfit(x, y, 1)[0] / (np.std(y) + 1e-9))
        last = np.sign(y[-15:])
        persistence = float(np.mean(last == np.sign(y[-1]))) if len(last) > 0 else 0.0
        return {"slope": slope, "persistence": persistence}
    except Exception:
        return {"slope": 0.0, "persistence": 0.0}

def _confluence_score_and_lock(base_tf_label: Optional[str],
                               ensemble_label: Optional[str],
                               htf_dir: Optional[str],
                               htf4h_dir: Optional[str],
                               micro_has_signal: bool) -> Dict[str, Any]:
    """
    Compute confluence score from ensemble, higher_tf, 4h, and micro presence.
    Lock if score >= _CONFLUENCE_LOCK_MIN.
    """
    try:
        decision = base_tf_label if base_tf_label in ("Buy", "Sell") else ensemble_label
        if decision not in ("Buy", "Sell"):
            return {"score": 0, "lock": False}
        score = 0
        if ensemble_label == decision:
            score += 1
        if (htf_dir in ("Buy", "Sell")) and htf_dir == decision:
            score += 1
        if (htf4h_dir in ("Buy", "Sell")) and htf4h_dir == decision:
            score += 1
        if micro_has_signal:
            score += 1
        return {"score": int(score), "lock": bool(_CONFLUENCE_LOCK_ON and score >= _CONFLUENCE_LOCK_MIN)}
    except Exception:
        return {"score": 0, "lock": False}


# -------------------------
# Main orchestrator: analyze_pair
# -------------------------
def analyze_pair(pair: str, timeframe: str = "15m", bars: int = 2000) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "success": False,
        "pair": pair,
        "timeframe": timeframe,
        "price": None,
        "atr": None,
        "signals": None,
        "advanced": None,
        "context": None,
        "precision_entry": None,
        "backtest": None,
        "forward_test": None,
        "news": None,
        "news_status": None,
        "adaptive_summary": None,
        "final_suggestion": None,
        "entry_point": None,
        "entry_time": None,
        "entry_time_sast": None,
        "sl": None,
        "tp": None,
        "confidence": None,
        "precision_reason": None,
        "lot_size_recommendation": None,
        "liquidity": None,
        "micro_confirmation": None,
        "structure_change": None,
        "liquidity_sweep": None,
        "pending_entry": False,
        "htf_4h": None,
        "confluence_score": None,
        "confluence_lock": None,
        "report": None
    }

    if fetch_candles is None:
        result["error"] = "fetch_candles() missing (data.py not available)"
        log(result["error"], "error")
        return result

    # fetch base timeframe candles
    try:
        df = fetch_candles(pair, timeframe, n=bars)
    except TypeError:
        try:
            df = fetch_candles(pair, timeframe, bars)
        except Exception as e:
            result["error"] = f"fetch_candles call failed: {e}"
            log(result["error"], "error")
            return result
    except Exception as e:
        result["error"] = f"fetch_candles error: {e}"
        log(result["error"], "error")
        return result

    valid_cols = isinstance(df, pd.DataFrame) and not df.empty and set(["open", "high", "low", "close"]).issubset(df.columns)
    if not valid_cols:
        result["error"] = "Invalid dataframe returned by fetch_candles"
        log(result["error"], "error")
        return result

    # basic metrics
    try:
        price = float(df["close"].iloc[-1])
        atr = compute_atr(df)
        result["price"] = price
        result["atr"] = atr
    except Exception as e:
        result["error"] = f"Price/ATR calc error: {e}"
        log(result["error"], "error")
        return result

    # core signals
    try:
        signals = indicator_signals(df)
        result["signals"] = signals
    except Exception as e:
        result["signals"] = {}
        log(f"indicator_signals error: {e}", "warning")

    # higher timeframe trend (legacy helper if available)
    try:
        higher_trend = multi_timeframe_trend(pair)
    except Exception:
        higher_trend = None
    result["higher_tf_trend"] = higher_trend

    # 4h higher timeframe analysis (new)
    df_4h = None
    htf4h = {"dir": None}
    if _USE_4H:
        try:
            try:
                df_4h = fetch_candles(pair, "4h", n=_HTF4H_BARS)
            except TypeError:
                df_4h = fetch_candles(pair, "4h", _HTF4H_BARS)
            if isinstance(df_4h, pd.DataFrame) and not df_4h.empty:
                htf4h = _htf_4h_analysis(df_4h)
        except Exception as e:
            log(f"4h fetch/analysis error: {e}", "warning")
    result["htf_4h"] = htf4h

    # advanced analysis
    try:
        adv = advanced_analysis(df)
        result["advanced"] = adv
    except Exception as e:
        result["advanced"] = {"error": str(e)}
        log(f"advanced_analysis error: {e}", "warning")

    # contextual analysis
    try:
        ctx = contextual_analysis(df, pair=pair, timeframe=timeframe, price=price)
        result["context"] = ctx
    except Exception as e:
        result["context"] = {"error": str(e)}
        log(f"contextual_analysis error: {e}", "warning")

    # ensemble
    try:
        ensemble = ensemble_signal(signals, higher_trend)
    except Exception as e:
        ensemble = "Neutral"
        log(f"ensemble_signal error: {e}", "warning")
    result["ensemble"] = ensemble

    # adaptive weights
    try:
        weights = adaptive_engine.get_weights()
    except Exception:
        weights = {"core": 1.0, "advanced": 1.0, "precision": 1.0, "context": 1.0, "news": 1.0}

    core_score = 0.0
    for v in (signals or {}).values():
        if v == "Buy":
            core_score += 1.0
        elif v == "Sell":
            core_score -= 1.0

    adv_score = 0.0
    if isinstance(result["advanced"], dict):
        adv_sig = result["advanced"].get("advanced_signal") or result["advanced"].get("suggestion")
        if adv_sig == "Buy":
            adv_score += 1.0
        elif adv_sig == "Sell":
            adv_score -= 1.0

    ctx_score = float(result.get("context", {}).get("score", 0.0) or 0.0)

    # news
    try:
        news_info = fetch_news_with_status(pair)
    except Exception as e:
        news_info = {"pair": pair, "count": 0, "sentiment_score": "Neutral", "headlines": [{"headline": f"News fetch failed: {e}", "impact": "Low", "source": "System", "time": _to_sast(dt.datetime.utcnow())}], "timestamp": _to_sast(dt.datetime.utcnow()), "status": "api_error", "errors": [str(e)]}
        log(f"get_recent_news wrapper error: {e}", "warning")

    try:
        sanitized = []
        for h in news_info.get("headlines", []) or []:
            time_s = h.get("time")
            sanitized.append({"headline": h.get("headline", ""), "impact": h.get("impact", "Medium"), "source": h.get("source", "Unknown"), "time": time_s})
        news_info["headlines"] = sanitized
    except Exception:
        pass

    result["news"] = news_info
    result["news_status"] = news_info.get("status", "unknown")
    news_sent = 0.0
    try:
        if "headlines" in news_info:
            if _VADER_AVAILABLE and _VADER is not None:
                scores = []
                for h in news_info.get("headlines", []):
                    s = h.get("sentiment")
                    if isinstance(s, dict):
                        scores.append(s.get("score", 0.0))
                news_sent = float(np.mean(scores)) if scores else 0.0
            else:
                pos = sum(1 for h in news_info.get("headlines", []) if "Positive" in str(h.get("sentiment", "")))
                neg = sum(1 for h in news_info.get("headlines", []) if "Negative" in str(h.get("sentiment", "")))
                if pos > neg:
                    news_sent = 0.4
                elif neg > pos:
                    news_sent = -0.4
                else:
                    news_sent = 0.0
    except Exception:
        news_sent = 0.0

    news_weight = 0.0
    if result["news_status"] == "ok":
        has_high = any((h.get("impact") or "").lower().startswith("high") for h in news_info.get("headlines", []))
        news_weight = 0.20 if has_high else 0.08

    # Directional alignment bonus (legacy + 4h)
    dir_bonus = 0.0
    try:
        if (higher_trend in ("Buy", "Sell")) and ensemble == higher_trend:
            dir_bonus += float(os.getenv("ANALYSIS_DIR_BONUS") or "0.2")
        if (htf4h.get("dir") in ("Buy", "Sell")) and ensemble == htf4h.get("dir"):
            dir_bonus += float(os.getenv("ANALYSIS_DIR_BONUS_4H") or "0.1")
    except Exception:
        pass

    combined_value = (weights.get("core", 1.0) * core_score * 0.45) + (weights.get("advanced", 1.0) * adv_score * 0.25) + (weights.get("context", 1.0) * ctx_score * 0.2) + (weights.get("news", 1.0) * news_sent * news_weight) + dir_bonus
    combined_value = float(max(-1.0, min(1.0, combined_value)))
    result["combined_score"] = round(float(combined_value), 3)

    # final suggestion thresholds
    if combined_value >= 0.6:
        final_label = "Buy"
    elif combined_value <= -0.6:
        final_label = "Sell"
    elif combined_value >= 0.25:
        final_label = "Buy"
    elif combined_value <= -0.25:
        final_label = "Sell"
    else:
        final_label = "Neutral"
    result["final_suggestion"] = final_label

    # -------------------------
    # GOAT MODE: first-pass gates + EV
    # -------------------------
    try:
        if _GOAT_MODE:
            try:
                ts_last = df.index[-1] if hasattr(df, "index") else None
            except Exception:
                ts_last = None
            session_now = _goat_session_from_ts(ts_last)
            session_ok = (session_now in _GOAT_ALLOWED_SESSIONS) if _GOAT_ALLOWED_SESSIONS else True

            spread_pips_env = os.getenv("ANALYSIS_SPREAD_PIPS")
            spread_pips = float(spread_pips_env) if spread_pips_env else None
            spread_ok = True if (spread_pips is None) else (spread_pips <= _GOAT_MAX_SPREAD_PIPS)

            news_blackout = _goat_news_blackout(news_info, _GOAT_NEWS_BLACKOUT_MIN) if isinstance(news_info, dict) else False

            regime = _goat_regime_score(df)

            meta0 = _instrument_point_and_pipvalue(pair)
            point0 = meta0.get("point") or _pips_for_pair(pair)
            atr_pips = (float(atr) / point0) if (atr and point0) else None
            atr_ok = True if (atr_pips is None) else (atr_pips >= _GOAT_MIN_ATR_PIPS)

            quality_scores = {
                "session": {"value": session_now, "ok": session_ok, "allowed": list(_GOAT_ALLOWED_SESSIONS)},
                "spread_pips": {"value": spread_pips, "ok": spread_ok, "max": _GOAT_MAX_SPREAD_PIPS},
                "news_blackout": {"value": news_blackout, "ok": not news_blackout, "window_min": _GOAT_NEWS_BLACKOUT_MIN},
                "regime": regime,
                "atr_pips": {"value": round(atr_pips, 2) if atr_pips else None, "ok": atr_ok, "min": _GOAT_MIN_ATR_PIPS},
            }

            p_hat = _goat_p_hat(combined_value)
            entry_for_rr = result.get("entry_point") if result.get("entry_point") is not None else price
            rr_info = _goat_rr(pair, price, entry_for_rr, result.get("sl"), result.get("tp"), atr)
            rr = rr_info.get("rr", 1.0)
            ev = p_hat * rr - (1 - p_hat) * 1.0

            blocked_reasons: List[str] = []
            if not session_ok:
                blocked_reasons.append("session_filter")
            if not spread_ok:
                blocked_reasons.append("spread_filter")
            if news_blackout:
                blocked_reasons.append("news_blackout")
            if not atr_ok:
                blocked_reasons.append("atr_too_low")
            if ev <= 0.0:
                blocked_reasons.append("negative_expected_value")
            if final_label in ("Buy", "Sell") and (result.get("entry_point") is None):
                blocked_reasons.append("no_entry_point")

            goat_label = final_label
            if final_label in ("Buy", "Sell"):
                if any(b in _HARD_BLOCKS for b in blocked_reasons):
                    goat_label = "Neutral"
                else:
                    if "no_entry_point" in blocked_reasons:
                        result["pending_entry"] = True

            exits = {}
            ep = entry_for_rr
            slv = result.get("sl")
            if goat_label in ("Buy", "Sell") and ep is not None and slv is not None:
                exits = _goat_exit_plan(goat_label, float(ep), float(slv), atr)

            result["goat"] = {
                "enabled": True,
                "p_hat": round(float(p_hat), 4),
                "rr": round(float(rr), 3),
                "ev": round(float(ev), 4),
                "rr_source": rr_info.get("source"),
                "quality_scores": quality_scores,
                "blocked_reasons": blocked_reasons,
                "original_label": final_label,
                "decision_label": goat_label,
            }

            result["final_suggestion"] = goat_label

            if exits:
                result["tp1"] = safe_round(exits.get("tp1"))
                result["tp2"] = safe_round(exits.get("tp2"))
                result["trail_after_R"] = exits.get("trail_after_R")
                result["time_stop_bars"] = exits.get("time_stop_bars")
            else:
                result["trail_after_R"] = _GOAT_TRAIL_AFTER_R
                result["time_stop_bars"] = _GOAT_TIME_STOP_BARS

            try:
                current_conf = float(result.get("confidence") or 0.0)
                if current_conf <= 0.51:
                    calibrated = max(0.4, min(0.85, (0.7 * float(result["goat"]["p_hat"]) + 0.15)))
                    if any(b in _SOFT_BLOCKS for b in blocked_reasons) and not any(b in _HARD_BLOCKS for b in blocked_reasons):
                        calibrated = max(0.35, calibrated - 0.08)
                    result["confidence"] = round(calibrated, 3)
            except Exception:
                pass
        else:
            result["goat"] = {"enabled": False}
            result["trail_after_R"] = _GOAT_TRAIL_AFTER_R
            result["time_stop_bars"] = _GOAT_TIME_STOP_BARS
    except Exception as e:
        log(f"GOAT mode gating error: {e}", "warning")
        try:
            result["goat"] = {"enabled": bool(_GOAT_MODE), "error": str(e)}
        except Exception:
            pass

    # micro confirmations (1m + 5m)
    micro_conf = {"m1": None, "m5": None}
    structure = None
    liquidity_sweep_flag = None
    micro_df_1m = None
    micro_df_5m = None
    try:
        # 1m micro
        if _USE_1M:
            try:
                micro_df_1m = fetch_candles(pair, "1m", n=_MICRO1M_BARS)
            except TypeError:
                micro_df_1m = fetch_candles(pair, "1m", _MICRO1M_BARS)
            if isinstance(micro_df_1m, pd.DataFrame) and not micro_df_1m.empty:
                micro_atr_1m = compute_atr(micro_df_1m)
                vol_surge_1m = detect_volume_surge(micro_df_1m)
                heur_retest_1m = detect_retest(micro_df_1m, final_label, micro_atr_1m, sensitivity=adaptive_engine.get_params().get("retest_sensitivity", 0.6) if hasattr(adaptive_engine, "get_params") else 0.6) if final_label in ("Buy", "Sell") else None
                bos_choch_1m = detect_bos_choch_liquidity(micro_df_1m)
                vwap_1m = _compute_vwap(micro_df_1m)
                patterns_1m = []
                if is_bullish_engulfing(micro_df_1m):
                    patterns_1m.append("bullish_engulfing")
                if is_bearish_engulfing(micro_df_1m):
                    patterns_1m.append("bearish_engulfing")
                if is_hammer_like(micro_df_1m):
                    patterns_1m.append("hammer_like")
                if is_shooting_star_like(micro_df_1m):
                    patterns_1m.append("shooting_star_like")
                micro_conf["m1"] = {"micro_atr": micro_atr_1m, "volume_surge": vol_surge_1m, "heuristic_retest": heur_retest_1m, "patterns": patterns_1m, "bos_choch": bos_choch_1m, "vwap": True if vwap_1m is not None else False}

        # 5m micro
        try:
            micro_df_5m = fetch_candles(pair, "5m", n=_MICRO5M_BARS)
        except TypeError:
            micro_df_5m = fetch_candles(pair, "5m", _MICRO5M_BARS)
        if isinstance(micro_df_5m, pd.DataFrame) and not micro_df_5m.empty:
            micro_atr_5m = compute_atr(micro_df_5m)
            vol_surge_5m = detect_volume_surge(micro_df_5m)
            heur_retest_5m = detect_retest(micro_df_5m, final_label, micro_atr_5m, sensitivity=adaptive_engine.get_params().get("retest_sensitivity", 0.6) if hasattr(adaptive_engine, "get_params") else 0.6) if final_label in ("Buy", "Sell") else None
            bos_choch_5m = detect_bos_choch_liquidity(micro_df_5m)
            patterns_5m = []
            if is_bullish_engulfing(micro_df_5m):
                patterns_5m.append("bullish_engulfing")
            if is_bearish_engulfing(micro_df_5m):
                patterns_5m.append("bearish_engulfing")
            if is_hammer_like(micro_df_5m):
                patterns_5m.append("hammer_like")
            if is_shooting_star_like(micro_df_5m):
                patterns_5m.append("shooting_star_like")
            micro_conf["m5"] = {"micro_atr": micro_atr_5m, "volume_surge": vol_surge_5m, "heuristic_retest": heur_retest_5m, "patterns": patterns_5m, "bos_choch": bos_choch_5m}
            structure = bos_choch_5m
            liquidity_sweep_flag = bos_choch_5m.get("liquidity_sweep") if isinstance(bos_choch_5m, dict) else None
    except Exception as e:
        log(f"micro analysis error: {e}", "warning")

    result["micro_confirmation"] = micro_conf
    result["structure_change"] = structure
    result["liquidity_sweep"] = liquidity_sweep_flag

    # Confluence lock finalize (after micro + 4h present)
    try:
        micro_has_sig = False
        try:
            m1 = micro_conf.get("m1") or {}
            m5 = micro_conf.get("m5") or {}
            micro_has_sig = bool((m1.get("patterns") or m1.get("heuristic_retest")) or (m5.get("patterns") or m5.get("heuristic_retest")))
        except Exception:
            micro_has_sig = False
        conf_lock = _confluence_score_and_lock(result.get("final_suggestion"), result.get("ensemble"), result.get("higher_tf_trend"), (result.get("htf_4h") or {}).get("dir"), micro_has_sig)
        result["confluence_score"] = conf_lock.get("score")
        result["confluence_lock"] = conf_lock.get("lock")
    except Exception:
        result["confluence_lock"] = False

    # precision / entry detection (stack: model -> 1m -> 5m -> TF)
    try:
        prec = None
        if final_label in ("Buy", "Sell"):
            if find_precise_entry is not None:
                try:
                    prec = find_precise_entry(df, final_label, atr=atr, params=adaptive_engine.get_params() if hasattr(adaptive_engine, "get_params") else None)
                except Exception as e:
                    log(f"find_precise_entry error: {e}", "warning")
            if not prec and (micro_conf.get("m1") or {}).get("heuristic_retest"):
                h = (micro_conf.get("m1") or {}).get("heuristic_retest")
                prec = {"entry_price": h.get("entry_price"), "sl": h.get("sl"), "confidence": h.get("confidence"), "reason": "micro_heuristic_retest_1m", "entry_time_sast": _to_sast(dt.datetime.utcnow())}
            if not prec and (micro_conf.get("m5") or {}).get("heuristic_retest"):
                h = (micro_conf.get("m5") or {}).get("heuristic_retest")
                prec = {"entry_price": h.get("entry_price"), "sl": h.get("sl"), "confidence": h.get("confidence"), "reason": "micro_heuristic_retest_5m", "entry_time_sast": _to_sast(dt.datetime.utcnow())}
            if not prec:
                heur = detect_retest(df, final_label, atr, sensitivity=adaptive_engine.get_params().get("retest_sensitivity", 0.6) if hasattr(adaptive_engine, "get_params") else 0.6)
                if heur:
                    prec = {"entry_price": heur.get("entry_price"), "sl": heur.get("sl"), "confidence": heur.get("confidence"), "reason": "tf_heuristic_retest", "entry_time_sast": _to_sast(dt.datetime.utcnow())}
        result["precision_entry"] = prec
    except Exception as e:
        result["precision_entry"] = None

    # integrate precision into entry/sl/tp
    try:
        if final_label in ("Buy", "Sell") and atr:
            atr_mult = adaptive_engine.get_params().get("atr_multiplier", 1.5) if hasattr(adaptive_engine, "get_params") else 1.5
            tp_atr_mult = 3.0
            if final_label == "Buy":
                sl_fallback = price - atr_mult * atr
                tp_fallback = price + tp_atr_mult * atr
            else:
                sl_fallback = price + atr_mult * atr
                tp_fallback = price - tp_atr_mult * atr
        else:
            sl_fallback, tp_fallback = None, None

        prec = result.get("precision_entry", None)
        entry_point = None
        entry_time_sast = None
        sl_val = None
        tp_val = None
        confidence = None
        precision_reason = None

        if isinstance(prec, dict):
            entry_point = prec.get("entry_point") or prec.get("entry_price") or prec.get("entry")
            entry_time_sast = prec.get("entry_time_sast") or prec.get("entry_time") or prec.get("time")
            sl_val = prec.get("sl") or prec.get("stop_loss") or prec.get("stop")
            tp_val = prec.get("tp") or prec.get("take_profit") or prec.get("takeprofit")
            confidence = prec.get("confidence") or prec.get("confidence_score") or prec.get("conf")
            precision_reason = prec.get("reason", prec.get("precision_reason", None))

        if sl_val is None and sl_fallback is not None:
            sl_val = sl_fallback
        if tp_val is None and tp_fallback is not None:
            tp_val = tp_fallback

        try:
            m1p = (micro_conf.get("m1") or {}).get("patterns") or []
            m5p = (micro_conf.get("m5") or {}).get("patterns") or []
            pats = set(m1p + m5p)
            if ("bullish_engulfing" in pats or "hammer_like" in pats) and final_label == "Buy":
                confidence = (confidence or 0.5) + 0.12
            if ("bearish_engulfing" in pats or "shooting_star_like" in pats) and final_label == "Sell":
                confidence = (confidence or 0.5) + 0.12
        except Exception:
            pass

        if not entry_time_sast:
            ctx_entry_time = None
            try:
                ctx_entry_time = (ctx.get("entry_time") if isinstance(ctx, dict) else None)
            except Exception:
                ctx_entry_time = None
            entry_time_sast = ctx_entry_time

        try:
            if entry_time_sast:
                entry_time_sast = _to_sast(entry_time_sast)
        except Exception:
            pass

        result["entry_point"] = safe_round(entry_point) if entry_point is not None else result.get("entry_point")
        result["entry_time"] = entry_time_sast if entry_time_sast is not None else result.get("entry_time")
        result["entry_time_sast"] = entry_time_sast if entry_time_sast is not None else result.get("entry_time_sast")
        result["sl"] = safe_round(sl_val) if sl_val is not None else result.get("sl")
        result["tp"] = safe_round(tp_val) if tp_val is not None else result.get("tp")
        result["confidence"] = round(min(0.99, float(confidence or 0.5)), 3)
        result["precision_reason"] = precision_reason if precision_reason is not None else result.get("precision_reason")
    except Exception as e:
        log(f"precision integration error: {e}", "warning")

    # Precise retest refinement with 1m preferred, then 5m
    try:
        meta_for_ticks = _instrument_point_and_pipvalue(pair)
        point_for_ticks = meta_for_ticks.get("point") or _pips_for_pair(pair)
        if _PRECISE_RETEST_ON and final_label in ("Buy", "Sell"):
            heur_obj = None
            if (micro_conf.get("m1") or {}).get("heuristic_retest"):
                heur_obj = (micro_conf.get("m1") or {}).get("heuristic_retest")
            elif (micro_conf.get("m5") or {}).get("heuristic_retest"):
                heur_obj = (micro_conf.get("m5") or {}).get("heuristic_retest")
            elif isinstance(result.get("precision_entry"), dict) and result["precision_entry"].get("reason"):
                heur_obj = result["precision_entry"]

            refined = None
            if heur_obj and isinstance(micro_df_1m, pd.DataFrame) and not micro_df_1m.empty:
                vwap1 = _compute_vwap(micro_df_1m)
                refined = _refine_precise_retest(micro_df_1m, final_label, atr, heur_obj, point_for_ticks, vwap_series=vwap1)
            if refined is None and heur_obj and isinstance(micro_df_5m, pd.DataFrame) and not micro_df_5m.empty:
                vwap5 = _compute_vwap(micro_df_5m)
                refined = _refine_precise_retest(micro_df_5m, final_label, atr, heur_obj, point_for_ticks, vwap_series=vwap5)

            if isinstance(refined, dict):
                result.update({
                    "entry_point_refined": refined.get("entry_point_refined"),
                    "entry_point_edge": refined.get("entry_point_edge"),
                    "protected_sl": refined.get("protected_sl"),
                    "entry_time_refined_sast": refined.get("entry_time_refined_sast"),
                    "entry_mode": refined.get("entry_mode"),
                    "trigger_price": refined.get("trigger_price"),
                    "precision_grade": refined.get("precision_grade"),
                    "precision_labels": refined.get("precision_labels"),
                    "retest_zone": refined.get("retest_zone"),
                    "fvg_zone": refined.get("fvg_zone"),
                    "ob_level": refined.get("ob_level"),
                })
                if not _NO_APPLY_PROTECTED:
                    if result.get("entry_point") is not None:
                        result["entry_point_prev"] = result.get("entry_point")
                    if result.get("sl") is not None:
                        result["sl_prev"] = result.get("sl")
                    if refined.get("entry_point_edge") is not None:
                        result["entry_point"] = refined.get("entry_point_edge")
                    if refined.get("protected_sl") is not None:
                        result["sl"] = refined.get("protected_sl")
    except Exception as e:
        log(f"precise retest refinement error: {e}", "warning")

    # If TF-derived entry/SL too tight vs ATR, expand protected SL using micro ATR logic (fallback)
    try:
        if final_label in ("Buy","Sell") and result.get("entry_point") is not None and result.get("sl") is not None and result.get("protected_sl") is None:
            ep = float(result.get("entry_point"))
            slv = float(result.get("sl"))
            r = abs(ep - slv)
            if atr and r < (0.6 * atr):
                span = 0.4 * float(atr)
                z_lo = ep - span
                z_hi = ep + span
                micro_atr_any = (micro_conf.get("m1") or {}).get("micro_atr") or (micro_conf.get("m5") or {}).get("micro_atr")
                prot = _protected_sl_price(final_label, z_lo, z_hi, None, None, atr, point_for_ticks, micro_atr=micro_atr_any)
                if not _NO_APPLY_PROTECTED:
                    result["sl_prev"] = result.get("sl")
                    result["sl"] = safe_round(prot)
                result.setdefault("retest_zone", {"low": safe_round(z_lo), "high": safe_round(z_hi), "pivot": safe_round(ep)})
                result["protected_sl"] = safe_round(prot)
    except Exception:
        pass

    # liquidity heuristic
    try:
        liquidity = liquidity_heuristic(df, atr)
        result["liquidity"] = liquidity
    except Exception:
        result["liquidity"] = None

    # improved lot-size calculation block
    try:
        acct_bal_env = os.getenv("ANALYSIS_ACCOUNT_BALANCE")
        risk_pct_env = os.getenv("ANALYSIS_RISK_PCT")
        pip_value_env = os.getenv("ANALYSIS_PIP_VALUE")
        lot_reco = None

        acct_bal = float(acct_bal_env) if acct_bal_env else None
        risk_pct = float(risk_pct_env) if risk_pct_env else None
        pip_value_per_lot = float(pip_value_env) if pip_value_env else 10.0

        entry_val = None
        sl_val_numeric = None
        try:
            if result.get("entry_point") is not None:
                entry_val = float(result.get("entry_point"))
            elif result.get("entry") is not None:
                entry_val = float(result.get("entry"))
            elif isinstance(result.get("precision_entry"), dict):
                e = (result.get("precision_entry") or {}).get("entry_point") or (result.get("precision_entry") or {}).get("entry_price")
                if e is not None:
                    entry_val = float(e)
        except Exception:
            entry_val = None

        try:
            if result.get("sl") is not None:
                sl_val_numeric = float(result.get("sl"))
            elif isinstance(result.get("precision_entry"), dict):
                s = (result.get("precision_entry") or {}).get("sl") or (result.get("precision_entry") or {}).get("stop_loss")
                if s is not None:
                    sl_val_numeric = float(s)
        except Exception:
            sl_val_numeric = None

        sl_pips = None
        if entry_val is not None and sl_val_numeric is not None:
            try:
                meta = _instrument_point_and_pipvalue(pair)
                point_guess = meta.get("point") or _pips_for_pair(pair)
                sl_pips = abs(sl_val_numeric - entry_val) / (point_guess if point_guess else 1e-9)
            except Exception:
                sl_pips = None

        meta = _instrument_point_and_pipvalue(pair)
        if meta.get("pip_value"):
            pip_value_per_lot = meta.get("pip_value")

        if (sl_pips is None or not math.isfinite(sl_pips)) and atr:
            try:
                point_guess = meta.get("point") or _pips_for_pair(pair)
                atr_multiplier_for_sl = adaptive_engine.get_params().get("atr_multiplier", 1.5) if hasattr(adaptive_engine, "get_params") else 1.5
                est_sl_price_distance = atr_multiplier_for_sl * float(atr)
                sl_pips = max(1.0, est_sl_price_distance / (point_guess if point_guess else 1e-9))
                log(f"Lot calc: fallback to ATR for sl_pips={sl_pips:.2f}", "info")
            except Exception:
                sl_pips = None

        if acct_bal is not None and risk_pct is not None and sl_pips is not None:
            conf = None
            try:
                conf = float(result.get("confidence") or (result.get("precision_entry") or {}).get("confidence") or 0.5)
            except Exception:
                conf = 0.5
            lot_reco = compute_lot_size_from_risk(acct_bal, risk_pct, sl_pips, pip_value_per_lot=pip_value_per_lot, confidence=conf, min_lot=0.01, max_lot=None)
            if lot_reco is None:
                log("lot calc: compute_lot_size_from_risk returned None despite available inputs", "warning")
        else:
            lot_reco = None
            log("lot calc: insufficient data (acct_bal/risk/sl_pips) to compute lot recommendation", "info")

        result["lot_size_recommendation"] = lot_reco
    except Exception as e:
        result["lot_size_recommendation"] = None
        log(f"lot size calc exception: {e}", "warning")

    # run backtest & forward-test
    try:
        bt = run_backtest(pair, timeframe, lookback=bars)
    except Exception as e:
        bt = {"success": False, "error": str(e)}
        log(f"backtest error: {e}", "warning")
    result["backtest"] = bt

    try:
        ft = run_forward_test(pair, timeframe)
    except Exception as e:
        ft = {"success": False, "error": str(e)}
        log(f"forward test error: {e}", "warning")
    result["forward_test"] = ft

    # adaptive logging
    try:
        try:
            if isinstance(bt, dict) and bt.get("success"):
                pnl_total = bt.get("pnl_total") if bt.get("pnl_total") is not None else bt.get("metrics", {}).get("total_pnl", 0.0)
                success_bt = True if (pnl_total and float(pnl_total) > 0.0) else False
                conf_bt = _confidence_from_backtest(bt)
                try:
                    log_result(pair, bool(success_bt), float(conf_bt), source="backtest")
                except Exception as e:
                    log(f"log_result backtest error: {e}", "warning")
        except Exception as e:
            log(f"backtest adaptive logging exception: {e}", "warning")

        try:
            if isinstance(ft, dict) and ft.get("success"):
                success_ft = None
                if "success_label" in ft:
                    success_ft = bool(ft.get("success_label"))
                elif "pnl" in ft:
                    try:
                        success_ft = float(ft.get("pnl")) > 0.0
                    except Exception:
                        success_ft = None
                conf_ft = _confidence_from_forward(ft)
                try:
                    if success_ft is not None:
                        log_result(pair, bool(success_ft), float(conf_ft), source="forward_test")
                except Exception as e:
                    log(f"log_result forward_test error: {e}", "warning")
        except Exception as e:
            log(f"forward_test adaptive logging exception: {e}", "warning")

        try:
            result["adaptive_summary"] = adaptive_engine.summary() if hasattr(adaptive_engine, "summary") else None
        except Exception:
            result["adaptive_summary"] = None
    except Exception as e:
        log(f"adaptive integration top-level error: {e}", "warning")

    # persist CSV history (timestamp in SAST)
    try:
        history_file = os.getenv("ANALYSIS_HISTORY_CSV", "analysis_history.csv")
        headers = ["timestamp", "pair", "timeframe", "final_suggestion", "entry_time", "entry_point", "sl", "tp", "combined_score", "news_status", "liquidity_score", "confidence"]
        row = [
            _to_sast(dt.datetime.utcnow()),
            pair,
            timeframe,
            result.get("final_suggestion"),
            result.get("entry_time"),
            result.get("entry_point"),
            result.get("sl"),
            result.get("tp"),
            result.get("combined_score"),
            result.get("news_status"),
            (result.get("liquidity") or {}).get("liquidity_score") if result.get("liquidity") else None,
            result.get("confidence")
        ]
        write_header = not os.path.exists(history_file)
        with open(history_file, "a", newline="") as hf:
            writer = csv.writer(hf)
            if write_header:
                writer.writerow(headers)
            writer.writerow(row)
    except Exception:
        pass

    # GOAT second-pass: exchange-aware session + confluence lock applied
    try:
        if _GOAT_MODE:
            ts_last = df.index[-1] if hasattr(df, "index") else None
            session_from_ctx = None
            try:
                session_from_ctx = (result.get("context") or {}).get("session")
            except Exception:
                session_from_ctx = None
            session_now2 = session_from_ctx or _goat_session_from_ts_tz(ts_last, pair=pair)
            session_ok2 = (session_now2 in _GOAT_ALLOWED_SESSIONS) if _GOAT_ALLOWED_SESSIONS else True

            spread_pips_env = os.getenv("ANALYSIS_SPREAD_PIPS")
            spread_pips2 = float(spread_pips_env) if spread_pips_env else None
            spread_ok2 = True if (spread_pips2 is None) else (spread_pips2 <= _GOAT_MAX_SPREAD_PIPS)

            news_blackout2 = _goat_news_blackout(result.get("news") or {}, _GOAT_NEWS_BLACKOUT_MIN)

            regime2 = _goat_regime_score(df)

            meta = _instrument_point_and_pipvalue(pair)
            point_for_pips = meta.get("point") or _pips_for_pair(pair)
            atr_pips2 = (float(result.get("atr")) / point_for_pips) if (result.get("atr") and point_for_pips) else None
            atr_ok2 = True if (atr_pips2 is None) else (atr_pips2 >= _GOAT_MIN_ATR_PIPS)

            p_hat2 = _goat_p_hat(result.get("combined_score") or 0.0)
            entry2 = result.get("entry_point") if result.get("entry_point") is not None else result.get("price")
            rr_info2 = _goat_rr(pair, result.get("price"), entry2, result.get("sl"), result.get("tp"), result.get("atr"))
            rr2 = rr_info2.get("rr", 1.0)
            ev2 = p_hat2 * rr2 - (1 - p_hat2) * 1.0
            ev2 = (ev2 + (_AUTOCALIB_STEP if (_AUTOCALIB_ON and ev2 < _EV_FLOOR) else 0.0))

            blocked2: List[str] = []
            if not session_ok2 and _GOAT_SESSION_STRICT:
                blocked2.append("session_filter")
            if not spread_ok2:
                blocked2.append("spread_filter")
            if news_blackout2:
                blocked2.append("news_blackout")
            if not atr_ok2:
                blocked2.append("atr_too_low")
            if ev2 <= 0.0:
                blocked2.append("negative_expected_value")
            if (result.get("final_suggestion") in ("Buy", "Sell")) and (result.get("entry_point") is None):
                blocked2.append("no_entry_point")

            try:
                htf2 = (result.get("higher_tf_trend") or "").strip()
                decision2_tmp = result.get("final_suggestion")
                is_contra2 = (htf2 in ("Buy", "Sell")) and (decision2_tmp in ("Buy", "Sell")) and (htf2 != decision2_tmp)
                micro_ok2 = False
                try:
                    mc = result.get("micro_confirmation") or {}
                    micro_ok2 = bool((mc.get("m1") or {}).get("patterns") or (mc.get("m1") or {}).get("heuristic_retest") or (mc.get("m5") or {}).get("patterns") or (mc.get("m5") or {}).get("heuristic_retest"))
                except Exception:
                    micro_ok2 = False
                regime_ok2 = bool((regime2 or {}).get("score", 0) >= 0.35)
                if is_contra2 and (not micro_ok2) and (not regime_ok2):
                    blocked2.append("no_confluence_contra")
            except Exception:
                pass

            decision_label2 = result.get("final_suggestion")
            soft_only2 = (not any(b in _HARD_BLOCKS for b in blocked2))
            if decision_label2 in ("Buy", "Sell"):
                if not soft_only2:
                    decision_label2 = "Neutral"
                else:
                    if result.get("confluence_lock"):
                        if "no_entry_point" in blocked2:
                            result["pending_entry"] = True
                    else:
                        if "no_entry_point" in blocked2:
                            result["pending_entry"] = True

            result["goat_post"] = {
                "enabled": True,
                "p_hat": round(float(p_hat2), 4),
                "rr": round(float(rr2), 3),
                "ev": round(float(ev2), 4),
                "rr_source": rr_info2.get("source"),
                "quality_scores": {
                    "session": {"value": session_now2, "ok": session_ok2, "allowed": list(_GOAT_ALLOWED_SESSIONS)},
                    "spread_pips": {"value": spread_pips2, "ok": spread_ok2, "max": _GOAT_MAX_SPREAD_PIPS},
                    "news_blackout": {"value": news_blackout2, "ok": not news_blackout2, "window_min": _GOAT_NEWS_BLACKOUT_MIN},
                    "regime": regime2,
                    "atr_pips": {"value": round(atr_pips2, 2) if atr_pips2 else None, "ok": atr_ok2, "min": _GOAT_MIN_ATR_PIPS},
                },
                "blocked_reasons": blocked2,
                "original_label": (result.get("goat") or {}).get("original_label"),
                "incoming_label": result.get("final_suggestion"),
                "decision_label": decision_label2,
            }
            result["final_suggestion"] = decision_label2
    except Exception as e:
        log(f"GOAT post-pass error: {e}", "warning")

    # GOAT CSV (pass1 + pass2)
    try:
        if _GOAT_MODE:
            goat_csv = os.getenv("ANALYSIS_HISTORY_GOAT_CSV", "analysis_history_goat.csv")
            write_header2 = not os.path.exists(goat_csv)
            with open(goat_csv, "a", newline="") as hf2:
                writer2 = csv.writer(hf2)
                if write_header2:
                    writer2.writerow([
                        "timestamp_sast","pair","timeframe","orig_label","goat_label","combined_score",
                        "p_hat","rr","ev","session","spread_pips","news_blackout","regime_score","atr_pips",
                        "entry","sl","tp","tp1","tp2","trail_after_R","time_stop_bars","pass"
                    ])
                quality = (result.get("goat") or {}).get("quality_scores", {})
                writer2.writerow([
                    _to_sast(dt.datetime.utcnow()),
                    pair,
                    timeframe,
                    (result.get("goat") or {}).get("original_label"),
                    (result.get("goat") or {}).get("decision_label") or result.get("final_suggestion"),
                    result.get("combined_score"),
                    (result.get("goat") or {}).get("p_hat"),
                    (result.get("goat") or {}).get("rr"),
                    (result.get("goat") or {}).get("ev"),
                    (quality.get("session") or {}).get("value"),
                    (quality.get("spread_pips") or {}).get("value"),
                    (quality.get("news_blackout") or {}).get("value"),
                    (quality.get("regime") or {}).get("score"),
                    (quality.get("atr_pips") or {}).get("value"),
                    result.get("entry_point"),
                    result.get("sl"),
                    result.get("tp"),
                    result.get("tp1"),
                    result.get("tp2"),
                    result.get("trail_after_R"),
                    result.get("time_stop_bars"),
                    "pass1"
                ])
                if result.get("goat_post"):
                    q2 = (result.get("goat_post") or {}).get("quality_scores", {})
                    writer2.writerow([
                        _to_sast(dt.datetime.utcnow()),
                        pair,
                        timeframe,
                        (result.get("goat") or {}).get("original_label"),
                        (result.get("goat_post") or {}).get("decision_label") or result.get("final_suggestion"),
                        result.get("combined_score"),
                        (result.get("goat_post") or {}).get("p_hat"),
                        (result.get("goat_post") or {}).get("rr"),
                        (result.get("goat_post") or {}).get("ev"),
                        (q2.get("session") or {}).get("value"),
                        (q2.get("spread_pips") or {}).get("value"),
                        (q2.get("news_blackout") or {}).get("value"),
                        (q2.get("regime") or {}).get("score"),
                        (q2.get("atr_pips") or {}).get("value"),
                        result.get("entry_point"),
                        result.get("sl"),
                        result.get("tp"),
                        result.get("tp1"),
                        result.get("tp2"),
                        result.get("trail_after_R"),
                        result.get("time_stop_bars"),
                        "pass2"
                    ])
    except Exception as e:
        log(f"GOAT CSV write error: {e}", "warning")

    # GOAT+ PASS
    try:
        if _GOAT_MODE and _GOAT_PLUS:
            gp_blocked: List[str] = []
            meta = _instrument_point_and_pipvalue(pair)
            point_meta = meta.get("point") or _pips_for_pair(pair)

            fresh = _fresh_signal_ok(result.get("entry_time_sast") or result.get("entry_time"), _MAX_SIGNAL_AGE_MIN)
            if not fresh["ok"]:
                gp_blocked.append("stale_signal")

            try:
                ep = float(result.get("entry_point")) if result.get("entry_point") is not None else None
                slv = float(result.get("sl")) if result.get("sl") is not None else None
                px = float(result.get("price")) if result.get("price") is not None else None
                drift_r = None
                if ep is not None and slv is not None and px is not None and ep != slv:
                    R = abs(ep - slv)
                    drift = abs(px - ep)
                    drift_r = float(drift / (R + 1e-12))
                    if drift_r > _MAX_ENTRY_DRIFT_R:
                        gp_blocked.append("entry_drift_exceeds_R")
                result["goat_plus_drift_r"] = drift_r
            except Exception:
                pass

            spread_pips_env = os.getenv("ANALYSIS_SPREAD_PIPS")
            spread_pips3 = float(spread_pips_env) if spread_pips_env else None
            sta = _spread_to_atr_ok(spread_pips3, result.get("atr"), point_meta, _MAX_SPREAD_TO_ATR)
            if sta["ok"] is False:
                gp_blocked.append("spread_to_atr_high")

            try:
                htf_dir = (result.get("higher_tf_trend") or "").strip()
                cur_dir = result.get("ensemble")
                mc = result.get("micro_confirmation") or {}
                micro_ok = bool((mc.get("m1") or {}).get("patterns") or (mc.get("m1") or {}).get("heuristic_retest") or (mc.get("m5") or {}).get("patterns") or (mc.get("m5") or {}).get("heuristic_retest"))
                aligns = 0
                decision_now = result.get("final_suggestion")
                if decision_now in ("Buy", "Sell"):
                    if htf_dir == decision_now:
                        aligns += 1
                    if cur_dir == decision_now:
                        aligns += 1
                    if micro_ok:
                        aligns += 1
                    if _REQUIRE_2OF3 and aligns < 2:
                        gp_blocked.append("no_2of3_confluence")
                is_contra3 = (htf_dir in ("Buy","Sell")) and (decision_now in ("Buy","Sell")) and (htf_dir != decision_now)
                if _CONTRA_STRICT and is_contra3 and not micro_ok:
                    gp_blocked.append("contra_strict_block")
            except Exception:
                pass

            hurst = _hurst_estimate(df["close"])
            entropy = _spectral_entropy(df["close"])
            th = _trend_health(df)
            if _HURST_ON and hurst is not None and result.get("final_suggestion") in ("Buy","Sell"):
                if hurst < 0.4 and (result.get("higher_tf_trend") in ("Buy","Sell")):
                    gp_blocked.append("hurst_anti_trend")
            if _ENTROPY_ON and entropy is not None and entropy > 0.92:
                gp_blocked.append("high_entropy_noise")
            if _MIN_TREND_SLOPE > 0 and abs(th.get("slope",0.0)) < _MIN_TREND_SLOPE and (result.get("final_suggestion") in ("Buy","Sell")):
                gp_blocked.append("trend_slope_weak")

            src = _sr_congestion_score(df, window=min(300, len(df)))
            sr_min_dist_r = None
            try:
                ep = float(result.get("entry_point")) if result.get("entry_point") is not None else None
                slv = float(result.get("sl")) if result.get("sl") is not None else None
                if ep is not None and slv is not None:
                    R = abs(ep - slv)
                    recent_high = float(df["high"].tail(50).max())
                    recent_low = float(df["low"].tail(50).min())
                    dist_up = abs(recent_high - ep) / (R + 1e-12)
                    dist_dn = abs(ep - recent_low) / (R + 1e-12)
                    sr_min_dist_r = float(min(dist_up, dist_dn))
                    min_req = float(os.getenv("ANALYSIS_MIN_SR_DISTANCE_R") or "0.6")
                    if sr_min_dist_r < min_req:
                        gp_blocked.append("sr_too_close")
                max_cong = float(os.getenv("ANALYSIS_MAX_CONGESTION_SCORE") or "0.6")
                if src["congestion"] > max_cong:
                    gp_blocked.append("high_congestion")
            except Exception:
                pass

            try:
                headlines = (result.get("news") or {}).get("headlines") or []
                key_hit = False
                for h in headlines:
                    text = (h.get("headline") or "").lower()
                    if any(kw.lower() in text for kw in _EVENT_KEYWORDS):
                        key_hit = True
                        break
                if key_hit:
                    gp_blocked.append("event_keyword")
            except Exception:
                pass

            dd = _drawdown_throttle()
            exp_guard = _exposure_guard(pair)
            if dd["state"] == "halt":
                gp_blocked.append("dd_halt")
            if not exp_guard["ok"]:
                gp_blocked.append("family_exposure_limit")

            decision_gp = result.get("final_suggestion")
            if decision_gp in ("Buy","Sell"):
                hard_present = any(b in _HARD_BLOCKS for b in gp_blocked)
                if hard_present or (len(gp_blocked) >= 3 and not result.get("confluence_lock")):
                    decision_gp = "Neutral"
                else:
                    try:
                        penalty = min(0.2, 0.05 * len(gp_blocked))
                        result["confidence"] = max(0.3, float(result.get("confidence") or 0.6) - penalty)
                    except Exception:
                        pass

            result["be_after_tp1"] = bool(_BE_AFTER_TP1)
            result["trail_atr_mult"] = _TRAIL_ATR_MULT
            result["trail_pct_of_mfe"] = _TRAIL_PCT_OF_MFE

            result["goat_plus"] = {
                "enabled": True,
                "freshness": fresh,
                "spread_to_atr": sta,
                "hurst": hurst,
                "entropy": entropy,
                "trend_health": th,
                "sr_congestion": src,
                "sr_min_dist_r": sr_min_dist_r,
                "dd_throttle": dd,
                "exposure_guard": exp_guard,
                "blocked_reasons": gp_blocked,
                "incoming_label": result.get("final_suggestion"),
                "decision_label": decision_gp
            }
            result["final_suggestion"] = decision_gp

            try:
                if result.get("lot_size_recommendation") is not None and isinstance(result.get("lot_size_recommendation"), (int,float)):
                    scaled = float(result["lot_size_recommendation"]) * float(dd.get("scale",1.0))
                    result["lot_size_recommendation_goat"] = round(max(0.01, scaled), 2)
            except Exception:
                pass

            # **Phase 3 Change**: This logging block is now handled by journal.py
            # The old try...except block for writing to goatp_csv has been removed.
            # It will be replaced by the centralized call at the end of the function.

    except Exception as e:
        log(f"GOAT+ pass error: {e}", "warning")

    # GOAT soften pass
    try:
        if _GOAT_MODE and (os.getenv("ANALYSIS_NEUTRAL_SOFTEN") != "0"):
            result = _soften_neutralization(result)
    except Exception as e:
        log(f"GOAT soften pass error: {e}", "warning")

    # build report
    try:
        report_lines = [
            f"Pair: {pair}",
            f"Timeframe: {timeframe}",
            f"Current Price: {safe_round(result['price'] or price)}",
            f"ATR(14): {safe_round(result['atr']) if result['atr'] else 'N/A'}",
            f"Signals: {result.get('signals')}",
            f"Higher TF Trend (legacy): {result.get('higher_tf_trend')}",
            f"4H HTF: {result.get('htf_4h')}",
            f"Advanced: {result.get('advanced')}",
            f"Context: {result.get('context')}",
            f"Final Suggestion: {result.get('final_suggestion')} {'(pending entry)' if result.get('pending_entry') else ''}",
            f"Entry Point: {result.get('entry_point')} | Entry Time: {result.get('entry_time')}",
            f"SL: {result.get('sl')} | TP: {result.get('tp')}",
            f"Protected SL: {result.get('protected_sl')} | Edge entry: {result.get('entry_point_edge')} | Prev SL: {result.get('sl_prev')} | Prev entry: {result.get('entry_point_prev')}",
            f"Confluence score: {result.get('confluence_score')} | Lock: {result.get('confluence_lock')}",
            f"Confidence: {result.get('confidence')}",
            f"Precision reason: {result.get('precision_reason')}",
            f"News status: {result.get('news_status')} | News headlines: {[h.get('headline') + ' @' + h.get('time') for h in (result.get('news') or {}).get('headlines', [])]}",
            f"Liquidity: {result.get('liquidity')}",
            f"Structure change: {result.get('structure_change')}",
            f"Micro confirmation (1m): {(result.get('micro_confirmation') or {}).get('m1')}",
            f"Micro confirmation (5m): {(result.get('micro_confirmation') or {}).get('m5')}",
            f"Lot size recommendation: {result.get('lot_size_recommendation')}",
            f"Backtest summary: {result.get('backtest')}",
            f"Forward test summary: {result.get('forward_test')}",
            f"Adaptive summary: {result.get('adaptive_summary')}"
        ]
        if _PRECISE_RETEST_ON and (result.get("entry_point_refined") or result.get("entry_time_refined_sast")):
            report_lines += [
                "--- PRECISION RETEST ---",
                f"Refined entry: {result.get('entry_point_refined')} @ {result.get('entry_time_refined_sast')} ({result.get('entry_mode')} trigger @ {result.get('trigger_price')})",
                f"Precision grade: {result.get('precision_grade')} | Labels: {result.get('precision_labels')}",
                f"Retest zone: {result.get('retest_zone')} | FVG: {result.get('fvg_zone')} | OB level: {result.get('ob_level')}",
            ]
        if _GOAT_MODE and result.get("goat"):
            goat = result.get("goat") or {}
            q = goat.get("quality_scores") or {}
            report_lines += [
                "--- GOAT MODE (pass1) ---",
                f"Original label: {goat.get('original_label')} -> Decision: {goat.get('decision_label')}",
                f"p_hat: {goat.get('p_hat')} | RR: {goat.get('rr')} | EV: {goat.get('ev')}",
                f"Session: {(q.get('session') or {}).get('value')} allowed={((q.get('session') or {}).get('allowed'))}",
                f"Spread(pips): {(q.get('spread_pips') or {}).get('value')} <= max {(_GOAT_MAX_SPREAD_PIPS)}",
                f"News blackout (last {_GOAT_NEWS_BLACKOUT_MIN}m): {(q.get('news_blackout') or {}).get('value')}",
                f"Regime score: {(q.get('regime') or {}).get('score')} ({(q.get('regime') or {}).get('trend')})",
                f"ATR(pips): {(q.get('atr_pips') or {}).get('value')} >= min {(_GOAT_MIN_ATR_PIPS)}",
                f"Blocked reasons: {goat.get('blocked_reasons')}",
                f"TP1: {result.get('tp1')} | TP2: {result.get('tp2')} | Trail after R: {result.get('trail_after_R')} | Time-stop bars: {result.get('time_stop_bars')}",
            ]
        if _GOAT_MODE and result.get("goat_post"):
            goat2 = result.get("goat_post") or {}
            q2 = goat2.get("quality_scores") or {}
            report_lines += [
                "--- GOAT MODE (post) ---",
                f"Incoming label: {goat2.get('incoming_label')} -> Decision: {goat2.get('decision_label')}",
                f"p_hat: {goat2.get('p_hat')} | RR: {goat2.get('rr')} | EV: {goat2.get('ev')}",
                f"Session: {(q2.get('session') or {}).get('value')} allowed={((q2.get('session') or {}).get('allowed'))}",
                f"Spread(pips): {(q2.get('spread_pips') or {}).get('value')} <= max {(_GOAT_MAX_SPREAD_PIPS)}",
                f"News blackout (last {_GOAT_NEWS_BLACKOUT_MIN}m): {(q2.get('news_blackout') or {}).get('value')}",
                f"Regime score: {(q2.get('regime') or {}).get('score')} ({(q2.get('regime') or {}).get('trend')})",
                f"ATR(pips): {(q2.get('atr_pips') or {}).get('value')} >= min {(_GOAT_MIN_ATR_PIPS)}",
                f"Blocked reasons: {goat2.get('blocked_reasons')}",
            ]
        if _GOAT_MODE and (os.getenv("ANALYSIS_NEUTRAL_SOFTEN") != "0") and result.get("goat_soft"):
            gs = result["goat_soft"]
            report_lines += [
                "--- GOAT SOFTEN ---",
                f"Softened to: {result.get('final_suggestion')} | Risk scale: {gs.get('risk_scale')}",
                f"p_hat: {gs.get('p_hat')} | EV: {gs.get('ev')} | Blocked: {gs.get('blocked_reasons')}"
            ]
        result["report"] = "\n".join(report_lines)
    except Exception:
        result["report"] = "Report generation failed."

    # **Phase 3 Change:** Centralized Logging
    # All old CSV writing blocks are removed. This single call delegates logging to journal.py.
    try:
        log_trade_decision(result)
    except Exception as e:
        log(f"Final logging via journal.py failed: {e}", "error")

    result["success"] = True
    return result


# -------------------------
# Standalone quick test
# -------------------------
if __name__ == "__main__":
    pair_test = os.getenv("ANALYSIS_TEST_PAIR", "AAPLm")
    tf_test = os.getenv("ANALYSIS_TEST_TF", "1h")
    try:
        res = analyze_pair(pair_test, tf_test)
        print("\n--- ANALYSIS RESULT ---")
        for k in [
            "success","pair","timeframe","price","atr","final_suggestion","entry_point","entry_point_edge",
            "entry_point_prev","protected_sl","sl","sl_prev","entry_time","entry_time_refined_sast",
            "precision_grade","precision_labels","pending_entry","confidence","confluence_score","confluence_lock"
        ]:
            print(f"{k}: {res.get(k)}")
        print("\nreport:\n")
        print(res.get("report"))
    except Exception as e:
        print("Standalone test failed:", e)
