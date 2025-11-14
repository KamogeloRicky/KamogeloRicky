"""
Robust backtest_engine for RickyFX Analysis Bot.

Replaces the lightweight prototype with a more realistic, configurable,
and compatible backtester + forward tester. Designed to be drop-in compatible:
- Exposes run_backtest(pair, timeframe, lookback=2000)
- Exposes run_forward_test(pair, timeframe, future_bars=100)

Core features:
- Realistic execution using OHLC per-bar high/low checks for TP/SL intrabar fills.
- Configurable spread, slippage and commission via environment variables.
- Position sizing by account balance & risk% (env vars or defaults).
- ATR-based SL/TP sizing with adaptive multiplier (reads adaptive_engine params if available).
- Detailed per-trade records (entry_index/time, exit_index/time, direction, entry_price, exit_price, pips, pnl).
- Robust metrics: total pnl, win rate, avg win/loss, expectancy, profit factor, Sharpe/Sortino proxies, max drawdown, final balance.
- Safe error handling and backward-compatible return shape (success, pair, timeframe, tested_trades, final_balance, pnl_total, trades, metrics).
- Deterministic (no randomness) unless environment variables request Monte Carlo sampling (off by default).

New in this version:
- GOAT filters enabled for backtest (session filter, spread cap, ATR pips minimum, optional EV gate).
- GOAT exits emulation: partial TP1, break-even after TP1, and trailing (configurable).
- Round-turn costs toggle (apply spread/slippage/commission at both entry and exit).
- Pluggable news calendar blackout (CSV), with high-impact window blocking.
- Confluence emulation: optional 4H alignment and simple micro confirmation checks.
- Drawdown throttle scaling for risk and optional trade halt at hard DD.
- Unified instrument metadata with analysis.py for consistent point (pip) and pip_value per lot.
- Optional CSV exporters for trades and equity curve.
"""

import os
import csv
import math
import traceback
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# optional tz for session logic
try:
    import pytz
    SAST_TZ = pytz.timezone("Africa/Johannesburg")
except Exception:
    SAST_TZ = None

# project imports (fetch_candles must be provided by data.py)
try:
    from data import fetch_candles
except Exception:
    fetch_candles = None

# indicator helpers - keep existing API usage
try:
    from utils import indicator_signals, ensemble_signal
except Exception:
    # minimal fallbacks (safe but conservative)
    def indicator_signals(df: pd.DataFrame) -> Dict[str, str]:
        return {}
    def ensemble_signal(signals: Dict[str, str], trend: Optional[str] = None) -> str:
        return "Neutral"

# adaptive params if present
try:
    from adaptive_learning import adaptive_engine
except Exception:
    adaptive_engine = None

# -------------------------
# Configurable execution params (via env)
# -------------------------
SPREAD_PIPS_DEFAULT = float(os.getenv("BACKTEST_SPREAD_PIPS", os.getenv("SPREAD_PIPS", "0.0")))
SLIPPAGE_PIPS = float(os.getenv("BACKTEST_SLIPPAGE_PIPS", os.getenv("SLIPPAGE_PIPS", "0.0")))
COMMISSION_PER_LOT = float(os.getenv("BACKTEST_COMMISSION_PER_LOT", os.getenv("COMMISSION_PER_LOT", "0.0")))
ACCOUNT_BALANCE_DEFAULT = float(os.getenv("BACKTEST_ACCOUNT_BALANCE", os.getenv("ANALYSIS_ACCOUNT_BALANCE") or 1000.0))
RISK_PCT_DEFAULT = float(os.getenv("BACKTEST_RISK_PCT", os.getenv("ANALYSIS_RISK_PCT") or 1.0))

# execution constraints
MAX_HOLD_BARS = int(os.getenv("BACKTEST_MAX_HOLD_BARS", "50"))   # horizon to close a trade if TP/SL not hit
MIN_LOOKBACK_BARS = 50

# -------------------------
# SAFE METRIC HELPERS (prevent ZeroDivisionError and NaNs)
# -------------------------
def _finite_or_none(x):
    try:
        xf = float(x)
        return xf if math.isfinite(xf) else None
    except Exception:
        return None

def _safe_ratio(numer, denom, default=None, allow_inf=False):
    try:
        n = float(numer)
        d = float(denom)
        if not math.isfinite(n) or not math.isfinite(d):
            return default
        if abs(d) < 1e-12:
            if allow_inf:
                if n > 0:
                    return float("inf")
                if n < 0:
                    return float("-inf")
            return default
        r = n / d
        return r if math.isfinite(r) else default
    except Exception:
        return default

def _std_safe(a, ddof=1):
    try:
        arr = np.asarray(a, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        if arr.size - ddof <= 0:
            # not enough samples for chosen ddof
            return 0.0
        s = float(np.std(arr, ddof=ddof))
        return s if math.isfinite(s) else None
    except Exception:
        return None

def _compute_sharpe_safe(returns: List[float], rf: float = 0.0) -> Optional[float]:
    try:
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        if r.size == 0:
            return None
        excess = r - rf
        mean_ex = float(np.mean(excess))
        std_ex = _std_safe(excess, ddof=1)
        return _safe_ratio(mean_ex, std_ex, default=None, allow_inf=False)
    except Exception:
        return None

def _compute_sortino_safe(returns: List[float], target: float = 0.0) -> Optional[float]:
    try:
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        if r.size == 0:
            return None
        excess = r - float(target)
        mean_ex = float(np.mean(excess))
        downside = excess[excess < 0.0]
        if downside.size == 0:
            # No downside: sortino is +/-inf by policy; return None for cleanliness
            return _safe_ratio(mean_ex, 0.0, default=None, allow_inf=True)
        std_dn = _std_safe(downside, ddof=1)
        return _safe_ratio(mean_ex, std_dn, default=None, allow_inf=True)
    except Exception:
        return None

# -------------------------
# Unify instrument meta with analysis.py
# -------------------------
def _instrument_point_and_pipvalue(pair: str) -> Dict[str, Optional[float]]:
    """
    Same mapping as analysis.py with env overrides ANALYSIS_POINT_SIZE and ANALYSIS_PIP_VALUE.
    Returns:
      - point: tick/pip size in price units (float or None)
      - pip_value: currency per pip for 1.0 lot (float or None)
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
    # US equities (suffix M frequently used in feeds)
    if any(p.startswith(t) for t in ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "NFLX", "AMD", "INTC"]) or p.endswith("M"):
        return {"point": 0.01, "pip_value": 1.0}
    # Default FX
    return {"point": None, "pip_value": None}

def _pips_for_pair(pair: str) -> float:
    p = (pair or "").upper()
    if p.endswith("JPY") or p.endswith("JPYC") or p.endswith("JPYc"):
        return 0.01
    return 0.0001

def _resolve_instrument_meta(pair: str) -> Tuple[float, float]:
    """
    Returns (point, pip_value_per_lot) with safe defaults if unknown.
    """
    meta = _instrument_point_and_pipvalue(pair)
    point = meta.get("point") if meta else None
    if not point:
        point = _pips_for_pair(pair)
    # pip value per lot: prefer env BACKTEST_PIP_VALUE/ANALYSIS_PIP_VALUE if set, else meta
    pv_env = os.getenv("BACKTEST_PIP_VALUE", os.getenv("ANALYSIS_PIP_VALUE") or "")
    if pv_env != "":
        pip_value_per_lot = float(pv_env)
    else:
        pip_value_per_lot = float(meta.get("pip_value")) if (meta and meta.get("pip_value")) else 10.0
    return float(point), float(pip_value_per_lot)

# -------------------------
# GOAT filter helpers for backtest
# -------------------------
# Enable/disable GOAT filters in backtest
BACKTEST_GOAT_FILTERS = os.getenv("BACKTEST_GOAT_FILTERS", "1") != "0"

# Allowed sessions, spread and ATR thresholds align with analysis.py
_ALLOWED_SESSIONS = tuple((os.getenv("BACKTEST_ALLOWED_SESSIONS", os.getenv("ANALYSIS_ALLOWED_SESSIONS") or "London,NY")).split(","))
_MAX_SPREAD_PIPS = float(os.getenv("BACKTEST_MAX_SPREAD_PIPS", os.getenv("ANALYSIS_MAX_SPREAD_PIPS") or "1.0"))
_MIN_ATR_PIPS = float(os.getenv("BACKTEST_MIN_ATR_PIPS", os.getenv("ANALYSIS_MIN_ATR_PIPS") or "0.0"))
_SESSION_STRICT = os.getenv("BACKTEST_SESSION_STRICT", os.getenv("ANALYSIS_SESSION_STRICT") or "0") == "1"
# Optional EV gate
_ENABLE_EV_GATE = os.getenv("BACKTEST_GOAT_EV", "0") == "1"

def _exchange_timezone(pair: str) -> Optional[str]:
    p = (pair or "").upper()
    if any(p.startswith(t) for t in ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "NFLX", "AMD", "INTC"]) or p.endswith("M"):
        return "America/New_York"
    return None

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
        # fallback: UTC approximation
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

def _goat_p_hat(combined_score: float) -> float:
    try:
        return float(max(0.01, min(0.99, 0.5 + 0.45 * math.tanh(combined_score))))
    except Exception:
        return 0.5

def _goat_rr(entry: Optional[float], sl: Optional[float], tp: Optional[float], atr: Optional[float]) -> float:
    try:
        if entry is not None and sl is not None and tp is not None and entry != sl:
            return float(abs(tp - entry) / (abs(entry - sl) + 1e-9))
        if atr is not None:
            return float(3.0 / 1.5)
    except Exception:
        pass
    return 1.0

def _apply_goat_filters(pair: str,
                        df: pd.DataFrame,
                        entry_idx: int,
                        atr: Optional[float],
                        point: float,
                        entry_price: float,
                        sl_price: float,
                        tp_price: float,
                        spread_pips: float) -> Dict[str, Any]:
    """
    Evaluate GOAT-style gates at a specific entry index. Returns dict:
      ok: bool
      reasons: list[str]
      session: str
      atr_pips: float|None
      ev: float|None
    """
    reasons: List[str] = []
    ts = df.index[entry_idx] if hasattr(df, "index") else None
    session_now = _goat_session_from_ts_tz(ts, pair=pair)
    session_ok = (session_now in _ALLOWED_SESSIONS) if _ALLOWED_SESSIONS else True

    # spread gate
    spread_ok = (spread_pips <= _MAX_SPREAD_PIPS)

    # atr pips gate
    atr_pips = (float(atr) / point) if (atr and point) else None
    atr_ok = True if (atr_pips is None) else (atr_pips >= _MIN_ATR_PIPS)

    # EV gate (optional; coarse since combined_score is not computed in BT; we use baseline 0.0)
    ev_ok = True
    ev_val = None
    if _ENABLE_EV_GATE:
        p_hat = _goat_p_hat(0.0)  # baseline without full combined_score
        rr = _goat_rr(entry_price, sl_price, tp_price, atr)
        ev_val = p_hat * rr - (1 - p_hat) * 1.0
        ev_ok = (ev_val > 0.0)

    # assemble gating
    if not spread_ok:
        reasons.append("spread_filter")
    if _SESSION_STRICT and not session_ok:
        reasons.append("session_filter")
    if not atr_ok:
        reasons.append("atr_too_low")
    if _ENABLE_EV_GATE and not ev_ok:
        reasons.append("negative_expected_value")

    ok = not reasons  # strict: any reason blocks in BT

    return {"ok": ok, "reasons": reasons, "session": session_now, "atr_pips": (round(atr_pips, 3) if atr_pips is not None else None), "ev": (round(ev_val, 4) if ev_val is not None else None)}

# -------------------------
# GOAT exits emulation (partial TP, BE, trailing)
# -------------------------
GOAT_EXITS_ON = os.getenv("BACKTEST_GOAT_EXITS", "1") != "0"
PARTIAL_TP_R = float(os.getenv("BACKTEST_PARTIAL_TP_R", os.getenv("ANALYSIS_PARTIAL_TP_R", "1.0")))
PARTIAL_SIZE = float(os.getenv("BACKTEST_PARTIAL_SIZE", "0.5"))  # fraction to close at TP1
BE_AFTER_TP1 = os.getenv("BACKTEST_BE_AFTER_TP1", os.getenv("ANALYSIS_BE_AFTER_TP1", "1")) != "0"
TRAIL_AFTER_R = float(os.getenv("BACKTEST_TRAIL_AFTER_R", os.getenv("ANALYSIS_TRAIL_AFTER_R", "1.5")))
TRAIL_ATR_MULT = float(os.getenv("BACKTEST_TRAIL_ATR_MULT", os.getenv("ANALYSIS_TRAIL_ATR_MULT", "1.0")))
TRAIL_PCT_OF_MFE = float(os.getenv("BACKTEST_TRAIL_PCT_OF_MFE", os.getenv("ANALYSIS_TRAIL_PCT_OF_MFE", "0.5")))
TP2_ATR_MULT = float(os.getenv("BACKTEST_TP2_ATR_MULT", "3.0"))

ROUND_TURN_COSTS = os.getenv("BACKTEST_ROUND_TURN", "0") == "1"

# -------------------------
# News blackout (pluggable CSV)
# -------------------------
NEWS_CSV_PATH = os.getenv("BACKTEST_NEWS_CSV", "")
NEWS_BLACKOUT_MIN = int(os.getenv("BACKTEST_NEWS_BLACKOUT_MIN", os.getenv("ANALYSIS_NEWS_BLACKOUT_MIN", "30")))
NEWS_IMPACT_LEVEL = os.getenv("BACKTEST_NEWS_IMPACT_LEVEL", "High").lower()  # level to block: High by default

def _load_news_csv(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return items
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get("time") or row.get("timestamp") or row.get("date")
                imp = (row.get("impact") or row.get("importance") or "Medium")
                if not t:
                    continue
                try:
                    ts = pd.to_datetime(t)
                except Exception:
                    continue
                items.append({"time": ts, "impact": str(imp)})
    except Exception:
        return []
    return items

def _news_blackout_at(ts: Any, pair: str, entries: List[Dict[str, Any]]) -> bool:
    try:
        t = pd.to_datetime(ts)
        if t is None or pd.isna(t):
            return False
        for it in entries:
            imp = str(it.get("impact", "Low")).lower()
            if NEWS_IMPACT_LEVEL == "high" and not imp.startswith("high"):
                continue
            t2 = it.get("time")
            if not isinstance(t2, pd.Timestamp):
                continue
            delta_min = abs((t - t2).total_seconds()) / 60.0
            if delta_min <= NEWS_BLACKOUT_MIN:
                return True
        return False
    except Exception:
        return False

# -------------------------
# Confluence emulation: 4H alignment and micro confirmation
# -------------------------
USE_4H = os.getenv("BACKTEST_USE_4H", os.getenv("ANALYSIS_USE_4H", "1")) != "0"
REQUIRE_4H_ALIGN = os.getenv("BACKTEST_REQUIRE_4H_ALIGN", "0") == "1"

USE_MICRO = os.getenv("BACKTEST_USE_MICRO", "1") != "0"
REQUIRE_MICRO_CONFIRM = os.getenv("BACKTEST_REQUIRE_MICRO_CONFIRM", "0") == "1"

def _trend_dir_4h(df_4h: pd.DataFrame, idx_time: Any) -> Optional[str]:
    try:
        if not isinstance(df_4h, pd.DataFrame) or df_4h.empty:
            return None
        # pick last 60 bars up to idx_time
        if "time" in df_4h.columns:
            dfh = df_4h[df_4h["time"] <= pd.to_datetime(idx_time)]
        else:
            dfh = df_4h[df_4h.index <= pd.to_datetime(idx_time)]
        if len(dfh) < 20:
            return None
        close = dfh["close"].astype(float).tail(80)
        if len(close) < 20:
            return None
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        if len(ma50.dropna()) == 0:
            return None
        spread = ma20.iloc[-1] - ma50.iloc[-1]
        if spread > 0:
            return "Buy"
        if spread < 0:
            return "Sell"
        return "Neutral"
    except Exception:
        return None

def _is_bullish_engulfing(df: pd.DataFrame) -> bool:
    try:
        if len(df) < 2:
            return False
        a = df.iloc[-2]; b = df.iloc[-1]
        return (b["close"] > b["open"]) and (a["close"] < a["open"]) and (b["close"] > a["open"]) and (b["open"] < a["close"])
    except Exception:
        return False

def _is_bearish_engulfing(df: pd.DataFrame) -> bool:
    try:
        if len(df) < 2:
            return False
        a = df.iloc[-2]; b = df.iloc[-1]
        return (b["close"] < b["open"]) and (a["close"] > a["open"]) and (b["open"] > a["close"]) and (b["close"] < a["open"])
    except Exception:
        return False

def _is_hammer_like(df: pd.DataFrame) -> bool:
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

def _is_shooting_star_like(df: pd.DataFrame) -> bool:
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

def _micro_confirm_ok(base_df: pd.DataFrame, direction: str) -> bool:
    """
    Simple micro confirmation proxy from base timeframe: engulfing or hammer/star at the entry bar.
    """
    try:
        if direction == "Buy":
            return _is_bullish_engulfing(base_df) or _is_hammer_like(base_df)
        if direction == "Sell":
            return _is_bearish_engulfing(base_df) or _is_shooting_star_like(base_df)
        return True
    except Exception:
        return True

# -------------------------
# Drawdown throttle (risk scaling) aligned with analysis envs
# -------------------------
DD_SOFT = float(os.getenv("BACKTEST_MAX_DD_SOFT", os.getenv("ANALYSIS_MAX_DD_SOFT", "0.06")))
DD_HARD = float(os.getenv("BACKTEST_MAX_DD_HARD", os.getenv("ANALYSIS_MAX_DD_HARD", "0.12")))
RISK_FLOOR = float(os.getenv("BACKTEST_RISK_FLOOR", os.getenv("ANALYSIS_RISK_FLOOR", "0.25")))

def _dd_throttle_from_equity(equity_series: List[float]) -> Dict[str, Any]:
    try:
        if not equity_series:
            return {"scale": 1.0, "dd": 0.0, "state": "no_data"}
        eq = float(equity_series[-1])
        peak = float(max(equity_series))
        if peak <= 0:
            return {"scale": 1.0, "dd": 0.0, "state": "no_data"}
        dd = max(0.0, (peak - eq) / peak)
        if dd >= DD_HARD:
            return {"scale": 0.0, "dd": dd, "state": "halt"}
        if dd <= DD_SOFT:
            return {"scale": 1.0, "dd": dd, "state": "ok"}
        span = DD_HARD - DD_SOFT
        frac = (dd - DD_SOFT) / (span + 1e-9)
        scale = max(RISK_FLOOR, 1.0 - frac)
        return {"scale": float(scale), "dd": float(dd), "state": "throttle"}
    except Exception:
        return {"scale": 1.0, "dd": 0.0, "state": "error"}

# -------------------------
# Helpers
# -------------------------
def _get_time_at_index(df: pd.DataFrame, idx: int):
    try:
        if "time" in df.columns:
            return df["time"].iloc[idx]
    except Exception:
        pass
    try:
        return df.index[idx]
    except Exception:
        return None

def _compute_atr_simple(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    try:
        import ta
        return float(ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=period).average_true_range().iloc[-1])
    except Exception:
        try:
            # fallback: average range
            return float(df["high"].subtract(df["low"]).tail(period).mean())
        except Exception:
            return None

def _pips_between(a: float, b: float, pip: float) -> float:
    return abs(a - b) / (pip if pip else 1e-9)

def _apply_spread_and_slippage(price: float, direction: str, spread_pips: float, slippage_pips: float, pip: float, for_exit: bool = False) -> float:
    """
    Adjust price to account for spread and slippage.
    For entry:
      - Buy: worse entry => add cost
      - Sell: worse entry => subtract cost
    For exit with ROUND_TURN_COSTS=True:
      - Buy: worse exit => subtract cost
      - Sell: worse exit => add cost
    """
    cost = (float(spread_pips) + float(slippage_pips)) * (pip if pip else 1e-9)
    if not for_exit:
        if direction == "Buy":
            return float(price) + cost
        else:
            return float(price) - cost
    else:
        if direction == "Buy":
            return float(price) - cost
        else:
            return float(price) + cost

# -------------------------
# Backtest core
# -------------------------
def run_backtest(pair: str, timeframe: str, lookback: int = 2000) -> Dict[str, Any]:
    """
    Run a detailed backtest simulation.
    Returns a dict with keys:
      success (bool), pair, timeframe, tested_trades (int), final_balance, pnl_total,
      trades (list of trade dicts), metrics (dict with win_rate, avg_win, avg_loss, expectancy, max_drawdown, etc.)
    """
    try:
        if fetch_candles is None:
            return {"success": False, "error": "fetch_candles not available (data.py missing)"}

        # fetch data
        try:
            df = fetch_candles(pair, timeframe, n=lookback)
        except TypeError:
            df = fetch_candles(pair, timeframe, lookback)
        if not isinstance(df, pd.DataFrame) or df.empty or len(df) < MIN_LOOKBACK_BARS:
            return {"success": False, "error": "Insufficient data for backtest"}

        # Ensure required columns present
        for c in ("open", "high", "low", "close"):
            if c not in df.columns:
                return {"success": False, "error": f"Missing required column in candles: {c}"}

        # unify meta with analysis.py
        point, pip_value_per_lot = _resolve_instrument_meta(pair)
        spread_pips = SPREAD_PIPS_DEFAULT  # per-entry used for gates and costs

        # use adaptive atr_multiplier if available
        atr_multiplier = 1.5
        if adaptive_engine is not None:
            try:
                atr_multiplier = float(adaptive_engine.get_params().get("atr_multiplier", atr_multiplier))
            except Exception:
                pass

        # prefetch 4H data if enabled
        df_4h = None
        if USE_4H:
            try:
                try:
                    df_4h = fetch_candles(pair, "4h", n=600)
                except TypeError:
                    df_4h = fetch_candles(pair, "4h", 600)
            except Exception:
                df_4h = None

        # load news events if any
        news_events = _load_news_csv(NEWS_CSV_PATH) if NEWS_CSV_PATH else []

        # account and risk
        starting_balance = float(ACCOUNT_BALANCE_DEFAULT)
        balance = float(starting_balance)
        risk_pct_base = float(RISK_PCT_DEFAULT) / 100.0

        trades: List[Dict[str, Any]] = []
        equity_curve = [starting_balance]
        current_idx = max(50, int(len(df) * 0.05))  # allow some warmup

        # iterate bars and open trades when ensemble produces a directional signal
        while current_idx < len(df) - 1:
            window = df.iloc[:current_idx + 1]
            try:
                signals = indicator_signals(window) or {}
            except Exception:
                signals = {}
            try:
                suggestion = ensemble_signal(signals, trend=None) or "Neutral"
            except Exception:
                suggestion = "Neutral"

            # only act on Buy/Sell (ignore Neutral)
            if suggestion not in ("Buy", "Sell"):
                current_idx += 1
                continue

            # prepare trade entry at next bar open if possible
            entry_idx = current_idx + 1
            if entry_idx >= len(df):
                break
            entry_bar = df.iloc[entry_idx]
            entry_time = _get_time_at_index(df, entry_idx)
            entry_price_raw = float(entry_bar["open"]) if "open" in entry_bar.index else float(entry_bar["close"])

            # news blackout gate
            if news_events and _SESSION_STRICT:
                if _news_blackout_at(entry_time, pair, news_events):
                    current_idx += 1
                    continue

            # compute ATR from window
            atr = _compute_atr_simple(window, period=14)
            if atr is None or not np.isfinite(atr) or atr <= 0:
                try:
                    atr = float(np.std(window["close"].diff().dropna().tail(14)))
                except Exception:
                    atr = 0.0

            # default SL/TP
            if suggestion == "Buy":
                sl_price = entry_price_raw - atr_multiplier * atr
                tp2_price = entry_price_raw + TP2_ATR_MULT * atr
            else:
                sl_price = entry_price_raw + atr_multiplier * atr
                tp2_price = entry_price_raw - TP2_ATR_MULT * atr

            # GOAT gating (session, spread, atr_pips, optional EV)
            goat_gate = {"ok": True, "reasons": []}
            if BACKTEST_GOAT_FILTERS:
                goat_gate = _apply_goat_filters(pair, df, entry_idx, atr, point, entry_price_raw, sl_price, tp2_price, spread_pips)
                if not goat_gate["ok"]:
                    # Skip this opportunity due to GOAT gating
                    current_idx += 1
                    continue

            # 4H confluence (optional)
            if REQUIRE_4H_ALIGN and isinstance(df_4h, pd.DataFrame) and entry_time is not None:
                dir4h = _trend_dir_4h(df_4h, entry_time)
                if dir4h in ("Buy", "Sell") and dir4h != suggestion:
                    current_idx += 1
                    continue

            # Micro confirmation (simple proxy)
            if REQUIRE_MICRO_CONFIRM:
                base_slice = df.iloc[max(0, entry_idx-2):entry_idx+1]
                if not _micro_confirm_ok(base_slice, suggestion):
                    current_idx += 1
                    continue

            # compute sl in pips to derive lot size
            sl_pips = _pips_between(entry_price_raw, sl_price, point) if (sl_price is not None) else None

            # drawdown throttle (scale risk)
            throttle = _dd_throttle_from_equity(equity_curve)
            if throttle.get("state") == "halt":
                # skip trading at hard DD
                current_idx += 1
                continue
            risk_pct = risk_pct_base * float(throttle.get("scale", 1.0))

            # determine lot size from balance and risk
            lot = 0.01
            try:
                if sl_pips and sl_pips > 0:
                    risk_amount = balance * risk_pct
                    lot = max(0.01, round(risk_amount / (sl_pips * pip_value_per_lot), 2))
            except Exception:
                lot = 0.01

            # effective entry price includes spread/slippage costs
            effective_entry = _apply_spread_and_slippage(entry_price_raw, suggestion, spread_pips, SLIPPAGE_PIPS, point, for_exit=False)

            # commission cost (per lot, charged once here; double if ROUND_TURN_COSTS for exit too)
            commission_entry = COMMISSION_PER_LOT * lot

            # TP1/TP2 setup
            R = abs(effective_entry - sl_price)
            if suggestion == "Buy":
                tp1_price = effective_entry + PARTIAL_TP_R * R
            else:
                tp1_price = effective_entry - PARTIAL_TP_R * R

            # trailing state
            partial_taken = False
            remaining_lot = float(lot)
            highest = effective_entry
            lowest = effective_entry
            exit_idx = None
            exit_price = None
            exit_reason = "timeout"
            pnl_value_net = 0.0  # accumulate partials

            max_exit_idx = min(len(df) - 1, entry_idx + MAX_HOLD_BARS)
            for j in range(entry_idx, max_exit_idx + 1):
                bar = df.iloc[j]
                high = float(bar["high"])
                low = float(bar["low"])

                # update extremes
                if high > highest:
                    highest = high
                if low < lowest:
                    lowest = low

                # Check order of hits inside bar for conservative fill logic:
                # We approximate: for Buy, SL first if both touched; for Sell, SL first if both touched.
                if suggestion == "Buy":
                    sl_hit = (low <= sl_price)
                    tp1_hit = (not partial_taken) and (high >= tp1_price) if GOAT_EXITS_ON else False
                    tp2_hit = (high >= tp2_price)
                    # order resolution
                    if sl_hit:
                        exit_idx = j
                        px = sl_price
                        if ROUND_TURN_COSTS:
                            px = _apply_spread_and_slippage(px, suggestion, spread_pips, SLIPPAGE_PIPS, point, for_exit=True)
                        exit_price = px
                        exit_reason = "sl"
                        # close all remaining
                        pnl_price = float(exit_price) - float(effective_entry)
                        pnl_pips = pnl_price / (point if point else 1e-9)
                        pnl_value = pnl_pips * pip_value_per_lot * remaining_lot
                        # commission at exit if round-turn
                        commission_exit = COMMISSION_PER_LOT * remaining_lot if ROUND_TURN_COSTS else 0.0
                        pnl_value_net += (pnl_value - commission_exit - (commission_entry if (j == entry_idx) else 0.0 if pnl_value_net != 0 else 0.0))
                        break
                    if GOAT_EXITS_ON and tp1_hit:
                        # take partial
                        px = tp1_price
                        if ROUND_TURN_COSTS:
                            px = _apply_spread_and_slippage(px, suggestion, spread_pips, SLIPPAGE_PIPS, point, for_exit=True)
                        # realize partial
                        part_lot = max(0.0, min(remaining_lot, PARTIAL_SIZE * lot))
                        pnl_price = float(px) - float(effective_entry)
                        pnl_pips = pnl_price / (point if point else 1e-9)
                        pnl_value = pnl_pips * pip_value_per_lot * part_lot
                        commission_exit = COMMISSION_PER_LOT * part_lot if ROUND_TURN_COSTS else 0.0
                        pnl_value_net += (pnl_value - commission_exit - (commission_entry if pnl_value_net == 0.0 else 0.0))
                        remaining_lot = max(0.0, remaining_lot - part_lot)
                        partial_taken = True
                        # move SL to BE if configured
                        if BE_AFTER_TP1:
                            sl_price = effective_entry
                    # trailing for remaining position after threshold
                    if GOAT_EXITS_ON and remaining_lot > 0:
                        # start trailing when price moves >= TRAIL_AFTER_R*R in favorable direction
                        if (highest - effective_entry) >= TRAIL_AFTER_R * R:
                            # trailing stop = max(current SL, highest - trail_offset)
                            trail_off = max(TRAIL_ATR_MULT * (atr if atr else 0.0), TRAIL_PCT_OF_MFE * max(0.0, (highest - effective_entry)))
                            new_sl = highest - trail_off
                            sl_price = max(sl_price, new_sl)
                    if tp2_hit:
                        exit_idx = j
                        px = tp2_price
                        if ROUND_TURN_COSTS:
                            px = _apply_spread_and_slippage(px, suggestion, spread_pips, SLIPPAGE_PIPS, point, for_exit=True)
                        exit_price = px
                        exit_reason = "tp2"
                        # realize remaining
                        pnl_price = float(exit_price) - float(effective_entry)
                        pnl_pips = pnl_price / (point if point else 1e-9)
                        pnl_value = pnl_pips * pip_value_per_lot * remaining_lot
                        commission_exit = COMMISSION_PER_LOT * remaining_lot if ROUND_TURN_COSTS else 0.0
                        pnl_value_net += (pnl_value - commission_exit - (commission_entry if pnl_value_net == 0.0 else 0.0))
                        remaining_lot = 0.0
                        break
                else:
                    # Sell
                    sl_hit = (high >= sl_price)
                    tp1_hit = (not partial_taken) and (low <= tp1_price) if GOAT_EXITS_ON else False
                    tp2_hit = (low <= tp2_price)
                    if sl_hit:
                        exit_idx = j
                        px = sl_price
                        if ROUND_TURN_COSTS:
                            px = _apply_spread_and_slippage(px, suggestion, spread_pips, SLIPPAGE_PIPS, point, for_exit=True)
                        exit_price = px
                        exit_reason = "sl"
                        pnl_price = float(effective_entry) - float(exit_price)
                        pnl_pips = pnl_price / (point if point else 1e-9)
                        pnl_value = pnl_pips * pip_value_per_lot * remaining_lot
                        commission_exit = COMMISSION_PER_LOT * remaining_lot if ROUND_TURN_COSTS else 0.0
                        pnl_value_net += (pnl_value - commission_exit - (commission_entry if (j == entry_idx) else 0.0 if pnl_value_net != 0 else 0.0))
                        break
                    if GOAT_EXITS_ON and tp1_hit:
                        px = tp1_price
                        if ROUND_TURN_COSTS:
                            px = _apply_spread_and_slippage(px, suggestion, spread_pips, SLIPPAGE_PIPS, point, for_exit=True)
                        part_lot = max(0.0, min(remaining_lot, PARTIAL_SIZE * lot))
                        pnl_price = float(effective_entry) - float(px)
                        pnl_pips = pnl_price / (point if point else 1e-9)
                        pnl_value = pnl_pips * pip_value_per_lot * part_lot
                        commission_exit = COMMISSION_PER_LOT * part_lot if ROUND_TURN_COSTS else 0.0
                        pnl_value_net += (pnl_value - commission_exit - (commission_entry if pnl_value_net == 0.0 else 0.0))
                        remaining_lot = max(0.0, remaining_lot - part_lot)
                        partial_taken = True
                        if BE_AFTER_TP1:
                            sl_price = effective_entry
                    if GOAT_EXITS_ON and remaining_lot > 0:
                        if (effective_entry - lowest) >= TRAIL_AFTER_R * R:
                            trail_off = max(TRAIL_ATR_MULT * (atr if atr else 0.0), TRAIL_PCT_OF_MFE * max(0.0, (effective_entry - lowest)))
                            new_sl = lowest + trail_off
                            sl_price = min(sl_price, new_sl)
                    if tp2_hit:
                        exit_idx = j
                        px = tp2_price
                        if ROUND_TURN_COSTS:
                            px = _apply_spread_and_slippage(px, suggestion, spread_pips, SLIPPAGE_PIPS, point, for_exit=True)
                        exit_price = px
                        exit_reason = "tp2"
                        pnl_price = float(effective_entry) - float(exit_price)
                        pnl_pips = pnl_price / (point if point else 1e-9)
                        pnl_value = pnl_pips * pip_value_per_lot * remaining_lot
                        commission_exit = COMMISSION_PER_LOT * remaining_lot if ROUND_TURN_COSTS else 0.0
                        pnl_value_net += (pnl_value - commission_exit - (commission_entry if pnl_value_net == 0.0 else 0.0))
                        remaining_lot = 0.0
                        break

            # if no TP/SL hit, exit at close of last bar considered (on remaining size)
            if exit_idx is None:
                exit_idx = max_exit_idx
                exit_bar = df.iloc[exit_idx]
                raw_exit = float(exit_bar["close"])
                px = raw_exit
                if ROUND_TURN_COSTS:
                    px = _apply_spread_and_slippage(px, suggestion, spread_pips, SLIPPAGE_PIPS, point, for_exit=True)
                exit_price = px
                exit_reason = "timeout"
                # realize remaining
                if remaining_lot > 0:
                    if suggestion == "Buy":
                        pnl_price = float(exit_price) - float(effective_entry)
                    else:
                        pnl_price = float(effective_entry) - float(exit_price)
                    pnl_pips = pnl_price / (point if point else 1e-9)
                    pnl_value = pnl_pips * pip_value_per_lot * remaining_lot
                    commission_exit = COMMISSION_PER_LOT * remaining_lot if ROUND_TURN_COSTS else 0.0
                    pnl_value_net += (pnl_value - commission_exit - (commission_entry if pnl_value_net == 0.0 else 0.0))

            # update balance
            balance += pnl_value_net
            equity_curve.append(balance)

            # record trade
            trade_record = {
                "entry_index": int(entry_idx),
                "entry_time": entry_time,
                "exit_index": int(exit_idx),
                "exit_time": _get_time_at_index(df, exit_idx),
                "direction": suggestion,
                "entry_price": round(entry_price_raw, 6),
                "effective_entry": round(effective_entry, 6),
                "exit_price": round(exit_price, 6) if exit_price is not None else None,
                "exit_reason": exit_reason,
                "sl_price_final": round(sl_price, 6) if sl_price is not None else None,
                "tp1_price": round(tp1_price, 6) if GOAT_EXITS_ON else None,
                "tp2_price": round(tp2_price, 6),
                "partial_taken": bool(partial_taken),
                "lot": float(lot),
                "remaining_lot_after": float(0.0 if exit_reason in ("sl", "tp2", "timeout") else remaining_lot),
                "commission_entry": round(commission_entry, 6),
                "pnl_value": round(pnl_value_net, 6),
                "dd_state": (_dd_throttle_from_equity(equity_curve).get("state")),
            }
            if BACKTEST_GOAT_FILTERS:
                trade_record.update({
                    "goat_session": goat_gate.get("session"),
                    "goat_atr_pips": goat_gate.get("atr_pips"),
                    "goat_ev": goat_gate.get("ev"),
                })
            if REQUIRE_4H_ALIGN and isinstance(df_4h, pd.DataFrame):
                trade_record["dir_4h_at_entry"] = _trend_dir_4h(df_4h, entry_time)
            if REQUIRE_MICRO_CONFIRM:
                trade_record["micro_confirm"] = _micro_confirm_ok(df.iloc[max(0, entry_idx-2):entry_idx+1], suggestion)
            trades.append(trade_record)

            # advance index beyond the exit to avoid overlapping immediate re-entry
            current_idx = max(current_idx + 1, exit_idx + 1)

        # compute metrics
        pnl_values = [float(t["pnl_value"]) for t in trades]
        wins = [p for p in pnl_values if p > 0]
        losses = [p for p in pnl_values if p <= 0]
        total_pnl = float(sum(pnl_values)) if pnl_values else 0.0
        tested_trades = len(trades)
        win_rate = (len(wins) / tested_trades * 100.0) if tested_trades else 0.0
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(losses) / len(losses)) if losses else 0.0

        # expectancy = (win_prob * avg_win + loss_prob * avg_loss) per trade
        expectancy = 0.0
        if tested_trades:
            expectancy = ((len(wins) / tested_trades) * avg_win) + ((len(losses) / tested_trades) * avg_loss)

        # compute equity curve and max drawdown
        peak = equity_curve[0] if equity_curve else 0.0
        max_dd = 0.0
        for v in equity_curve:
            if v > peak:
                peak = v
            dd = (peak - v)
            if dd > max_dd:
                max_dd = dd
        max_dd_pct = (max_dd / peak * 100.0) if peak > 0 else 0.0

        # profit factor
        gross_win = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

        # Sharpe/Sortino proxies using per-trade returns vs previous equity (SAFE)
        returns = []
        try:
            for i in range(1, len(equity_curve)):
                prev = equity_curve[i-1]
                r = (equity_curve[i] - prev) / (prev if prev else 1e-9)
                returns.append(r)
        except Exception:
            returns = []
        sharpe_val = _compute_sharpe_safe(returns, rf=0.0)
        sortino_val = _compute_sortino_safe(returns, target=0.0)

        metrics = {
            "tested_trades": tested_trades,
            "trade_count": tested_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_percent": round(win_rate, 2),
            "avg_win": round(avg_win, 6),
            "avg_loss": round(avg_loss, 6),
            "total_pnl": round(total_pnl, 6),
            "expectancy": round(expectancy, 6),
            "profit_factor": _finite_or_none(profit_factor),
            "max_drawdown": round(max_dd, 6),
            "max_dd_percent": round(max_dd_pct, 4),
            "sharpe": _finite_or_none(sharpe_val),
            "sortino": _finite_or_none(sortino_val),
            "final_balance": round(balance, 6),
        }

        settings = {
            "pair": pair,
            "timeframe": timeframe,
            "lookback": lookback,
            "spread_pips": spread_pips,
            "slippage_pips": SLIPPAGE_PIPS,
            "commission_per_lot": COMMISSION_PER_LOT,
            "account_balance_start": starting_balance,
            "risk_pct": RISK_PCT_DEFAULT,
            "pip_value_per_lot": pip_value_per_lot,
            "point": point,
            "atr_multiplier": atr_multiplier,
            "max_hold_bars": MAX_HOLD_BARS,
            "goat_filters": bool(BACKTEST_GOAT_FILTERS),
            "goat_allowed_sessions": list(_ALLOWED_SESSIONS),
            "goat_max_spread_pips": _MAX_SPREAD_PIPS,
            "goat_min_atr_pips": _MIN_ATR_PIPS,
            "goat_session_strict": _SESSION_STRICT,
            "goat_ev_gate": _ENABLE_EV_GATE,
            "goat_exits_on": bool(GOAT_EXITS_ON),
            "partial_tp_r": PARTIAL_TP_R,
            "partial_size": PARTIAL_SIZE,
            "be_after_tp1": bool(BE_AFTER_TP1),
            "trail_after_R": TRAIL_AFTER_R,
            "trail_atr_mult": TRAIL_ATR_MULT,
            "trail_pct_of_mfe": TRAIL_PCT_OF_MFE,
            "tp2_atr_mult": TP2_ATR_MULT,
            "round_turn_costs": bool(ROUND_TURN_COSTS),
            "news_csv": NEWS_CSV_PATH,
            "news_blackout_min": NEWS_BLACKOUT_MIN,
            "use_4h": bool(USE_4H),
            "require_4h_align": bool(REQUIRE_4H_ALIGN),
            "use_micro": bool(USE_MICRO),
            "require_micro_confirm": bool(REQUIRE_MICRO_CONFIRM),
            "dd_soft": DD_SOFT,
            "dd_hard": DD_HARD,
            "risk_floor": RISK_FLOOR,
        }

        # CSV exporters
        try:
            trades_csv = os.getenv("BACKTEST_TRADES_CSV", "")
            if trades_csv:
                write_header = not os.path.exists(trades_csv)
                with open(trades_csv, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["entry_time","exit_time","dir","entry","effective_entry","exit","reason","tp1","tp2","sl_final","lot","pnl_value"])
                    for t in trades:
                        w.writerow([
                            t.get("entry_time"),
                            t.get("exit_time"),
                            t.get("direction"),
                            t.get("entry_price"),
                            t.get("effective_entry"),
                            t.get("exit_price"),
                            t.get("exit_reason"),
                            t.get("tp1_price"),
                            t.get("tp2_price"),
                            t.get("sl_price_final"),
                            t.get("lot"),
                            t.get("pnl_value"),
                        ])
        except Exception:
            pass

        try:
            equity_csv = os.getenv("BACKTEST_EQUITY_CSV", "")
            if equity_csv and equity_curve:
                write_header = not os.path.exists(equity_csv)
                with open(equity_csv, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["step","equity"])
                    for i, v in enumerate(equity_curve):
                        w.writerow([i, round(v, 6)])
        except Exception:
            pass

        return {
            "success": True,
            "pair": pair,
            "timeframe": timeframe,
            "tested_trades": tested_trades,
            "final_balance": round(balance, 6),
            "pnl_total": round(total_pnl, 6),
            "trades": trades,
            "metrics": metrics,
            "settings": settings
        }

    except Exception as e:
        tb = traceback.format_exc()
        return {"success": False, "error": f"backtest error: {e}", "trace": tb}

# -------------------------
# Forward test (quick simulation)
# -------------------------
def run_forward_test(pair: str, timeframe: str, future_bars: int = 100) -> Dict[str, Any]:
    """
    Forward test: simulate applying current strategy to the most recent bars and report pnl over future_bars.

    Returns:
      success, pair, timeframe, suggestion, price_now, price_future, pnl, details (basic)
    """
    try:
        if fetch_candles is None:
            return {"success": False, "error": "fetch_candles not available (data.py missing)"}

        # fetch enough bars (warmup + future)
        lookback = max(500, future_bars + 200)
        try:
            df = fetch_candles(pair, timeframe, n=lookback)
        except TypeError:
            df = fetch_candles(pair, timeframe, lookback)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {"success": False, "error": "No data for forward test"}

        # compute current signal
        try:
            signals = indicator_signals(df) or {}
            suggestion = ensemble_signal(signals, trend=None) or "Neutral"
        except Exception:
            suggestion = "Neutral"

        price_now = float(df["close"].iloc[-1])

        # simple forward projection: linear trend estimate on recent bars
        recent = df["close"].tail(min(30, len(df)))
        if len(recent) >= 2:
            slope = (float(recent.iloc[-1]) - float(recent.iloc[0])) / max(1, (len(recent) - 1))
            price_future = round(price_now + slope * future_bars, 6)
        else:
            price_future = price_now

        # compute pnl in price units
        point, pip_value_per_lot = _resolve_instrument_meta(pair)
        if suggestion == "Buy":
            pnl_price = float(price_future) - float(price_now)
        elif suggestion == "Sell":
            pnl_price = float(price_now) - float(price_future)
        else:
            pnl_price = 0.0

        pnl_pips = pnl_price / (point if point else 1e-9)
        # one-lot evaluation for forward test value
        pnl_value = pnl_pips * pip_value_per_lot * 1.0

        return {
            "success": True,
            "pair": pair,
            "timeframe": timeframe,
            "suggestion": suggestion,
            "price_now": round(price_now, 6),
            "price_future": round(price_future, 6),
            "pnl_price": round(pnl_price, 6),
            "pnl_pips": round(pnl_pips, 2),
            "pnl": round(pnl_value, 6),
            "success_label": bool(pnl_value > 0.0)
        }
    except Exception as e:
        tb = traceback.format_exc()
        return {"success": False, "error": f"forward_test error: {e}", "trace": tb}
#backtest_engine.py
