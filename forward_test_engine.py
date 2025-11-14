"""
Robust forward_test_engine for RickyFX Analysis Bot.

Replaces the lightweight prototype with a more reliable, configurable forward-testing module.
Goals:
- Deterministic, explainable forward projections by default (no randomness).
- Optional Monte‑Carlo simulations (configurable) with aggregate statistics.
- Return pnl in price/pips/account-currency and an explicit 'success_label' (pnl > 0).
- Provide confidence metrics compatible with adaptive_learning ingestion.
- Preserve public function signature: run_forward_test(pair, timeframe, prediction_horizon=50)

Enhancements aligned with analysis.py and backtest_engine.py:
- Unified instrument metadata (point and pip_value per lot) identical to analysis.py.
- GOAT filters (session, spread cap, ATR pips minimum, optional EV gate).
- Confluence emulation: 4H alignment and simple micro confirmation proxy.
- Pluggable news blackout via CSV and time window.
- Cost model: spread, slippage, and optional round‑turn commission at exit too.
- SAST timestamp in outputs; optional CSV export of a forward summary row.

Environment variables (key ones):
- FORWARD_SIMULATIONS (int, default 0) — number of Monte Carlo runs (0 = deterministic)
- FORWARD_SIM_RANDOM_SEED (int) — RNG seed
- FORWARD_PRED_HORIZON (int) — overrides function arg if >0
- BACKTEST_SPREAD_PIPS / SPREAD_PIPS (float) — spread in pips (used for cost and GOAT spread gate)
- BACKTEST_SLIPPAGE_PIPS / SLIPPAGE_PIPS (float) — slippage in pips
- BACKTEST_COMMISSION_PER_LOT / COMMISSION_PER_LOT (float) — commission per lot (entry; exit too if round‑turn)
- BACKTEST_ROUND_TURN=1 — apply spread/slippage/commission on both entry and exit when computing forward pnl
- ANALYSIS_POINT_SIZE / ANALYSIS_PIP_VALUE — instrument overrides
- ANALYSIS_ALLOWED_SESSIONS / BACKTEST_ALLOWED_SESSIONS — GOAT sessions (e.g., "London,NY")
- ANALYSIS_MAX_SPREAD_PIPS / BACKTEST_MAX_SPREAD_PIPS — GOAT spread gate
- ANALYSIS_MIN_ATR_PIPS / BACKTEST_MIN_ATR_PIPS — GOAT min ATR(pips)
- ANALYSIS_SESSION_STRICT / BACKTEST_SESSION_STRICT=1 — enforce session/news blackout
- BACKTEST_GOAT_EV=1 — enable conservative EV gate (p_hat baseline)
- BACKTEST_NEWS_CSV=path.csv — events CSV with columns time, impact
- BACKTEST_NEWS_BLACKOUT_MIN / ANALYSIS_NEWS_BLACKOUT_MIN — blackout window minutes
- BACKTEST_USE_4H, BACKTEST_REQUIRE_4H_ALIGN — 4H confluence emulation
- BACKTEST_USE_MICRO, BACKTEST_REQUIRE_MICRO_CONFIRM — micro confirmation proxy
- FORWARD_SUMMARY_CSV=path.csv — append one-line summary per run
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import csv
import math
import time
import datetime
import numpy as np
import pandas as pd

# project import
try:
    from data import fetch_candles
except Exception:
    fetch_candles = None

# indicator helpers (optional use)
try:
    from utils import indicator_signals, ensemble_signal
except Exception:
    def indicator_signals(df: pd.DataFrame) -> Dict[str, str]:
        return {}
    def ensemble_signal(signals: Dict[str, str], trend: Optional[str] = None) -> str:
        return "Neutral"

# adaptive params (optional)
try:
    from adaptive_learning import adaptive_engine
except Exception:
    adaptive_engine = None

# optional tz for SAST
try:
    import pytz
    SAST_TZ = pytz.timezone("Africa/Johannesburg")
except Exception:
    SAST_TZ = None

# --------------- Shared instrument meta (unified with analysis.py) ---------------
def _instrument_point_and_pipvalue(pair: str) -> Dict[str, Optional[float]]:
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
    if any(k in p for k in ["US30", "DJI", "DJ30"]):
        return {"point": 1.0, "pip_value": 1.0}
    if any(k in p for k in ["US500", "SPX", "SP500"]):
        return {"point": 0.1, "pip_value": 1.0}
    if any(k in p for k in ["US100", "NAS100", "NDX"]):
        return {"point": 1.0, "pip_value": 1.0}
    if any(k in p for k in ["DE40", "GER40", "DAX"]):
        return {"point": 1.0, "pip_value": 1.0}
    if "XAU" in p:
        return {"point": 0.1, "pip_value": 1.0}
    if "XAG" in p:
        return {"point": 0.01, "pip_value": 0.5}
    if any(p.startswith(t) for t in ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "NFLX", "AMD", "INTC"]) or p.endswith("M"):
        return {"point": 0.01, "pip_value": 1.0}
    return {"point": None, "pip_value": None}

def _pips_for_pair(pair: str) -> float:
    try:
        p = (pair or "").upper()
        if p.endswith("JPY") or p.endswith("JPYC") or p.endswith("JPYc"):
            return 0.01
    except Exception:
        pass
    return 0.0001

def _resolve_instrument_meta(pair: str) -> Tuple[float, float]:
    meta = _instrument_point_and_pipvalue(pair)
    point = meta.get("point") if meta else None
    if not point:
        point = _pips_for_pair(pair)
    pv_env = os.getenv("BACKTEST_PIP_VALUE", os.getenv("ANALYSIS_PIP_VALUE") or "")
    if pv_env != "":
        pip_value_per_lot = float(pv_env)
    else:
        pip_value_per_lot = float(meta.get("pip_value")) if (meta and meta.get("pip_value")) else 10.0
    return float(point), float(pip_value_per_lot)

# --------------- Config and GOAT gates ---------------
FORWARD_SIMULATIONS = int(os.getenv("FORWARD_SIMULATIONS", "0"))  # 0 => deterministic
FORWARD_SIM_RANDOM_SEED = os.getenv("FORWARD_SIM_RANDOM_SEED", None)
DEFAULT_HORIZON = int(os.getenv("FORWARD_PRED_HORIZON", "0"))  # if >0 overrides function arg

SPREAD_PIPS = float(os.getenv("BACKTEST_SPREAD_PIPS", os.getenv("SPREAD_PIPS", "0.0")))
SLIPPAGE_PIPS = float(os.getenv("BACKTEST_SLIPPAGE_PIPS", os.getenv("SLIPPAGE_PIPS", "0.0")))
COMMISSION_PER_LOT = float(os.getenv("BACKTEST_COMMISSION_PER_LOT", os.getenv("COMMISSION_PER_LOT", "0.0")))
ROUND_TURN_COSTS = os.getenv("BACKTEST_ROUND_TURN", "0") == "1"

# GOAT filters
_ALLOWED_SESSIONS = tuple((os.getenv("BACKTEST_ALLOWED_SESSIONS", os.getenv("ANALYSIS_ALLOWED_SESSIONS") or "London,NY")).split(","))
_MAX_SPREAD_PIPS = float(os.getenv("BACKTEST_MAX_SPREAD_PIPS", os.getenv("ANALYSIS_MAX_SPREAD_PIPS") or "1.0"))
_MIN_ATR_PIPS = float(os.getenv("BACKTEST_MIN_ATR_PIPS", os.getenv("ANALYSIS_MIN_ATR_PIPS") or "0.0"))
_SESSION_STRICT = os.getenv("BACKTEST_SESSION_STRICT", os.getenv("ANALYSIS_SESSION_STRICT") or "0") == "1"
_ENABLE_EV_GATE = os.getenv("BACKTEST_GOAT_EV", "0") == "1"

# Confluence emulation
USE_4H = os.getenv("BACKTEST_USE_4H", os.getenv("ANALYSIS_USE_4H", "1")) != "0"
REQUIRE_4H_ALIGN = os.getenv("BACKTEST_REQUIRE_4H_ALIGN", "0") == "1"
USE_MICRO = os.getenv("BACKTEST_USE_MICRO", "1") != "0"
REQUIRE_MICRO_CONFIRM = os.getenv("BACKTEST_REQUIRE_MICRO_CONFIRM", "0") == "1"

# News blackout
NEWS_CSV_PATH = os.getenv("BACKTEST_NEWS_CSV", "")
NEWS_BLACKOUT_MIN = int(os.getenv("BACKTEST_NEWS_BLACKOUT_MIN", os.getenv("ANALYSIS_NEWS_BLACKOUT_MIN", "30")))
NEWS_IMPACT_LEVEL = os.getenv("BACKTEST_NEWS_IMPACT_LEVEL", "High").lower()

def _exchange_timezone(pair: str) -> Optional[str]:
    p = (pair or "").upper()
    if any(p.startswith(t) for t in ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "NFLX", "AMD", "INTC"]) or p.endswith("M"):
        return "America/New_York"
    return None

def _session_from_ts_tz(ts: Any, pair: str = None) -> str:
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

def _to_sast(ts: Any) -> str:
    try:
        t = pd.to_datetime(ts)
        if SAST_TZ is not None:
            if getattr(t, "tzinfo", None) is None:
                t = t.tz_localize("UTC").astimezone(SAST_TZ)
            else:
                t = t.tz_convert(SAST_TZ)
            return t.strftime("%Y-%m-%d %H:%M:%S %Z")
        else:
            # add +2h if tz lib missing
            return (t + pd.Timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S") + " SAST"
    except Exception:
        try:
            return str(ts)
        except Exception:
            return ""

def _compute_atr_simple(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    try:
        import ta
        return float(ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=period).average_true_range().iloc[-1])
    except Exception:
        try:
            return float(df["high"].subtract(df["low"]).tail(period).mean())
        except Exception:
            try:
                return float(np.std(df["close"].diff().dropna().tail(period)))
            except Exception:
                return None

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

def _apply_spread_slip(price: float, direction: str, point: float, for_exit: bool = False) -> float:
    cost = (SPREAD_PIPS + SLIPPAGE_PIPS) * (point if point else 1e-9)
    if not ROUND_TURN_COSTS and for_exit:
        # no exit costs
        return price
    if direction == "Buy":
        # entry worse: +cost; exit worse: -cost
        return float(price) - cost if for_exit else float(price) + cost
    else:
        # Sell: entry worse: -cost; exit worse: +cost
        return float(price) + cost if for_exit else float(price) - cost

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

def _news_blackout_at(ts: Any, entries: List[Dict[str, Any]]) -> bool:
    try:
        t = pd.to_datetime(ts)
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
    except Exception:
        return False
    return False

# Confluence utils
def _trend_dir_4h(df_4h: Optional[pd.DataFrame], idx_time: Any) -> Optional[str]:
    try:
        if df_4h is None or df_4h.empty:
            return None
        if "time" in df_4h.columns:
            dfh = df_4h[df_4h["time"] <= pd.to_datetime(idx_time)]
        else:
            dfh = df_4h[df_4h.index <= pd.to_datetime(idx_time)]
        if len(dfh) < 50:
            return None
        close = dfh["close"].astype(float).tail(100)
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
    try:
        if direction == "Buy":
            return _is_bullish_engulfing(base_df) or _is_hammer_like(base_df)
        if direction == "Sell":
            return _is_bearish_engulfing(base_df) or _is_shooting_star_like(base_df)
        return True
    except Exception:
        return True

# --------------- Core projection helpers ---------------
def _safe_mean(arr: List[float]) -> float:
    try:
        return float(np.mean(arr)) if arr else 0.0
    except Exception:
        return 0.0

def _deterministic_projection(df: pd.DataFrame, horizon: int) -> float:
    if horizon <= 0:
        return float(df["close"].iloc[-1])
    n = min(len(df), 60)
    if n < 3:
        return float(df["close"].iloc[-1])
    recent = df["close"].tail(n).astype(float).values
    x = np.arange(len(recent))
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        m, c = np.linalg.lstsq(A, recent, rcond=None)[0]
        projected = recent[-1] + m * horizon
        return float(projected)
    except Exception:
        mu = float(np.mean(np.diff(recent))) if len(recent) > 1 else 0.0
        return float(recent[-1] + mu * horizon)

def _simulate_one_path(df: pd.DataFrame, horizon: int, rng: np.random.RandomState) -> List[float]:
    prices = df["close"].astype(float).values
    last = float(prices[-1])
    recent = np.diff(prices[-min(len(prices), 200):])
    mu = float(np.mean(recent)) if len(recent) > 0 else 0.0
    sigma = float(np.std(recent)) if len(recent) > 0 else 0.0
    path = [last]
    for _ in range(horizon):
        step = rng.normal(mu, sigma)
        path.append(path[-1] + step)
    return path

def _compute_confidence_from_projection(df: pd.DataFrame, projected_price: float, suggestion: str) -> float:
    try:
        atr = _compute_atr_simple(df, period=14)
        if not atr or atr <= 0:
            vol_norm = float(np.std(df["close"].diff().dropna().tail(20)) or 1.0)
        else:
            vol_norm = float(atr)
        last = float(df["close"].iloc[-1])
        move = (projected_price - last) if suggestion == "Buy" else (last - projected_price)
        score = move / (vol_norm + 1e-9)
        conf = 1.0 / (1.0 + math.exp(-0.5 * score))
        if abs(move) < 0.1 * vol_norm:
            conf *= 0.5
        return float(max(0.0, min(0.99, conf)))
    except Exception:
        return 0.5

# -------------------------
# Public API
# -------------------------
def run_forward_test(pair: str, timeframe: str, prediction_horizon: int = 50) -> Dict[str, Any]:
    """
    Run forward test and return a structured result dict (never raises).
    """
    try:
        horizon = DEFAULT_HORIZON if DEFAULT_HORIZON and DEFAULT_HORIZON > 0 else int(prediction_horizon or 50)
        if fetch_candles is None:
            return {"success": False, "error": "fetch_candles not available (data.py missing)"}

        # fetch sufficient history
        lookback = max(500, horizon + 200)
        try:
            df = fetch_candles(pair, timeframe, n=lookback)
        except TypeError:
            df = fetch_candles(pair, timeframe, lookback)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {"success": False, "error": "No data for forward test"}
        for col in ("close",):
            if col not in df.columns:
                return {"success": False, "error": f"Missing required column: {col}"}

        # current time and price
        now_idx = -1
        price_now = float(df["close"].iloc[now_idx])
        ts_now = df["time"].iloc[now_idx] if "time" in df.columns else (df.index[now_idx] if hasattr(df, "index") else datetime.datetime.utcnow())

        # compute current suggestion
        try:
            signals = indicator_signals(df)
            suggestion = ensemble_signal(signals, trend=None) or "Neutral"
        except Exception:
            suggestion = "Neutral"

        # Instrument meta and costs
        point, pip_value_per_lot = _resolve_instrument_meta(pair)

        # Deterministic projection
        price_proj = _deterministic_projection(df, horizon)

        # Optional Monte-Carlo sims
        sims = max(0, int(FORWARD_SIMULATIONS))
        avg_sim_price = None
        med_sim_price = None
        pct_positive = None
        if sims > 0:
            try:
                seed = int(FORWARD_SIM_RANDOM_SEED) if FORWARD_SIM_RANDOM_SEED is not None else None
            except Exception:
                seed = None
            rng = np.random.RandomState(seed if seed is not None else int(time.time() % (2**32 - 1)))
            sim_final_prices = []
            for _ in range(sims):
                path = _simulate_one_path(df, horizon, rng)
                sim_final_prices.append(path[-1])
            if sim_final_prices:
                avg_sim_price = float(np.mean(sim_final_prices))
                med_sim_price = float(np.median(sim_final_prices))
                pct_positive = float(sum(1 for p in sim_final_prices if (p - price_now) > 0) / max(1, sims))
                price_proj = med_sim_price

        # GOAT-style gates at now
        reasons: List[str] = []
        session_now = _session_from_ts_tz(ts_now, pair=pair)
        session_ok = (session_now in _ALLOWED_SESSIONS) if _ALLOWED_SESSIONS else True
        spread_ok = (SPREAD_PIPS <= _MAX_SPREAD_PIPS)
        atr_val = _compute_atr_simple(df, period=14)
        atr_pips = (float(atr_val) / point) if (atr_val and point) else None
        atr_ok = True if (atr_pips is None) else (atr_pips >= _MIN_ATR_PIPS)
        ev_val = None
        if _ENABLE_EV_GATE:
            # coarse EV estimate (no full combined_score available here)
            # assume SL = 1*ATR and TP = 2*ATR around now for RR proxy
            if atr_val and atr_val > 0:
                sl_tmp = price_now - atr_val if suggestion == "Buy" else price_now + atr_val
                tp_tmp = price_now + 2 * atr_val if suggestion == "Buy" else price_now - 2 * atr_val
            else:
                sl_tmp, tp_tmp = None, None
            p_hat = _goat_p_hat(0.0)
            rr = _goat_rr(price_now, sl_tmp, tp_tmp, atr_val)
            ev_val = p_hat * rr - (1 - p_hat) * 1.0
        if not spread_ok:
            reasons.append("spread_filter")
        if _SESSION_STRICT and not session_ok:
            reasons.append("session_filter")
        if not atr_ok:
            reasons.append("atr_too_low")
        if _ENABLE_EV_GATE and (ev_val is not None) and (ev_val <= 0.0):
            reasons.append("negative_expected_value")
        goat_gate = {
            "session": session_now,
            "session_ok": session_ok,
            "spread_pips": SPREAD_PIPS,
            "spread_ok": spread_ok,
            "atr": float(atr_val) if atr_val is not None else None,
            "atr_pips": round(atr_pips, 3) if atr_pips is not None else None,
            "atr_ok": atr_ok,
            "ev": round(ev_val, 4) if ev_val is not None else None,
            "reasons": reasons,
            "ok": (len(reasons) == 0)
        }

        # News blackout
        news_events = _load_news_csv(NEWS_CSV_PATH) if NEWS_CSV_PATH else []
        news_blocked = (_news_blackout_at(ts_now, news_events) if (news_events and _SESSION_STRICT) else False)
        if news_blocked:
            goat_gate["reasons"].append("news_blackout")
            goat_gate["ok"] = False

        # 4H alignment
        dir4h = None
        if USE_4H:
            try:
                df_4h = fetch_candles(pair, "4h", n=600)
            except TypeError:
                df_4h = fetch_candles(pair, "4h", 600)
            except Exception:
                df_4h = None
            dir4h = _trend_dir_4h(df_4h, ts_now) if isinstance(df_4h, pd.DataFrame) else None

        # Micro confirmation (using last 2 bars)
        micro_confirm = True
        if REQUIRE_MICRO_CONFIRM or USE_MICRO:
            base_slice = df.tail(3)
            micro_confirm = _micro_confirm_ok(base_slice, suggestion)

        # Forward PnL including costs (1-lot evaluation)
        # Apply costs at entry (now) and optionally at exit (projected)
        entry_effective = _apply_spread_slip(price_now, suggestion, point, for_exit=False)
        exit_effective = _apply_spread_slip(price_proj, suggestion, point, for_exit=True)
        if suggestion == "Buy":
            pnl_price = float(exit_effective) - float(entry_effective)
        elif suggestion == "Sell":
            pnl_price = float(entry_effective) - float(exit_effective)
        else:
            pnl_price = 0.0

        # commission on entry (and exit if round-turn)
        lot_one = 1.0
        commission_total = COMMISSION_PER_LOT * lot_one
        if ROUND_TURN_COSTS:
            commission_total += COMMISSION_PER_LOT * lot_one

        pnl_pips = pnl_price / (point if point else 1e-9)
        pnl_value = pnl_pips * pip_value_per_lot * lot_one
        pnl_value_net = pnl_value - commission_total

        # Confidence with small confluence adjustments
        confidence = _compute_confidence_from_projection(df, price_proj, suggestion)
        try:
            if dir4h in ("Buy", "Sell") and suggestion in ("Buy", "Sell") and dir4h == suggestion:
                confidence = min(0.99, confidence + 0.05)
            if micro_confirm and suggestion in ("Buy", "Sell"):
                confidence = min(0.99, confidence + 0.03)
            if not goat_gate["ok"]:
                confidence = max(0.2, confidence - 0.1)
        except Exception:
            pass

        success_flag = bool(pnl_value_net > 0)

        # Result
        out: Dict[str, Any] = {
            "success": True,
            "pair": pair,
            "timeframe": timeframe,
            "suggestion": suggestion,
            "price_now": round(price_now, 6),
            "price_future": round(price_proj, 6),       # alias maintained
            "price_projected": round(price_proj, 6),    # explicit
            "pnl_price": round(pnl_price, 6),
            "pnl_pips": round(pnl_pips, 2),
            "pnl": round(pnl_value_net, 6),             # net of commission and costs
            "pnl_per_lot": round(pnl_value_net, 6),
            "confidence": round(float(confidence), 4),
            "prediction_horizon": horizon,
            "simulations": sims,
            "sim_median_price": round(med_sim_price, 6) if med_sim_price is not None else None,
            "sim_avg_price": round(avg_sim_price, 6) if avg_sim_price is not None else None,
            "sim_win_prob": round(float(pct_positive), 4) if pct_positive is not None else None,
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_sast": _to_sast(datetime.datetime.utcnow()),
            "instrument_meta": {"point": point, "pip_value_per_lot": pip_value_per_lot},
            "costs": {
                "spread_pips": SPREAD_PIPS,
                "slippage_pips": SLIPPAGE_PIPS,
                "commission_per_lot": COMMISSION_PER_LOT,
                "round_turn_costs": bool(ROUND_TURN_COSTS)
            },
            "goat_gate": goat_gate,
            "confluence": {
                "dir_4h": dir4h,
                "micro_confirm": micro_confirm
            },
            "news_blocked": bool(news_blocked),
            "success_label": success_flag
        }

        # CSV export (single-row summary)
        try:
            fcsv = os.getenv("FORWARD_SUMMARY_CSV", "")
            if fcsv:
                write_header = not os.path.exists(fcsv)
                with open(fcsv, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow([
                            "timestamp_sast","pair","timeframe","suggestion","price_now","price_projected",
                            "pnl","pnl_pips","confidence","horizon","session","goat_ok","dir_4h","micro_ok","news_blocked"
                        ])
                    w.writerow([
                        out["timestamp_sast"],
                        pair,
                        timeframe,
                        suggestion,
                        out["price_now"],
                        out["price_projected"],
                        out["pnl"],
                        out["pnl_pips"],
                        out["confidence"],
                        horizon,
                        goat_gate.get("session"),
                        goat_gate.get("ok"),
                        dir4h,
                        micro_confirm,
                        news_blocked
                    ])
        except Exception:
            pass

        # adaptive params for debugging (not required)
        try:
            if adaptive_engine is not None:
                out["adaptive_params"] = adaptive_engine.get_params()
        except Exception:
            out["adaptive_params"] = None

        return out

    except Exception as e:
        import traceback
        return {"success": False, "error": f"forward_test error: {e}", "trace": traceback.format_exc()}
#forward_test_engine.py
