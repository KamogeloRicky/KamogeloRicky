"""
Precision entry detection for RickyFX — enhanced with ETA / expected-bars prediction + edge-entry refinement + VWAP filter.

Key features:
- Micro confirmations (BOS/CHoCH heuristic, patterns, retest).
- ETA predictions (expected_micro_bars, expected_main_bars, eta_time_sast).
- Edge-entry refinement:
  - Retest zone from pivot ± tighten*ATR.
  - Edge-of-zone entry with FVG/OB confluence and VWAP alignment filter.
  - Protected SL beyond zone and recent swing with ATR/tick buffers.
- Unified instrument meta for tick-aware offsets.
- Defensive fallbacks and backward compatibility.

VWAP filter:
- Enabled by default; requires micro confirm bar to align with VWAP if volume is present:
  - Buy: close >= VWAP
  - Sell: close <= VWAP
- Adds a small confidence boost when aligned.
"""

import os
import math
from typing import Optional, Dict, Any, List, Tuple
import statistics
import pandas as pd
import numpy as np

# ta is optional
try:
    import ta  # type: ignore
except Exception:
    ta = None  # graceful fallback

# timezone handling
try:
    import pytz  # type: ignore
    SAST_TZ = pytz.timezone("Africa/Johannesburg")
except Exception:
    SAST_TZ = None  # fallback to manual +02 if pytz not available

# Try to import fetch_candles if available (used for micro confirmations/ETA)
try:
    from data import fetch_candles
except Exception:
    fetch_candles = None

# Try to import helpers from analysis.py if the project exposes them
try:
    from analysis import detect_bos_choch_liquidity as _ext_detect_bos_choch  # type: ignore
except Exception:
    _ext_detect_bos_choch = None

try:
    from analysis import detect_retest as _ext_detect_retest  # type: ignore
except Exception:
    _ext_detect_retest = None

# Adaptive / tuning defaults (compatible with adaptive_learning)
DEFAULTS = {
    "atr_period": 14,
    "body_ratio_threshold": 0.55,      # body >= 55% of candle range
    "liquidity_sweep_bonus": 0.12,     # extra confidence if liquidity sweep detected
    "buffer_atr_mult": 0.6,            # buffer around wick for SL (fallback when no edge refinement)
    "tp_atr_mult": 2.0,                # TP = entry +/- tp_atr_mult * ATR
    "min_confidence": 0.45,
    "min_volume_ratio": 0.6,           # last vol must be >= 0.6 * mean recent vol (otherwise low-vol)
    "micro_bars": 200,
    "micro_timeframe": "5m",
    "retest_window": 30,               # bars to look for pivot
    "pivot_lookback": 200,             # bars used to compute historical retest durations
    "min_retest_samples": 2,           # min samples to compute median duration
    "default_expected_micro_bars": 2,  # fallback expected bars if no history
    # edge-entry refinement
    "refine_edge_enable": True,        # compute edge-of-zone entry and protected SL
    "apply_edge_as_entry": True,       # when refined available, set entry_point to the edge price
    "retest_tighten": 0.4,             # zone width factor (pivot ± tighten*ATR)
    "sl_buffer_atr": 0.35,             # protected SL additional ATR buffer
    "sl_min_ticks": 3,                 # protected SL minimum ticks beyond zone
    "entry_tick_offset": 1,            # edge offset in ticks for limit placement
    "edge_search_bars": 180,           # how many recent micro bars to scan for touch+confirm
    "ob_search": 10,                   # lookback bars to find last opposite OB
    "micro_confirm_required": False,   # if True, require micro confirmation to confirm entry
    # VWAP filter
    "vwap_filter_enable": True,        # require VWAP alignment on micro confirm bar if volume exists
    "vwap_align_boost": 0.03           # confidence boost when VWAP alignment passes
}

# -------------------------
# Instrument meta (point/tick size and pip value per lot)
# -------------------------
def _instrument_point_and_pipvalue(pair: Optional[str]) -> Dict[str, Optional[float]]:
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
    # FX default
    if p.endswith("JPY") or p.endswith("JPYC") or p.endswith("JPYc"):
        return {"point": 0.01, "pip_value": 10.0}
    return {"point": 0.0001, "pip_value": 10.0}

def _point_from_context(context: Optional[Dict[str, Any]]) -> float:
    try:
        pair = (context or {}).get("pair") or (context or {}).get("symbol")
        meta = _instrument_point_and_pipvalue(pair)
        pt = meta.get("point") or 0.0001
        return float(pt)
    except Exception:
        return 0.0001

# -------------------------
# Utilities: timezones, ATR, helpers
# -------------------------
def _to_sast(ts):
    try:
        if isinstance(ts, (pd.Timestamp,)):
            t = ts
        else:
            t = pd.to_datetime(ts)
        if t.tzinfo is None or str(t.tzinfo) == 'None':
            try:
                t = t.tz_localize("UTC")
            except Exception:
                pass
        if SAST_TZ is not None:
            try:
                t = t.tz_convert(SAST_TZ)
            except Exception:
                try:
                    t = t + pd.Timedelta(hours=2)
                except Exception:
                    pass
        else:
            try:
                t = t + pd.Timedelta(hours=2)
            except Exception:
                pass
        try:
            return t.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            return t.strftime("%Y-%m-%d %H:%M:%S") + " SAST"
    except Exception:
        try:
            t2 = pd.to_datetime(ts)
            t2 = t2 + pd.Timedelta(hours=2)
            return t2.strftime("%Y-%m-%d %H:%M:%S") + " SAST"
        except Exception:
            return str(ts)


def _compute_atr(df: pd.DataFrame, period: int) -> Optional[float]:
    try:
        if ta is not None:
            atr_ser = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=period).average_true_range()
            return float(atr_ser.iloc[-1])
    except Exception:
        pass
    try:
        return float(np.std(df["close"].diff().dropna().tail(period)))
    except Exception:
        try:
            return float((df["high"] - df["low"]).tail(period).mean())
        except Exception:
            return None


def _is_strong_body(candle: pd.Series, threshold: float) -> bool:
    try:
        rng = float(candle["high"] - candle["low"])
        if rng == 0 or math.isnan(rng):
            return False
        body = float(abs(candle["close"] - candle["open"]))
        return body >= threshold * rng
    except Exception:
        return False


def _liquidity_sweep_check(df: pd.DataFrame, direction: str) -> bool:
    try:
        if len(df) < 3:
            return False
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        if direction == "Buy":
            return float(last["low"]) < min(float(prev["low"]), float(prev2["low"]))
        else:
            return float(last["high"]) > max(float(prev["high"]), float(prev2["high"]))
    except Exception:
        return False


def detect_volume_surge(df: pd.DataFrame, lookback: int = 20, multiplier: float = 1.5) -> bool:
    try:
        if "volume" not in df.columns or len(df) < lookback + 1:
            return False
        recent = df["volume"].iloc[-lookback:]
        last = float(df["volume"].iloc[-1])
        return last > (recent.mean() * multiplier)
    except Exception:
        return False


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
        body = abs(float(c["close"]) - float(c["open"]))
        lower_wick = (float(c["open"]) - float(c["low"])) if float(c["open"]) >= float(c["close"]) else (float(c["close"]) - float(c["low"]))
        upper_wick = float(c["high"]) - max(float(c["close"]), float(c["open"]))
        if body == 0:
            return False
        return lower_wick > (body * 2) and upper_wick < (body * 0.5)
    except Exception:
        return False


def is_shooting_star_like(df: pd.DataFrame) -> bool:
    try:
        if len(df) < 1:
            return False
        c = df.iloc[-1]
        body = abs(float(c["close"]) - float(c["open"]))
        upper_wick = float(c["high"]) - max(float(c["close"]), float(c["open"]))
        lower_wick = min(float(c["close"]), float(c["open"])) - float(c["low"])
        if body == 0:
            return False
        return upper_wick > (body * 2) and lower_wick < (body * 0.5)
    except Exception:
        return False

# -------------------------
# Structure and retest helpers (local fallbacks)
# -------------------------
def detect_bos_choch_liquidity(df: pd.DataFrame) -> Dict[str, Any]:
    if callable(_ext_detect_bos_choch):
        try:
            return _ext_detect_bos_choch(df)  # type: ignore
        except Exception:
            pass
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
        liquidity = _liquidity_sweep_check(df, "Buy" if (bos == "Sell") else "Sell") if bos else False
        return {"bos": bos, "choch": choch, "liquidity_sweep": liquidity, "details": {"recent_high": recent_high, "recent_low": recent_low}}
    except Exception:
        return {"bos": None, "choch": False, "liquidity_sweep": False, "details": "error"}


def detect_retest(df: pd.DataFrame, direction: str, atr: Optional[float], sensitivity: float = 1.0) -> Optional[Dict[str, Any]]:
    if callable(_ext_detect_retest):
        try:
            return _ext_detect_retest(df, direction, atr, sensitivity)  # type: ignore
        except Exception:
            pass
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
        if atr and abs(last_price - pivot_price) <= max(0.00001, float(sensitivity) * atr):
            if direction == "Buy":
                entry_price = last_price
                sl = pivot_price - 0.5 * float(atr)
            else:
                entry_price = last_price
                sl = pivot_price + 0.5 * float(atr)
            confidence = 0.45 + min(0.45, 0.25 * (1.0 / (1.0 + abs(last_price - pivot_price))))
            return {"entry_price": float(entry_price), "sl": float(sl), "confidence": float(confidence), "reason": "heuristic_retest", "pivot_price": float(pivot_price)}
    except Exception:
        return None
    return None

# -------------------------
# Micro ETA helpers
# -------------------------
def _find_local_pivots(df: pd.DataFrame, lookback: int = 30) -> List[int]:
    idxs = []
    n = len(df)
    w = min(lookback, max(3, n // 10))
    if n < 5:
        return idxs
    highs = df["high"].values
    lows = df["low"].values
    for i in range(w, n - w):
        seg_h = highs[i - w:i + w + 1]
        seg_l = lows[i - w:i + w + 1]
        if highs[i] == seg_h.max():
            idxs.append(i)
        elif lows[i] == seg_l.min():
            idxs.append(i)
    return sorted(idxs)


def _compute_retest_durations_micro(micro_df: pd.DataFrame, direction: str, pivot_lookback: int = 200, search_window: int = 60) -> List[int]:
    try:
        n = len(micro_df)
        if n < 20:
            return []
        lookback = min(pivot_lookback, n // 2)
        sub = micro_df.iloc[-lookback:].reset_index(drop=True)
        pivots = _find_local_pivots(sub, lookback=min(30, max(5, lookback // 10)))
        durations = []
        closes = sub["close"].values
        for p in pivots:
            pivot_price = float(sub["high"].iloc[p]) if direction == "Sell" else float(sub["low"].iloc[p])
            for j in range(p + 1, min(len(sub), p + 1 + search_window)):
                c = closes[j]
                if direction == "Buy" and c >= pivot_price:
                    durations.append(j - p)
                    break
                if direction == "Sell" and c <= pivot_price:
                    durations.append(j - p)
                    break
        return durations
    except Exception:
        return []


def _median_positive_int(values: List[int], fallback: int) -> int:
    try:
        vals = [int(v) for v in values if isinstance(v, (int, float)) and v > 0]
        if not vals:
            return fallback
        return int(max(1, int(statistics.median(vals))))
    except Exception:
        return fallback


def _parse_timeframe_to_minutes(tf: Optional[str]) -> Optional[int]:
    if not tf:
        return None
    try:
        s = str(tf).strip().lower()
        if s.endswith("m"):
            return int(s[:-1])
        if s.endswith("h"):
            return int(s[:-1]) * 60
        if s.endswith("d"):
            return int(s[:-1]) * 1440
        return int(s)
    except Exception:
        return None

# -------------------------
# Edge-entry refinement helpers (FVG, OB, VWAP, protected SL)
# -------------------------
def _tick_from_point(point: Optional[float]) -> float:
    try:
        p = float(point)
        if p > 0:
            return p
    except Exception:
        pass
    return 0.0001

def _protected_sl_price(direction: str,
                        z_lo: float,
                        z_hi: float,
                        swing_low: Optional[float],
                        swing_high: Optional[float],
                        atr: Optional[float],
                        point: float,
                        sl_buffer_atr: float,
                        sl_min_ticks: int) -> float:
    try:
        tick = _tick_from_point(point)
        a_buf = float(atr) * float(sl_buffer_atr) if (atr is not None) else 0.0
        tick_buf = int(sl_min_ticks) * tick
        buf = max(a_buf, tick_buf)
        if direction == "Buy":
            base = min([v for v in [swing_low, z_lo] if v is not None])
            return float(base - buf)
        if direction == "Sell":
            base = max([v for v in [swing_high, z_hi] if v is not None])
            return float(base + buf)
    except Exception:
        pass
    return float(z_lo - sl_min_ticks * _tick_from_point(point)) if direction == "Buy" else float(z_hi + sl_min_ticks * _tick_from_point(point))

def _micro_swing_extremes(df: pd.DataFrame, i: int, lb: int = 3) -> Tuple[Optional[float], Optional[float]]:
    try:
        lo = float(df["low"].iloc[max(0, i - lb): i + 1].min())
        hi = float(df["high"].iloc[max(0, i - lb): i + 1].max())
        return lo, hi
    except Exception:
        return None, None

def _detect_fvg_zone(micro_df: pd.DataFrame, i: int, direction: str, lookback: int = 6) -> Optional[Tuple[float, float]]:
    try:
        start = max(1, i - lookback)
        end = i
        for j in range(end, start - 1, -1):
            if j - 1 < 0 or j >= len(micro_df):
                continue
            a = micro_df.iloc[j - 1]
            b = micro_df.iloc[j]
            if direction == "Buy" and float(a["low"]) > float(b["high"]):
                return (float(b["high"]), float(a["low"]))
            if direction == "Sell" and float(a["high"]) < float(b["low"]):
                return (float(a["high"]), float(b["low"]))
    except Exception:
        pass
    return None

def _last_opposite_order_block_level(micro_df: pd.DataFrame, i: int, direction: str, search: int = 10) -> Optional[float]:
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

def _micro_confirm_bar(micro_df: pd.DataFrame, i: int, direction: str) -> bool:
    try:
        row = micro_df.iloc[i]
        if direction == "Buy":
            return float(row["close"]) > float(row["open"])
        if direction == "Sell":
            return float(row["close"]) < float(row["open"])
    except Exception:
        return False
    return False

def _qualify_microstructure(micro_df: pd.DataFrame, i: int, direction: str) -> Dict[str, Any]:
    try:
        labels = []
        score = 0.0
        row = micro_df.iloc[i]
        rng = float(row["high"]) - float(row["low"]) + 1e-12
        upper_wick = float(row["high"]) - max(float(row["close"]), float(row["open"]))
        lower_wick = min(float(row["close"]), float(row["open"])) - float(row["low"])
        wick_ratio = (lower_wick if direction == "Buy" else upper_wick) / rng
        if wick_ratio > 0.4:
            labels.append("wick_reject")
            score += 0.35
        try:
            prev = micro_df.iloc[i-1]
            if direction == "Buy":
                if float(prev["low"]) > float(row["high"]):
                    labels.append("micro_fvg")
                    score += 0.3
            else:
                if float(prev["high"]) < float(row["low"]):
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
                    if float(row["low"]) <= ob_hi <= float(row["high"]):
                        labels.append("ob_tap")
                        score += 0.25
            else:
                opp = lookback[(lookback["close"] > lookback["open"])]
                if not opp.empty:
                    ob_lo = float(opp["low"].iloc[-1])
                    if float(row["low"]) <= ob_lo <= float(row["high"]):
                        labels.append("ob_tap")
                        score += 0.25
        except Exception:
            pass
        score = float(max(0.0, min(1.0, score)))
        return {"precision_score": score, "labels": labels}
    except Exception:
        return {"precision_score": 0.0, "labels": []}

def _compute_vwap(df: pd.DataFrame) -> Optional[pd.Series]:
    try:
        if "volume" not in df.columns:
            return None
        tp = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
        vol = df["volume"].astype(float)
        cum_vol = vol.cumsum().replace(0, np.nan)
        vwap = (tp * vol).cumsum() / cum_vol
        return vwap
    except Exception:
        return None

def _pick_best_entry_edge(direction: str,
                          z_lo: float,
                          z_hi: float,
                          point: float,
                          fvg_zone: Optional[Tuple[float, float]],
                          ob_level: Optional[float],
                          entry_tick_offset: int) -> float:
    tick = _tick_from_point(point)
    off = entry_tick_offset * tick
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
    return float(max(candidates) if direction == "Buy" else min(candidates))

# -------------------------
# Core detection with ETA / edge refinement + VWAP
# -------------------------
def detect_precision_entry(df: pd.DataFrame,
                           context: Optional[Dict[str, Any]] = None,
                           expected_direction: Optional[str] = None,
                           params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    if params is None:
        params = DEFAULTS.copy()
    if df is None or len(df) < 3:
        return None

    try:
        context = context or {}
        retest_info = context.get("retest") or {}
        retest_level = retest_info.get("level")

        direction = expected_direction or context.get("signal") or context.get("direction") or None
        if direction is None:
            last = float(df["close"].iloc[-1])
            prev = float(df["close"].iloc[-2])
            direction = "Buy" if last > prev else "Sell"
        direction = "Buy" if str(direction).lower().startswith("b") else "Sell"

        last_candle = df.iloc[-1]
        candle_dir_ok = (float(last_candle["close"]) > float(last_candle["open"])) if direction == "Buy" else (float(last_candle["close"]) < float(last_candle["open"]))
        strong_body = _is_strong_body(last_candle, float(params.get("body_ratio_threshold", DEFAULTS["body_ratio_threshold"])))

        closes_past_level = True
        if retest_level is not None:
            try:
                if direction == "Buy":
                    closes_past_level = float(last_candle["close"]) >= float(retest_level) - 1e-12
                else:
                    closes_past_level = float(last_candle["close"]) <= float(retest_level) + 1e-12
            except Exception:
                closes_past_level = True

        atr = _compute_atr(df, int(params.get("atr_period", DEFAULTS["atr_period"]))) or 0.0
        vol_ok = True
        try:
            if "volume" in df.columns:
                recent_mean = float(df["volume"].tail(20).mean())
                last_vol = float(df["volume"].iloc[-1])
                vol_ok = last_vol >= (recent_mean * float(params.get("min_volume_ratio", DEFAULTS["min_volume_ratio"])))
        except Exception:
            vol_ok = True

        sweep = _liquidity_sweep_check(df, direction)
        struct = detect_bos_choch_liquidity(df)

        bull_eng = is_bullish_engulfing(df)
        bear_eng = is_bearish_engulfing(df)
        hammer = is_hammer_like(df)
        shooter = is_shooting_star_like(df)
        pattern_confirm = False
        if direction == "Buy" and (bull_eng or hammer):
            pattern_confirm = True
        if direction == "Sell" and (bear_eng or shooter):
            pattern_confirm = True

        micro_confirm = None
        expected_micro_bars = None
        expected_main_bars = None
        eta_time_sast = None

        entry_point_edge = None
        protected_sl = None
        precision_grade = None
        precision_labels: List[str] = []
        retest_zone = None
        fvg_zone = None
        ob_level = None
        entry_time_refined_sast = None
        vwap_at_refined = None
        vwap_align_ok_edge = None

        micro_tf = str(params.get("micro_timeframe", DEFAULTS["micro_timeframe"]))
        micro_bars = int(params.get("micro_bars", DEFAULTS["micro_bars"]))
        edge_search = int(params.get("edge_search_bars", DEFAULTS["edge_search_bars"]))
        ob_search = int(params.get("ob_search", DEFAULTS["ob_search"]))
        entry_tick_offset = int(params.get("entry_tick_offset", DEFAULTS["entry_tick_offset"]))
        sl_buffer_atr = float(params.get("sl_buffer_atr", DEFAULTS["sl_buffer_atr"]))
        sl_min_ticks = int(params.get("sl_min_ticks", DEFAULTS["sl_min_ticks"]))
        retest_tighten = float(params.get("retest_tighten", DEFAULTS["retest_tighten"]))
        refine_on = bool(params.get("refine_edge_enable", DEFAULTS["refine_edge_enable"]))
        apply_edge_as_entry = bool(params.get("apply_edge_as_entry", DEFAULTS["apply_edge_as_entry"]))
        vwap_filter_enable = bool(params.get("vwap_filter_enable", DEFAULTS["vwap_filter_enable"]))
        vwap_align_boost = float(params.get("vwap_align_boost", DEFAULTS["vwap_align_boost"]))

        point = _point_from_context(context)

        try:
            pair = context.get("pair") or context.get("symbol")
            micro_df = None
            if fetch_candles is not None and pair is not None:
                try:
                    micro_df = fetch_candles(pair, micro_tf, n=micro_bars)
                except TypeError:
                    micro_df = fetch_candles(pair, micro_tf, micro_bars)
            if isinstance(micro_df, pd.DataFrame) and not micro_df.empty:
                micro_atr = _compute_atr(micro_df, int(params.get("atr_period", DEFAULTS["atr_period"])))
                micro_sweep = _liquidity_sweep_check(micro_df, direction)
                micro_bull_eng = is_bullish_engulfing(micro_df)
                micro_bear_eng = is_bearish_engulfing(micro_df)
                micro_hammer = is_hammer_like(micro_df)
                micro_shooter = is_shooting_star_like(micro_df)
                micro_patterns = []
                if micro_bull_eng: micro_patterns.append("bullish_engulfing")
                if micro_bear_eng: micro_patterns.append("bearish_engulfing")
                if micro_hammer: micro_patterns.append("hammer_like")
                if micro_shooter: micro_patterns.append("shooting_star_like")
                micro_heur_retest = detect_retest(micro_df, direction, micro_atr, sensitivity=0.8)
                vwap_series = _compute_vwap(micro_df)
                micro_confirm = {
                    "micro_atr": micro_atr,
                    "micro_sweep": micro_sweep,
                    "micro_patterns": micro_patterns,
                    "micro_heuristic_retest": micro_heur_retest,
                    "micro_df_len": len(micro_df),
                    "micro_vwap": True if vwap_series is not None else False
                }

                durations = _compute_retest_durations_micro(micro_df, direction, pivot_lookback=int(params.get("pivot_lookback", DEFAULTS["pivot_lookback"])))
                median_dur = _median_positive_int(durations, int(params.get("default_expected_micro_bars", DEFAULTS["default_expected_micro_bars"])))
                if micro_heur_retest:
                    expected_micro_bars = 0
                else:
                    expected_micro_bars = median_dur if median_dur and median_dur > 0 else int(params.get("default_expected_micro_bars", DEFAULTS["default_expected_micro_bars"]))

                main_tf = context.get("timeframe") or context.get("base_timeframe")
                micro_minutes = _parse_timeframe_to_minutes(micro_tf)
                main_minutes = _parse_timeframe_to_minutes(main_tf) if main_tf else None
                if main_minutes and micro_minutes and expected_micro_bars is not None:
                    ratio = max(1, int(round(main_minutes / micro_minutes)))
                    expected_main_bars = int(math.ceil((expected_micro_bars or 0) / ratio))
                else:
                    expected_main_bars = None

                last_time = None
                if "time" in micro_df.columns:
                    try:
                        last_time = micro_df["time"].iloc[-1]
                    except Exception:
                        last_time = None
                if last_time is None:
                    try:
                        last_time = micro_df.index[-1]
                    except Exception:
                        last_time = None
                if last_time is not None and expected_micro_bars is not None and micro_minutes:
                    ts = pd.to_datetime(last_time)
                    eta = ts + pd.Timedelta(minutes=int(expected_micro_bars * micro_minutes))
                    eta_time_sast = _to_sast(eta)

                # Edge-entry refinement with VWAP filter
                if refine_on:
                    heur = micro_heur_retest or detect_retest(df, direction, atr, sensitivity=0.8)
                    pivot = None
                    if heur and (("pivot_price" in heur) or ("entry_price" in heur)):
                        pivot = float(heur.get("pivot_price", heur.get("entry_price")))
                    else:
                        look = min(60, len(micro_df))
                        if direction == "Buy":
                            pivot = float(micro_df["low"].tail(look).min())
                        else:
                            pivot = float(micro_df["high"].tail(look).max())
                    span = float((micro_atr if (micro_atr and not math.isnan(micro_atr)) else (atr or 0.0)))*retest_tighten
                    if span <= 0:
                        span = abs(pivot) * 0.0002
                    z_lo = float(pivot - span)
                    z_hi = float(pivot + span)
                    retest_zone = {"low": round(z_lo, 6), "high": round(z_hi, 6), "pivot": round(pivot, 6)}

                    N = min(edge_search, len(micro_df))
                    for i in range(len(micro_df)-N, len(micro_df)):
                        row = micro_df.iloc[i]
                        touched = (float(row["low"]) <= z_hi and float(row["high"]) >= z_lo)
                        if not touched:
                            continue
                        if not _micro_confirm_bar(micro_df, i, direction):
                            continue
                        # VWAP alignment filter (only if we have vwap and it is enabled)
                        if vwap_filter_enable and vwap_series is not None:
                            try:
                                v = float(vwap_series.iloc[i])
                                vwap_at_refined = v
                                if direction == "Buy" and float(row["close"]) < v:
                                    continue
                                if direction == "Sell" and float(row["close"]) > v:
                                    continue
                                vwap_align_ok_edge = True
                            except Exception:
                                vwap_align_ok_edge = None

                        qual = _qualify_microstructure(micro_df, i, direction)
                        precision_grade = qual["precision_score"]
                        precision_labels = qual["labels"].copy()
                        if vwap_align_ok_edge:
                            precision_labels.append("vwap_align")

                        fvg_zone = _detect_fvg_zone(micro_df, i, direction, lookback=6)
                        ob_level = _last_opposite_order_block_level(micro_df, i, direction, search=ob_search)
                        entry_point_edge = _pick_best_entry_edge(direction, z_lo, z_hi, point, fvg_zone, ob_level, entry_tick_offset)
                        sw_lo, sw_hi = _micro_swing_extremes(micro_df, i, lb=3)
                        protected_sl = _protected_sl_price(direction, z_lo, z_hi, sw_lo, sw_hi, (micro_atr or atr), point, sl_buffer_atr, sl_min_ticks)
                        entry_time_refined_sast = _to_sast(micro_df.index[i] if hasattr(micro_df, "index") else last_time)
                        break
        except Exception:
            micro_confirm = None

        # Confirmation logic (conservative)
        confirmed = False
        if candle_dir_ok and strong_body and closes_past_level and vol_ok:
            confirmed = True
        if (not confirmed) and micro_confirm:
            micro_ok = False
            if micro_confirm.get("micro_heuristic_retest"):
                micro_ok = True
            if micro_confirm.get("micro_patterns"):
                pats = micro_confirm.get("micro_patterns", [])
                if direction == "Buy" and any(p in pats for p in ("bullish_engulfing", "hammer_like")):
                    micro_ok = True
                if direction == "Sell" and any(p in pats for p in ("bearish_engulfing", "shooting_star_like")):
                    micro_ok = True
            if micro_ok and (candle_dir_ok or pattern_confirm):
                confirmed = True
        if (not confirmed) and sweep and pattern_confirm and vol_ok:
            confirmed = True
        if params.get("micro_confirm_required", DEFAULTS["micro_confirm_required"]) and not micro_confirm:
            confirmed = False

        # If not confirmed: return prediction/ETA-only payload
        if not confirmed:
            return {
                "confirmed": False,
                "expected_micro_bars": int(expected_micro_bars) if expected_micro_bars is not None else None,
                "expected_main_bars": int(expected_main_bars) if expected_main_bars is not None else None,
                "eta_time_sast": eta_time_sast,
                "debug": {
                    "strong_body": bool(strong_body),
                    "pattern_confirm": bool(pattern_confirm),
                    "sweep": bool(sweep),
                    "vol_ok": bool(vol_ok),
                    "structure": struct,
                    "micro_confirm": micro_confirm,
                    "retest_zone": retest_zone,
                    "fvg_zone": fvg_zone,
                    "ob_level": ob_level,
                    "vwap_filter": {
                        "enabled": bool(vwap_filter_enable),
                        "available": bool(micro_confirm.get("micro_vwap") if micro_confirm else False)
                    }
                }
            }

        # compute entry/SL/TP (fallback first)
        entry_point = float(last_candle["close"])
        buffer = float(params.get("buffer_atr_mult", DEFAULTS["buffer_atr_mult"])) * (atr or 0.0)
        if direction == "Buy":
            sl_fallback = float(last_candle["low"]) - (buffer if not math.isnan(buffer) else 0.0)
            tp_val = entry_point + float(params.get("tp_atr_mult", DEFAULTS["tp_atr_mult"])) * (atr or 0.0)
        else:
            sl_fallback = float(last_candle["high"]) + (buffer if not math.isnan(buffer) else 0.0)
            tp_val = entry_point - float(params.get("tp_atr_mult", DEFAULTS["tp_atr_mult"])) * (atr or 0.0)

        # Apply edge-entry refinement if available
        if refine_on and entry_point_edge is not None:
            if apply_edge_as_entry:
                entry_point = float(entry_point_edge)
            if protected_sl is not None:
                sl_val = float(protected_sl)
            else:
                sl_val = float(sl_fallback)
        else:
            sl_val = float(sl_fallback)

        # confidence composition
        base_conf = 0.55 if strong_body else 0.45
        if pattern_confirm:
            base_conf += 0.08
        if sweep:
            base_conf += float(params.get("liquidity_sweep_bonus", DEFAULTS["liquidity_sweep_bonus"]))
        if micro_confirm:
            if micro_confirm.get("micro_heuristic_retest"):
                base_conf += 0.12
            if micro_confirm.get("micro_patterns"):
                base_conf += 0.08
        if entry_point_edge is not None and protected_sl is not None:
            base_conf += 0.03
        if vwap_align_ok_edge:
            base_conf += float(vwap_align_boost)
        confidence = min(0.99, max(float(params.get("min_confidence", DEFAULTS["min_confidence"])), base_conf))

        entry_time_src = last_candle.get("time") if "time" in last_candle.index else None
        if entry_time_src is None:
            try:
                entry_time_src = df.index[-1]
            except Exception:
                entry_time_src = pd.Timestamp.utcnow()
        entry_time_sast = _to_sast(entry_time_src)

        reasons = []
        if strong_body:
            reasons.append("strong_body")
        if pattern_confirm:
            reasons.append("pattern_confirm")
        if sweep:
            reasons.append("liquidity_sweep")
        if micro_confirm:
            if micro_confirm.get("micro_heuristic_retest"):
                reasons.append("micro_retest")
            if micro_confirm.get("micro_patterns"):
                reasons.append("micro_patterns")
        if entry_point_edge is not None:
            reasons.append("edge_entry")
        if vwap_align_ok_edge:
            reasons.append("vwap_align")
        reason = " + ".join(reasons) if reasons else "confirmed"

        return {
            "confirmed": True,
            "entry_point": round(entry_point, 6),
            "entry_time_sast": entry_time_sast,
            "direction": direction,
            "sl": round(sl_val, 6) if sl_val is not None else None,
            "tp": round(tp_val, 6) if tp_val is not None else None,
            "atr": float(atr),
            "confidence": float(confidence),
            "reason": reason,
            # ETA / expected-bars predictions
            "expected_micro_bars": int(expected_micro_bars) if expected_micro_bars is not None else None,
            "expected_main_bars": int(expected_main_bars) if expected_main_bars is not None else None,
            "eta_time_sast": eta_time_sast,
            # Edge-entry refinement outputs
            "entry_point_edge": round(entry_point_edge, 6) if entry_point_edge is not None else None,
            "protected_sl": round(protected_sl, 6) if protected_sl is not None else None,
            "precision_grade": float(precision_grade) if precision_grade is not None else None,
            "precision_labels": precision_labels,
            "retest_zone": retest_zone,
            "fvg_zone": fvg_zone,
            "ob_level": float(ob_level) if ob_level is not None else None,
            "entry_time_refined_sast": entry_time_refined_sast,
            "vwap_at_refined": float(vwap_at_refined) if vwap_at_refined is not None else None,
            "debug": {
                "strong_body": bool(strong_body),
                "pattern_confirm": bool(pattern_confirm),
                "sweep": bool(sweep),
                "vol_ok": bool(vol_ok),
                "structure": struct,
                "micro_confirm": micro_confirm,
                "vwap_filter": {
                    "enabled": bool(vwap_filter_enable),
                    "available": bool(micro_confirm.get("micro_vwap") if micro_confirm else False),
                    "aligned": bool(vwap_align_ok_edge) if vwap_align_ok_edge is not None else None
                }
            }
        }
    except Exception:
        return None


def precision_analysis(df: pd.DataFrame, context: Optional[Dict[str, Any]] = None, expected_direction: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        res = detect_precision_entry(df, context=context, expected_direction=expected_direction, params=params or DEFAULTS.copy())
        if res is None:
            atr_val = _compute_atr(df, int((params or DEFAULTS).get("atr_period", DEFAULTS["atr_period"])))
            return {
                "entry_point": None,
                "entry_time_sast": None,
                "direction": "Wait",
                "sl": None,
                "tp": None,
                "atr": float(atr_val) if atr_val is not None else None,
                "confidence": 0.0,
                "reason": "precision_detection_failed"
            }

        if res.get("confirmed"):
            out = {
                "entry_point": float(res["entry_point"]),
                "entry_time_sast": res["entry_time_sast"],
                "direction": res["direction"],
                "sl": float(res["sl"]) if res.get("sl") is not None else None,
                "tp": float(res["tp"]) if res.get("tp") is not None else None,
                "atr": float(res.get("atr", 0.0)),
                "confidence": float(res.get("confidence", 0.0)),
                "reason": res.get("reason", ""),
                "expected_micro_bars": res.get("expected_micro_bars"),
                "expected_main_bars": res.get("expected_main_bars"),
                "eta_time_sast": res.get("eta_time_sast"),
                "entry_point_edge": res.get("entry_point_edge"),
                "protected_sl": res.get("protected_sl"),
                "precision_grade": res.get("precision_grade"),
                "precision_labels": res.get("precision_labels"),
                "retest_zone": res.get("retest_zone"),
                "fvg_zone": res.get("fvg_zone"),
                "ob_level": res.get("ob_level"),
                "entry_time_refined_sast": res.get("entry_time_refined_sast"),
                "vwap_at_refined": res.get("vwap_at_refined"),
                "debug": res.get("debug", {})
            }
            return out

        atr_val = _compute_atr(df, int((params or DEFAULTS).get("atr_period", DEFAULTS["atr_period"])))
        return {
            "entry_point": None,
            "entry_time_sast": None,
            "direction": "Wait",
            "sl": None,
            "tp": None,
            "atr": float(atr_val) if atr_val is not None else None,
            "confidence": 0.0,
            "reason": "no precision entry",
            "expected_micro_bars": res.get("expected_micro_bars"),
            "expected_main_bars": res.get("expected_main_bars"),
            "eta_time_sast": res.get("eta_time_sast"),
            "debug": res.get("debug")
        }
    except Exception as e:
        return {
            "entry_point": None,
            "entry_time_sast": None,
            "direction": "Wait",
            "sl": None,
            "tp": None,
            "atr": None,
            "confidence": 0.0,
            "reason": f"precision_analysis_error: {e}"
        }


def find_precise_entry(df: pd.DataFrame,
                       direction: str,
                       atr: Optional[float] = None,
                       params: Optional[Dict[str, Any]] = None,
                       context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        p = params or DEFAULTS.copy()
        try:
            env_apply = os.getenv("ANALYSIS_RETEST_EDGE_ENTRY")
            if env_apply is not None:
                p["apply_edge_as_entry"] = (env_apply != "0")
        except Exception:
            pass

        ctx = context or {}
        res = detect_precision_entry(df, context=ctx, expected_direction=direction, params=p)
        if isinstance(res, dict) and res.get("confirmed"):
            entry_price = float(res["entry_point"])
            out = {
                "entry_price": entry_price,
                "sl": float(res["sl"]) if res.get("sl") is not None else None,
                "tp": float(res["tp"]) if res.get("tp") is not None else None,
                "confidence": float(res.get("confidence", 0.6)),
                "reason": f"precision_confirmed:{res.get('reason','')}".strip(":"),
                "entry_time_sast": res.get("entry_time_sast"),
                "entry_point_edge": res.get("entry_point_edge"),
                "protected_sl": res.get("protected_sl"),
                "precision_grade": res.get("precision_grade"),
                "precision_labels": res.get("precision_labels"),
                "retest_zone": res.get("retest_zone"),
                "fvg_zone": res.get("fvg_zone"),
                "ob_level": res.get("ob_level"),
                "entry_time_refined_sast": res.get("entry_time_refined_sast"),
                "vwap_at_refined": res.get("vwap_at_refined"),
            }
            return out

        atr_val = float(atr) if atr is not None else (_compute_atr(df, int(p.get("atr_period", DEFAULTS["atr_period"]))) or 0.0)
        heur = detect_retest(df, "Buy" if str(direction).lower().startswith("b") else "Sell", atr_val, sensitivity=0.8)
        if heur:
            entry = float(heur.get("entry_price", df["close"].iloc[-1]))
            if str(direction).lower().startswith("b"):
                tp = entry + float(p.get("tp_atr_mult", DEFAULTS["tp_atr_mult"])) * (atr_val or 0.0)
            else:
                tp = entry - float(p.get("tp_atr_mult", DEFAULTS["tp_atr_mult"])) * (atr_val or 0.0)
            return {
                "entry_price": entry,
                "sl": float(heur.get("sl")) if heur.get("sl") is not None else None,
                "tp": float(tp),
                "confidence": float(heur.get("confidence", 0.5)),
                "reason": heur.get("reason", "tf_heuristic_retest"),
                "entry_time_sast": _to_sast(df.index[-1] if hasattr(df, "index") else pd.Timestamp.utcnow())
            }

        return None
    except Exception:
        return None
