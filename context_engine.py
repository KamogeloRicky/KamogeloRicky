"""
Context Engine (ultimate confluence, HTF-aware, execution-ready, safe fallbacks)

API (backward compatible)
- contextual_analysis(
    df,
    pair=None,
    timeframe=None,
    price=None,
    fetcher=None,        # optional: data.fetch_candles for HTF sampling
    htf="4h",
    config=None,         # optional dict to override weights/thresholds
    news_events=None     # optional [{'time': pd.Timestamp|str, 'impact':'High'|'Medium'|'Low'}]
  ) -> dict

Key upgrades
- Liquidity mapping: recent equal highs/lows, previous day high/low/close (PDH/PDL/PDC), session open
- Anchored VWAPs: daily/week anchored VWAP (falls back to TWAP if volume missing)
- FVG (fair value gap) proximity + Fib confluence + golden pocket awareness
- Liquidity sweeps and NR7 compression detection for timing edge
- HTF bias blending (soft boost only when aligned; no IO unless fetcher provided)
- Session-aware weighting (London open/overlap boost) and news blackout damping
- Execution plan: direction, best entry zone, invalidation, SL/TP1/TP2, position size suggestion
- Confidence calibrated by confluence count, regime, HTF alignment, news proximity
- Fully defensive: runs without ta, without volume/time columns; never mutates input df

Returns (additions)
- score_breakdown: component contributions
- confidence: [0,1]
- confluence_count: how many major edges agree now
- htf_bias: Buy/Sell/Neutral/None
- fib, fvg, liquidity_map, vwap: detailed dicts
- entry_plan: {direction, entry_zone, sl, tp1, tp2, size_suggestion, invalidation, alerts}
- regime: {volatility, compression, nr7}
"""

from __future__ import annotations

import os
import math
import datetime as dt
from typing import Any, Dict, Optional, Callable, List

import numpy as np
import pandas as pd

# Optional deps
try:
    import pytz  # type: ignore
except Exception:
    pytz = None  # type: ignore

try:
    import ta  # type: ignore
except Exception:
    ta = None  # type: ignore


# -------------------- SAFE/CORE HELPERS --------------------
def _ema(series: pd.Series, window: int) -> pd.Series:
    try:
        if ta is not None:
            return ta.trend.EMAIndicator(series.astype(float), window=window).ema_indicator()
        return series.astype(float).ewm(span=window, adjust=False).mean()
    except Exception:
        return series.astype(float).rolling(window).mean()


def _atr(df: pd.DataFrame, window: int = 14) -> float:
    try:
        if ta is not None:
            atr_series = ta.volatility.AverageTrueRange(
                df["high"].astype(float),
                df["low"].astype(float),
                df["close"].astype(float),
                window=window
            ).average_true_range()
            val = float(atr_series.iloc[-1])
            return val if math.isfinite(val) else 0.0
        # fallback
        tr = (df["high"].astype(float) - df["low"].astype(float))
        val = float(tr.tail(window).mean())
        return val if math.isfinite(val) else 0.0
    except Exception:
        return 0.0


def _to_session_tz(ts: Optional[dt.datetime]) -> dt.datetime:
    try:
        tzname = os.getenv("ANALYSIS_SESSION_TZ", "Africa/Johannesburg")
        if ts is None:
            ts = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        if pytz is not None:
            tz = pytz.timezone(tzname)
            return ts.astimezone(tz)
        return ts.astimezone(dt.timezone(dt.timedelta(hours=2)))
    except Exception:
        x = ts or dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        return x if x.tzinfo else x.replace(tzinfo=dt.timezone.utc)


def _session_name(now_local: dt.datetime) -> str:
    try:
        h = int(now_local.hour)
        if 8 <= h < 12:
            return "London Open"
        if 12 <= h < 17:
            return "London–NY Overlap"
        if 17 <= h < 22:
            return "New York"
        return "Asian"
    except Exception:
        return "Unknown"


# -------------------- LIQUIDITY / LEVELS --------------------
def _equal_highs_lows(df: pd.DataFrame, look: int = 50, tol_frac: float = 0.0001) -> Dict[str, Any]:
    out = {"equal_highs": [], "equal_lows": []}
    try:
        seg = df.tail(max(look, 10))
        highs = seg["high"].astype(float).values
        lows = seg["low"].astype(float).values
        price = float(seg["close"].iloc[-1])
        tol = max(price * tol_frac, 1e-9)
        # simple adjacency equality
        for i in range(1, len(highs)):
            if abs(highs[i] - highs[i-1]) <= tol:
                out["equal_highs"].append((seg.index[i-1], float(highs[i-1])))
        for i in range(1, len(lows)):
            if abs(lows[i] - lows[i-1]) <= tol:
                out["equal_lows"].append((seg.index[i-1], float(lows[i-1])))
    except Exception:
        pass
    return out


def _previous_day_levels(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Compute previous day high/low/close using 'time' if present; else fallback by index date.
    """
    res = {"PDH": None, "PDL": None, "PDC": None}
    try:
        if "time" in df.columns:
            t = pd.to_datetime(df["time"])
            df2 = df.copy()
            df2["day"] = t.dt.date
        else:
            idx = pd.to_datetime(df.index)
            df2 = df.copy()
            df2["day"] = idx.date
        # last day present
        days = df2["day"].unique()
        if len(days) < 2:
            return res
        prev_day = days[-2]
        d = df2[df2["day"] == prev_day]
        if len(d) == 0:
            return res
        res["PDH"] = float(d["high"].max())
        res["PDL"] = float(d["low"].min())
        res["PDC"] = float(d["close"].iloc[-1])
        return res
    except Exception:
        return res


def _session_open_level(df: pd.DataFrame) -> Optional[float]:
    """
    Price at session open (approx last midnight local TZ).
    """
    try:
        if "time" in df.columns:
            t = pd.to_datetime(df["time"])
        else:
            t = pd.to_datetime(df.index)
        now_local = _to_session_tz(None)
        date_local = now_local.date()
        # bars for today's date in local TZ
        tz = now_local.tzinfo
        t_local = t.dt.tz_localize("UTC").dt.tz_convert(tz) if t.dt.tz is None else t.dt.tz_convert(tz)  # type: ignore
        today_mask = (t_local.dt.date == date_local)
        today = df.loc[today_mask]
        if len(today) > 0:
            return float(today["open"].iloc[0])
        return None
    except Exception:
        return None


# -------------------- VWAP / FVG / FIB --------------------
def _anchored_vwap(df: pd.DataFrame, anchor: str = "day") -> Optional[float]:
    """
    Anchored VWAP from session/day/week open.
    If volume missing, uses TWAP of typical price as fallback (approximation).
    """
    try:
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        if "time" in df.columns:
            t = pd.to_datetime(df["time"])
        else:
            t = pd.to_datetime(df.index)
        tz_now = _to_session_tz(None).tzinfo
        t_local = t.dt.tz_localize("UTC").dt.tz_convert(tz_now) if t.dt.tz is None else t.dt.tz_convert(tz_now)  # type: ignore
        if anchor == "day":
            mask = (t_local.dt.date == t_local.iloc[-1].date())
        elif anchor == "week":
            mask = (t_local.dt.isocalendar().week == t_local.iloc[-1].isocalendar().week)  # type: ignore
        else:
            mask = np.ones(len(df), dtype=bool)
        tp_seg = tp[mask]
        if len(tp_seg) == 0:
            return None
        if "volume" in df.columns:
            vol_seg = pd.to_numeric(df["volume"][mask], errors="coerce").fillna(0.0)
            denom = vol_seg.cumsum().replace(0, np.nan)
            vwap = (tp_seg * vol_seg).cumsum() / denom
            return float(vwap.iloc[-1]) if math.isfinite(float(vwap.iloc[-1])) else None
        # TWAP fallback
        twap = float(tp_seg.mean())
        return twap if math.isfinite(twap) else None
    except Exception:
        return None


def _detect_fvg(df: pd.DataFrame, look: int = 50) -> List[Dict[str, Any]]:
    """
    Simple two-bar FVG detector (aligned with advanced_signals).
    """
    gaps: List[Dict[str, Any]] = []
    try:
        n = min(len(df), look)
        start = len(df) - n
        for idx in range(start, len(df) - 1):
            c1_open = float(df["open"].iat[idx]); c1_close = float(df["close"].iat[idx])
            c2_open = float(df["open"].iat[idx+1]); c2_close = float(df["close"].iat[idx+1])
            if c1_close > c1_open and c2_close > c2_open:
                top1 = max(c1_open, c1_close); bot2 = min(c2_open, c2_close)
                if bot2 > top1:
                    gaps.append({"type": "bull", "start_idx": idx, "end_idx": idx+1, "low": top1, "high": bot2})
            if c1_close < c1_open and c2_close < c2_open:
                bot1 = min(c1_open, c1_close); top2 = max(c2_open, c2_close)
                if top2 < bot1:
                    gaps.append({"type": "bear", "start_idx": idx, "end_idx": idx+1, "low": top2, "high": bot1})
    except Exception:
        pass
    return gaps


def _recent_swing(df: pd.DataFrame, look: int = 80) -> Optional[Dict[str, float]]:
    try:
        seg = df.tail(max(look, 20))
        hp = float(seg["high"].max())
        lp = float(seg["low"].min())
        if not math.isfinite(hp) or not math.isfinite(lp) or hp <= lp:
            return None
        return {"high": hp, "low": lp, "range": hp - lp}
    except Exception:
        return None


def _fib_levels_from_swing(high_price: float, low_price: float, direction_up: bool) -> Dict[str, float]:
    """
    Compute key Fibonacci retracement levels for the swing.
    """
    try:
        hp = float(high_price); lp = float(low_price)
        rng = hp - lp
        if rng <= 0:
            return {}
        levels = {}
        key = [0.236, 0.382, 0.5, 0.618, 0.65, 0.786]
        if direction_up:
            # swing low -> high; retrace down
            for l in key:
                levels[str(l)] = hp - l * rng
            levels["0.0"] = hp; levels["1.0"] = lp
        else:
            # swing high -> low; retrace up
            for l in key:
                levels[str(l)] = lp + l * rng
            levels["0.0"] = lp; levels["1.0"] = hp
        return levels
    except Exception:
        return {}


def _fib_confluence(df: pd.DataFrame, direction_up: Optional[bool], atr_val: float) -> Dict[str, Any]:
    """
    Build Fibonacci data from most recent min/max swing.
    """
    out = {"available": False}
    try:
        swing = _recent_swing(df, 80)
        if not swing:
            return out
        if direction_up is None:
            # infer from last two closes
            direction_up = bool(df["close"].iloc[-1] >= df["close"].iloc[-2])
        levels = _fib_levels_from_swing(swing["high"], swing["low"], direction_up)
        price = float(df["close"].iloc[-1])
        tol = max(atr_val * 0.25, price * 0.0003)

        priority = ["0.382", "0.5", "0.618", "0.65", "0.786"]
        nearest = None
        min_d = float("inf")
        for k in priority:
            v = levels.get(k)
            if v is None:
                continue
            d = abs(price - float(v))
            if d < min_d:
                min_d = d; nearest = (k, float(v))

        gp = {"low": levels.get("0.65"), "high": levels.get("0.618"), "inside": False}
        if gp["low"] is not None and gp["high"] is not None:
            lo, hi = sorted([float(gp["low"]), float(gp["high"])])
            gp["inside"] = (lo <= price <= hi)

        out = {
            "available": True,
            "levels": levels,
            "nearest": {"level": nearest[0] if nearest else None,
                        "price": nearest[1] if nearest else None,
                        "distance": float(min_d) if nearest else None,
                        "within_tolerance": bool(nearest and min_d <= tol)},
            "golden_pocket": gp,
            "tolerance": float(tol),
        }
        return out
    except Exception:
        return out


def _liquidity_sweep(df: pd.DataFrame, look: int = 20) -> Dict[str, Any]:
    out = {"swept_high": False, "swept_low": False, "details": {}}
    try:
        recent = df.tail(max(look, 5))
        hi = float(recent["high"].iloc[:-1].max())
        lo = float(recent["low"].iloc[:-1].min())
        last = recent.iloc[-1]
        swept_high = (float(last["high"]) > hi) and (float(last["close"]) < hi)
        swept_low = (float(last["low"]) < lo) and (float(last["close"]) > lo)
        out["swept_high"] = bool(swept_high)
        out["swept_low"] = bool(swept_low)
        out["details"] = {"recent_high": hi, "recent_low": lo}
    except Exception:
        pass
    return out


def _nr7_compression(df: pd.DataFrame) -> bool:
    """
    NR7: current range is the narrowest of last 7 bars -> compression signal.
    """
    try:
        rng = (df["high"] - df["low"]).astype(float).tail(7)
        if len(rng) < 7: return False
        return bool(rng.iloc[-1] == rng.min())
    except Exception:
        return False


def _compression_proxy(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple compression/expansion proxy using rolling std of returns.
    """
    out = {"compression": False, "zscore": 0.0}
    try:
        closes = df["close"].astype(float).tail(200)
        if len(closes) < 20:
            return out
        ret = closes.pct_change().dropna()
        std20 = float(ret.rolling(20).std().iloc[-1])
        std80 = float(ret.rolling(80).std().iloc[-1]) if len(ret) >= 80 else float(ret.rolling(40).std().iloc[-1])
        if std80 == 0:
            return out
        ratio = std20 / (std80 + 1e-12)
        # compression when short-term vol much lower than baseline
        out["compression"] = bool(ratio < 0.6)
        out["zscore"] = float(round((ratio - 1.0), 3))
    except Exception:
        pass
    return out


def _htf_bias(fetcher: Optional[Callable[[str, str, int], Optional[pd.DataFrame]]],
              pair: Optional[str], htf: str) -> Optional[str]:
    """
    Optional HTF alignment via external fetcher (no IO by default).
    """
    try:
        if fetcher is None or not pair:
            return None
        dfh = fetcher(pair, htf, 300)
        if not isinstance(dfh, pd.DataFrame) or dfh.empty:
            return None
        close = pd.to_numeric(dfh["close"], errors="coerce").dropna()
        if len(close) < 55:
            return None
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        return "Buy" if ema20 > ema50 else "Sell"
    except Exception:
        return None


def _news_blackout_penalty(now_ts: Optional[pd.Timestamp], news_events: Optional[List[Dict[str, Any]]], window_min: int = 45) -> float:
    """
    Returns penalty in [0, 0.2] if within blackout window of high-impact news.
    """
    try:
        if not news_events or now_ts is None: return 0.0
        t = pd.to_datetime(now_ts)
        for ev in news_events:
            ev_t = pd.to_datetime(ev.get("time"))
            imp = str(ev.get("impact", "Low")).lower()
            if not isinstance(ev_t, pd.Timestamp): continue
            if imp.startswith("high"):
                delta = abs((t - ev_t).total_seconds()) / 60.0
                if delta <= window_min:
                    return 0.2
        return 0.0
    except Exception:
        return 0.0


# -------------------- MAIN --------------------
def contextual_analysis(df: pd.DataFrame,
                        pair: Optional[str] = None,
                        timeframe: Optional[str] = None,
                        price: Optional[float] = None,
                        fetcher: Optional[Callable[[str, str, int], Optional[pd.DataFrame]]] = None,
                        htf: str = "4h",
                        config: Optional[Dict[str, Any]] = None,
                        news_events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "session": None,
        "signal": "Neutral",
        "score": 0.0,
        "score_breakdown": {},
        "retest": {"status": "none", "level": None, "confidence": 0.0},
        "entry_time": None,
        "notes": [],
        "confidence": 0.5,
        "confluence_count": 0,
        "htf_bias": None,
        "liquidity_sweep": {"swept_high": False, "swept_low": False, "details": {}},
        "fib": {"available": False},
        "fvg": [],
        "liquidity_map": {},
        "levels": {},
        "vwap": {"day": None, "week": None},
        "regime": {"volatility": None, "compression": False, "nr7": False, "details": {}},
        "entry_plan": {},
        "details": {},
    }

    # weights/params (overridable via config)
    W = {
        "bos_cont": 0.35,
        "choch_rev": -0.35,
        "retest": 0.25,
        "pattern_up": 0.2,
        "pattern_dn": -0.2,
        "liquidity_sweep": 0.2,
        "fib_confluence": 0.25,
        "compression": 0.1,
        "nr7": 0.08,
        "htf_align": 0.2,
        "volatility_penalty": -0.1,
        "fvg_near": 0.15,
        "pdh_pdl_reaction": 0.15,
        "vwap_align": 0.15,
    }
    if isinstance(config, dict) and "weights" in config and isinstance(config["weights"], dict):
        W.update(config["weights"])

    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            result["error"] = "DataFrame empty or None"
            return result

        # Work on a copy; ensure price columns are numeric
        data = df.copy()
        for col in ("open", "high", "low", "close"):
            if col not in data.columns:
                result["error"] = f"Missing column: {col}"
                return result
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=["open", "high", "low", "close"])
        if len(data) < 20:
            result["error"] = "Insufficient bars (need >= 20)"
            return result

        # Session and timestamp
        now_local = _to_session_tz(None)
        session = _session_name(now_local)
        result["session"] = session
        now_ts = pd.to_datetime(data["time"].iloc[-1]) if "time" in data.columns else pd.to_datetime(data.index[-1])

        # Structure: CHoCH + BOS
        ema_fast = _ema(data["close"], 12)
        ema_slow = _ema(data["close"], 26)
        if len(ema_fast.dropna()) < 2 or len(ema_slow.dropna()) < 2:
            choch = False
        else:
            prev_fast, prev_slow = float(ema_fast.iloc[-2]), float(ema_slow.iloc[-2])
            last_fast, last_slow = float(ema_fast.iloc[-1]), float(ema_slow.iloc[-1])
            choch = (prev_fast < prev_slow and last_fast > last_slow) or (prev_fast > prev_slow and last_fast < last_slow)

        window_n = 10
        recent_high = float(data["high"].iloc[-window_n:].max())
        recent_low = float(data["low"].iloc[-window_n:].min())
        last_close = float(data["close"].iloc[-1]) if price is None else float(price)
        bos_up = last_close > recent_high
        bos_dn = last_close < recent_low
        bos = bool(bos_up or bos_dn)

        score = 0.0
        breakdown: Dict[str, float] = {}
        confluence = 0

        if bos and not choch:
            score += W["bos_cont"]; breakdown["bos_cont"] = W["bos_cont"]; confluence += 1
            result["notes"].append("BOS detected → continuation bias")
        elif choch:
            score += W["choch_rev"]; breakdown["choch_rev"] = W["choch_rev"]; confluence += 1
            result["notes"].append("CHoCH detected → possible reversal")

        # Retest near contextual level (BOS level or mid zone)
        mid_zone = (recent_high + recent_low) / 2.0
        retest_level = recent_high if bos_up else recent_low if bos_dn else mid_zone
        atr_val = _atr(data, 14)
        tol = max(atr_val * 0.5, last_close * 0.0002)
        dist_to_retest = abs(last_close - retest_level)
        if dist_to_retest <= tol:
            proximity = max(0.0, 1.0 - (dist_to_retest / (tol + 1e-9)))
            result["retest"] = {"status": "retest_in_progress", "level": float(round(retest_level, 6)), "confidence": float(round(0.6 + 0.4 * proximity, 3))}
            score += W["retest"] * (0.5 + 0.5 * proximity); breakdown["retest"] = W["retest"] * (0.5 + 0.5 * proximity); confluence += 1
            result["notes"].append("Retest in play (within ATR tolerance)")
        else:
            result["retest"] = {"status": "waiting_next_retest", "level": float(round(retest_level, 6)), "confidence": 0.3}

        # Pattern bias via last-3 closes
        pattern = "Sideways / Wedge"
        try:
            closes = data["close"].tail(10).values.astype(float)
            if len(closes) >= 3:
                d3 = np.diff(closes[-3:])
                if np.all(d3 > 0):
                    pattern = "Ascending Flag"
                    score += W["pattern_up"]; breakdown["pattern_up"] = W["pattern_up"]; confluence += 1
                elif np.all(d3 < 0):
                    pattern = "Descending Flag"
                    score += W["pattern_dn"]; breakdown["pattern_dn"] = W["pattern_dn"]; confluence += 1
            if len(closes) >= 5:
                if closes[-5] < closes[-4] and closes[-4] > closes[-3] and closes[-3] < closes[-2]:
                    pattern = "Head & Shoulders"
                    score += W["pattern_dn"]; breakdown["pattern_dn(H&S)"] = W["pattern_dn"]; result["notes"].append("H&S heuristic → bearish bias"); confluence += 1
        except Exception:
            pass
        result["pattern"] = pattern

        # Liquidity sweep
        sweep = _liquidity_sweep(data, 20)
        result["liquidity_sweep"] = sweep
        if sweep["swept_high"] or sweep["swept_low"]:
            score += W["liquidity_sweep"]; breakdown["liquidity_sweep"] = W["liquidity_sweep"]; confluence += 1
            result["notes"].append("Liquidity sweep detected")

        # FVG proximity
        fvg = _detect_fvg(data, 50)
        result["fvg"] = fvg
        try:
            near_fvg = False
            for g in fvg[-5:]:
                if float(g["low"]) <= last_close <= float(g["high"]):
                    near_fvg = True; break
            if near_fvg:
                score += W["fvg_near"]; breakdown["fvg_near"] = W["fvg_near"]; confluence += 1
                result["notes"].append("Price within FVG")
        except Exception:
            pass

        # Fib confluence
        dir_up = True if bos_up else False if bos_dn else None
        fib = _fib_confluence(data, dir_up, atr_val)
        result["fib"] = fib
        if fib.get("available"):
            near = fib.get("nearest", {}).get("within_tolerance")
            in_gp = fib.get("golden_pocket", {}).get("inside")
            if near or in_gp:
                score += W["fib_confluence"]; breakdown["fib_confluence"] = W["fib_confluence"]; confluence += 1
                if in_gp:
                    result["notes"].append("In golden pocket")

        # Levels: PDH/PDL/PDC and session open
        levels = _previous_day_levels(data)
        levels["SESSION_OPEN"] = _session_open_level(data)
        result["levels"] = levels
        try:
            # reaction near PDH/PDL
            if levels["PDH"] and abs(last_close - float(levels["PDH"])) <= tol:
                score += W["pdh_pdl_reaction"]; breakdown["near_PDH"] = W["pdh_pdl_reaction"]; confluence += 1
            if levels["PDL"] and abs(last_close - float(levels["PDL"])) <= tol:
                score += W["pdh_pdl_reaction"]; breakdown["near_PDL"] = W["pdh_pdl_reaction"]; confluence += 1
        except Exception:
            pass

        # Anchored VWAP day/week
        vwap_day = _anchored_vwap(data, "day")
        vwap_week = _anchored_vwap(data, "week")
        result["vwap"] = {"day": vwap_day, "week": vwap_week}
        try:
            if vwap_day is not None:
                # align with BOS direction
                vwap_align = (last_close >= vwap_day) if bos_up else (last_close <= vwap_day) if bos_dn else None
                if vwap_align:
                    score += W["vwap_align"]; breakdown["vwap_align"] = W["vwap_align"]; confluence += 1
        except Exception:
            pass

        # Equal highs/lows map (liquidity pools)
        result["liquidity_map"] = _equal_highs_lows(data, 60, 0.0001)

        # Regime: ATR vs price, compression proxies
        v_regime = "High" if atr_val > last_close * 0.002 else "Normal"
        compression = _compression_proxy(data)
        nr7 = _nr7_compression(data)
        result["regime"] = {"volatility": v_regime, "compression": bool(compression["compression"]), "nr7": bool(nr7), "details": compression}
        if compression["compression"]:
            score += W["compression"]; breakdown["compression"] = W["compression"]; confluence += 1
            result["notes"].append("Compression (vol contraction)")
        if nr7:
            score += W["nr7"]; breakdown["nr7"] = W["nr7"]; confluence += 1
            result["notes"].append("NR7 (narrow range)")

        if v_regime == "High":
            score += W["volatility_penalty"]; breakdown["volatility_penalty"] = W["volatility_penalty"]
            result["notes"].append("High volatility → reduce size")

        # HTF bias (optional)
        htf_dir = _htf_bias(fetcher, pair, htf) if pair else None
        result["htf_bias"] = htf_dir
        if htf_dir in ("Buy", "Sell"):
            if (htf_dir == "Buy" and bos_up) or (htf_dir == "Sell" and bos_dn):
                score += W["htf_align"]; breakdown["htf_align"] = W["htf_align"]; confluence += 1

        # Session boost
        if session in ("London Open", "London–NY Overlap"):
            boost = 0.05
            if "bos_cont" in breakdown: score += boost; breakdown["session_boost_bos"] = boost
            if "retest" in breakdown: score += boost; breakdown["session_boost_retest"] = boost

        # News blackout penalty
        penalty = _news_blackout_penalty(pd.to_datetime(now_ts), news_events, window_min=45)
        if penalty > 0:
            score -= penalty
            breakdown["news_blackout_penalty"] = -penalty
            result["notes"].append("Nearby high-impact news → penalty")

        # Timing / entry hint
        delay_min = 15 if result["retest"]["status"] == "retest_in_progress" else 30
        entry_time_local = now_local + dt.timedelta(minutes=delay_min)
        result["entry_time"] = entry_time_local.strftime("%Y-%m-%d %H:%M:%S")

        # Final signal + confidence
        result["score_breakdown"] = {k: float(round(v, 4)) for k, v in breakdown.items()}
        score = float(round(max(-1.0, min(1.0, score)), 3))
        result["score"] = score
        if score > 0.3:
            result["signal"] = "Buy Bias"
        elif score < -0.3:
            result["signal"] = "Sell Bias"
        else:
            result["signal"] = "Neutral"

        # Confidence calibration
        mag = min(1.0, abs(score))
        conf = 0.45 + 0.35 * mag + 0.03 * confluence
        if htf_dir in ("Buy", "Sell") and result["signal"] in ("Buy Bias", "Sell Bias"):
            same = (htf_dir == "Buy" and result["signal"] == "Buy Bias") or (htf_dir == "Sell" and result["signal"] == "Sell Bias")
            conf += 0.08 if same else -0.04
        if v_regime == "High": conf -= 0.05
        conf = float(max(0.05, min(0.97, conf - penalty)))
        result["confidence"] = conf
        result["confluence_count"] = int(confluence)

        # Execution plan (directional template + position size suggestion)
        direction = "Buy" if score >= 0.2 else "Sell" if score <= -0.2 else "Wait"
        entry_zone = float(round(result["retest"]["level"], 6)) if result["retest"]["level"] is not None else float(round(mid_zone, 6))
        atr = atr_val or max(1e-9, abs(recent_high - recent_low) * 0.2)
        if direction == "Buy":
            sl = float(round(entry_zone - 1.2 * atr, 6))
            tp1 = float(round(entry_zone + 1.5 * atr, 6))
            tp2 = float(round(entry_zone + 3.0 * atr, 6))
            invalidation = float(round(recent_low - 0.2 * atr, 6))
        elif direction == "Sell":
            sl = float(round(entry_zone + 1.2 * atr, 6))
            tp1 = float(round(entry_zone - 1.5 * atr, 6))
            tp2 = float(round(entry_zone - 3.0 * atr, 6))
            invalidation = float(round(recent_high + 0.2 * atr, 6))
        else:
            sl = tp1 = tp2 = invalidation = None

        # position size suggestion (if env has balance and desired risk%)
        size_suggestion = None
        try:
            balance = float(os.getenv("ANALYSIS_ACCOUNT_BALANCE", os.getenv("BACKTEST_ACCOUNT_BALANCE", "1000")))
            risk_pct = float(os.getenv("ANALYSIS_RISK_PCT", os.getenv("BACKTEST_RISK_PCT", "1.0"))) / 100.0
            pip_value = float(os.getenv("ANALYSIS_PIP_VALUE", os.getenv("BACKTEST_PIP_VALUE", "10")))
            point = float(os.getenv("ANALYSIS_POINT_SIZE", "0.0001"))
            sl_pips = abs(entry_zone - sl) / (point if point else 1e-9) if (sl is not None and entry_zone is not None) else None
            if sl_pips and sl_pips > 0:
                risk_amount = balance * risk_pct
                lots = max(0.01, round(risk_amount / (sl_pips * pip_value), 2))
                size_suggestion = float(lots)
        except Exception:
            pass

        alerts = []
        try:
            if fib.get("available"):
                near = fib["nearest"]
                if near.get("level") and near.get("within_tolerance"):
                    alerts.append(f"Price near Fib {near['level']} @ {round(near['price'],6)}")
                gp = fib.get("golden_pocket", {})
                if gp.get("inside"):
                    lo, hi = sorted([float(gp["low"]), float(gp["high"])])
                    alerts.append(f"In golden pocket [{round(lo,6)} - {round(hi,6)}]")
            if vwap_day is not None:
                alerts.append(f"Day VWAP @ {round(vwap_day,6)}")
            if levels.get("PDH"): alerts.append(f"PDH @ {round(levels['PDH'],6)}")
            if levels.get("PDL"): alerts.append(f"PDL @ {round(levels['PDL'],6)}")
        except Exception:
            pass

        result["entry_plan"] = {
            "direction": direction,
            "entry_zone": entry_zone,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "invalidation": invalidation,
            "size_suggestion": size_suggestion,
            "alerts": alerts
        }

        # Traceability
        result["details"] = {
            "pair": pair,
            "timeframe": timeframe,
            "last_close": float(round(last_close, 6)),
            "recent_high": float(round(recent_high, 6)),
            "recent_low": float(round(recent_low, 6)),
            "bos_up": bool(bos_up),
            "bos_dn": bool(bos_dn),
            "choch": bool(choch),
            "atr": float(round(atr_val, 6)),
            "tolerance": float(round(tol, 6)),
            "htf": htf,
        }

        return result

    except Exception as e:
        return {"error": str(e)}
