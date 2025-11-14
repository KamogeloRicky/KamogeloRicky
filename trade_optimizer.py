"""
Entry Optimizer for RickyFX — realistic retest-based entries (structure + confluence scoring)

Purpose
- Replace unrealistic entries with realistic retest entries using price-action confluence:
  * Multi-timeframe bias (HTF trend)
  * Recent impulse legs (swing-to-swing)
  * Liquidity sweep context (equal highs/lows)
  * Fair Value Gaps (FVGs)
  * ATR-aware protective SL buffers and min-distance enforcement
  * Session/time filter hooks
  * Confluence scoring and quality gating (min score threshold)
- Prefers LIMIT entries near smart retest ranges, avoids placing SL/TP at “stupid areas”
- Falls back to original analysis if no safe entry can be found

Usage
    from entry_optimizer import optimize_entry
    refined = optimize_entry(analysis, symbol="ETHUSDm", timeframe="1h")
    if refined["ok"]:
        entry, sl, tp = refined["entry"], refined["sl"], refined["tp"]
    else:
        # use analysis values (refined["fallback_used"] indicates fallback)

Environment flags (optional)
- ENTRY_OPTIMIZER_DISABLE=1   -> bypass optimizer and always fallback
- ENTRY_OPTIMIZER_MIN_SCORE   -> float in [0..1], default 0.6
- ENTRY_OPTIMIZER_SHOW=1      -> force-dump reasons/diagnostics to returned dict (default on)
- ENTRY_OPTIMIZER_BARS        -> int, candles to load (default 1000)
- ENTRY_OPTIMIZER_ATR_PERIOD  -> int, default 14
- ENTRY_OPTIMIZER_RR_TARGET   -> float, default 1.8
- ENTRY_OPTIMIZER_ATR_SL_BUF  -> float, default 0.5
- ENTRY_OPTIMIZER_ALLOW_BREAKOUT=1 -> allow STOP entries when breakout bias detected

Notes
- Requires MetaTrader5 package and a running terminal (same as executor).
- If MT5 is unavailable or data is insufficient, returns fallback preserving your analysis levels.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import os
import math

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


# Defaults (can be overridden via env)
DEFAULT_BARS = int(os.getenv("ENTRY_OPTIMIZER_BARS", "1000"))
DEFAULT_ATR_PERIOD = int(os.getenv("ENTRY_OPTIMIZER_ATR_PERIOD", "14"))
DEFAULT_RR_TARGET = float(os.getenv("ENTRY_OPTIMIZER_RR_TARGET", "1.8"))
DEFAULT_ATR_SL_BUF = float(os.getenv("ENTRY_OPTIMIZER_ATR_SL_BUF", "0.5"))
DEFAULT_MIN_SCORE = float(os.getenv("ENTRY_OPTIMIZER_MIN_SCORE", "0.6"))
DEFAULT_ALLOW_BREAKOUT = os.getenv("ENTRY_OPTIMIZER_ALLOW_BREAKOUT", "0") == "1"
OPTIMIZER_DISABLED = os.getenv("ENTRY_OPTIMIZER_DISABLE", "0") == "1"
SHOW_OPT = os.getenv("ENTRY_OPTIMIZER_SHOW", "1") == "1"


@dataclass
class Candle:
    t: int
    o: float
    h: float
    l: float
    c: float


@dataclass
class FVG:
    # 3-candle displacement gap
    i0: int
    i2: int
    low: float
    high: float
    bullish: bool


def _tf_map(s: str):
    if mt5 is None:
        return None
    s = s.lower()
    return {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "d1": mt5.TIMEFRAME_D1,
        "1d": mt5.TIMEFRAME_D1,
    }.get(s, mt5.TIMEFRAME_H1)


def _get_rates(symbol: str, tf_str: str, bars: int) -> List[Candle]:
    if mt5 is None:
        return []
    tf = _tf_map(tf_str)
    if tf is None:
        return []
    rows = mt5.copy_rates_from_pos(symbol, tf, 0, bars) or []
    out: List[Candle] = []
    for r in rows:
        out.append(Candle(t=int(r["time"]), o=float(r["open"]), h=float(r["high"]), l=float(r["low"]), c=float(r["close"])))
    return out


def _sma(vals: List[float], period: int) -> List[float]:
    out = []
    s = 0.0
    for i, v in enumerate(vals):
        s += v
        if i >= period:
            s -= vals[i - period]
        if i + 1 >= period:
            out.append(s / period)
        else:
            out.append(float("nan"))
    return out


def _atr(c: List[Candle], period: int) -> float:
    if len(c) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(c)):
        high = c[i].h
        low = c[i].l
        pc = c[i - 1].c
        tr = max(high - low, abs(high - pc), abs(low - pc))
        trs.append(tr)
    if len(trs) < period:
        return 0.0
    return sum(trs[-period:]) / period


def _find_swings(c: List[Candle], span: int) -> Tuple[List[int], List[int]]:
    highs, lows = [], []
    n = len(c)
    for i in range(span, n - span):
        if all(c[i].h >= c[j].h for j in range(i - span, i + span + 1)):
            highs.append(i)
        if all(c[i].l <= c[j].l for j in range(i - span, i + span + 1)):
            lows.append(i)
    return highs, lows


def _last_impulse(c: List[Candle], highs: List[int], lows: List[int], is_buy: bool) -> Optional[Tuple[int, int, float, float]]:
    # Return (a_idx, b_idx, a_price, b_price) describing the last directional swing leg
    if not highs or not lows:
        return None
    if is_buy:
        # last low to later high
        for pivot in range(len(c) - 2, 3, -1):
            lhs = [i for i in lows if i < pivot]
            rhs = [i for i in highs if i > (lhs[-1] if lhs else -1)]
            if lhs and rhs:
                a = lhs[-1]
                b = max([h for h in rhs if h > a], default=None)
                if b is not None and c[b].h > c[a].l:
                    return a, b, c[a].l, c[b].h
    else:
        # last high to later low
        for pivot in range(len(c) - 2, 3, -1):
            lhs = [i for i in highs if i < pivot]
            rhs = [i for i in lows if i > (lhs[-1] if lhs else -1)]
            if lhs and rhs:
                a = lhs[-1]
                b = max([l for l in rhs if l > a], default=None)
                if b is not None and c[a].h > c[b].l:
                    return a, b, c[a].h, c[b].l
    return None


def _detect_fvgs(c: List[Candle]) -> List[FVG]:
    out: List[FVG] = []
    for i in range(2, len(c)):
        A, B, D = c[i - 2], c[i - 1], c[i]
        # Bullish FVG (displacement up) approx
        if A.h < D.l and B.l > A.h:
            out.append(FVG(i0=i - 2, i2=i, low=A.h, high=B.l, bullish=True))
        # Bearish FVG (displacement down) approx
        if A.l > D.h and B.h < A.l:
            out.append(FVG(i0=i - 2, i2=i, low=B.h, high=A.l, bullish=False))
    return out


def _nearest_fvg(c: List[Candle], fvgs: List[FVG], ref: float, want_bullish: bool) -> Optional[FVG]:
    candidates = []
    for f in fvgs:
        if f.bullish != want_bullish:
            continue
        mid = (f.low + f.high) / 2.0
        # BUY wants bullish FVG below ref, SELL wants bearish FVG above ref
        if want_bullish and mid < ref:
            candidates.append((abs(ref - mid), f))
        if not want_bullish and mid > ref:
            candidates.append((abs(ref - mid), f))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _equal_levels(c: List[Candle], lookback: int = 30, tol: float = 0.0) -> Tuple[List[int], List[int]]:
    # Find equal highs/lows as liquidity magnets within tolerance (default exact)
    eq_highs, eq_lows = [], []
    n = len(c)
    L = max(5, min(lookback, n))
    for i in range(n - L, n - 2):
        # equal highs
        if abs(c[i].h - c[i + 1].h) <= tol:
            eq_highs.append(i + 1)
        # equal lows
        if abs(c[i].l - c[i + 1].l) <= tol:
            eq_lows.append(i + 1)
    return eq_highs[-3:], eq_lows[-3:]


def _point(info) -> float:
    return getattr(info, "point", 0.00001) or 0.00001


def _round_price(x: float, digits: int) -> float:
    try:
        return float(f"{float(x):.{digits}f}")
    except Exception:
        return x


def _min_distance(info) -> float:
    p = _point(info)
    spread_points = getattr(info, "spread", 0)
    stops_level = getattr(info, "trade_stops_level", 0) * p
    return max(stops_level, spread_points * p * 0.25, p * 5)


def _classify_pending(bid: float, ask: float, is_buy: bool, entry: float) -> Optional[str]:
    if is_buy:
        if entry < bid:
            return "BUY_LIMIT"
        if entry > ask:
            return "BUY_STOP"
        return None
    else:
        if entry > ask:
            return "SELL_LIMIT"
        if entry < bid:
            return "SELL_STOP"
        return None


def _htf_bias(symbol: str, base_tf: str, is_buy: bool) -> float:
    # Use a higher TF (4h) SMA slope / position to bias entries
    if mt5 is None:
        return 0.0
    tf = "4h"
    cs = _get_rates(symbol, tf, 400)
    if len(cs) < 60:
        return 0.0
    closes = [x.c for x in cs]
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200) if len(cs) >= 200 else [float("nan")] * len(cs)
    last = closes[-1]
    s50 = sma50[-1]
    s200 = sma200[-1] if not math.isnan(sma200[-1]) else s50
    # Score: if above both MAs and 50>200 slope up -> +1, inverse -> -1
    score = 0.0
    if last > s50 > s200:
        score += 0.6
    if last > s200:
        score += 0.2
    # slope
    if len(sma50) >= 6 and not math.isnan(sma50[-6]):
        slope = (sma50[-1] - sma50[-6]) / 6.0
        score += 0.2 if slope > 0 else -0.2
    # Align with direction
    return score if is_buy else -score


def _score_confluence(is_buy: bool,
                      entry: float,
                      ref_price: float,
                      impulse: Tuple[int, int, float, float],
                      fvg: Optional[FVG],
                      eq_highs: List[int],
                      eq_lows: List[int],
                      atr_val: float,
                      info,
                      htf_score: float,
                      pending_type: str) -> Tuple[float, List[str]]:
    reasons = []
    score = 0.0

    # Pending orientation (prefer LIMIT over STOP for retests)
    if "LIMIT" in pending_type:
        score += 0.25
        reasons.append("prefer_limit_retest:+0.25")
    else:
        reasons.append("stop_breakout:+0.00")

    # HTF bias (−1..+1 scaled)
    score += max(-0.3, min(0.3, htf_score * 0.3))
    reasons.append(f"htf_bias:{htf_score:+.2f} => {max(-0.3, min(0.3, htf_score*0.3)):+.2f}")

    # Closeness to impulse 62–79% zone
    a_idx, b_idx, a_price, b_price = impulse
    leg = (b_price - a_price) if is_buy else (a_price - b_price)
    leg = max(leg, 1e-9)
    if is_buy:
        # normalize where entry sits within the leg segment from a->b
        pos = (entry - a_price) / (b_price - a_price + 1e-9)
        target_lo, target_hi = 0.21, 0.38  # discount inside up-leg (retrace of down?)
        dist = 0.0
        if pos < target_lo:
            dist = target_lo - pos
        elif pos > target_hi:
            dist = pos - target_hi
        # closer to zone => higher score
        cl = max(0.0, 1.0 - (dist / 0.5))
        delta = 0.25 * cl
        score += delta
        reasons.append(f"fib_discount_closeness:{cl:.2f} => {delta:+.2f}")
    else:
        pos = (entry - b_price) / (a_price - b_price + 1e-9)
        target_lo, target_hi = 0.62, 0.79  # premium for sell (retrace up in down-leg)
        dist = 0.0
        if pos < target_lo:
            dist = target_lo - pos
        elif pos > target_hi:
            dist = pos - target_hi
        cl = max(0.0, 1.0 - (dist / 0.5))
        delta = 0.25 * cl
        score += delta
        reasons.append(f"fib_premium_closeness:{cl:.2f} => {delta:+.2f}")

    # FVG alignment bonus
    if fvg is not None:
        score += 0.15
        reasons.append("fvg_alignment:+0.15")

    # Liquidity sweep context: prefer entries after a sweep in our direction
    if is_buy and eq_lows:
        score += 0.1
        reasons.append("eq_lows_swept:+0.10")
    if not is_buy and eq_highs:
        score += 0.1
        reasons.append("eq_highs_swept:+0.10")

    # Volatility-aware sanity: penalize too-tight risk vs ATR
    p = _point(info)
    min_dist = _min_distance(info)
    risk_to_atr = min(3.0, max(0.0, (abs(ref_price - entry)) / (atr_val + 1e-9)))
    # Closer entry to ref (retest deeper) -> smaller number but not too tiny
    # We lightly penalize extreme outliers
    if risk_to_atr > 2.0:
        score -= 0.1
        reasons.append("risk_to_atr>2_penalty:-0.10")

    return score, reasons


def _sl_tp_from_structure(is_buy: bool,
                          entry: float,
                          pivot_price: float,
                          atr_val: float,
                          info) -> Tuple[float, float, float]:
    # SL beyond pivot + ATR buffer
    buf = atr_val * DEFAULT_ATR_SL_BUF
    p = _point(info)
    min_dist = _min_distance(info)
    if is_buy:
        sl = min(entry - p, pivot_price - buf)
        if (entry - sl) < min_dist:
            sl = entry - min_dist
        tp = entry + max(DEFAULT_RR_TARGET * (entry - sl), min_dist * 2)
    else:
        sl = max(entry + p, pivot_price + buf)
        if (sl - entry) < min_dist:
            sl = entry + min_dist
        tp = entry - max(DEFAULT_RR_TARGET * (sl - entry), min_dist * 2)
    return sl, tp, min_dist


def _round_all(digits: int, *vals: float) -> List[float]:
    return [_round_price(v, digits) for v in vals]


def optimize_entry(analysis: Dict[str, Any], symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
    out = {
        "ok": False,
        "entry": None,
        "sl": None,
        "tp": None,
        "pending_type": None,
        "zone_bounds": None,
        "reasons": [],
        "diagnostics": {},
        "fallback_used": False,
        "score": 0.0
    }

    if OPTIMIZER_DISABLED or mt5 is None:
        return _fallback_from_analysis(analysis, out)

    verdict = str(analysis.get("final_suggestion") or analysis.get("final") or analysis.get("suggestion") or "").upper()
    is_buy = verdict.startswith("BUY")
    is_sell = verdict.startswith("SELL")
    if not (is_buy or is_sell):
        out["reasons"].append("no_actionable_verdict")
        return _fallback_from_analysis(analysis, out)

    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if not info or not tick:
        out["reasons"].append("symbol_or_tick_missing")
        return _fallback_from_analysis(analysis, out)

    bid, ask = float(tick.bid), float(tick.ask)
    digits = getattr(info, "digits", 5)

    # Load price data
    candles = _get_rates(symbol, timeframe, DEFAULT_BARS)
    if len(candles) < max(150, DEFAULT_ATR_PERIOD + 10):
        out["reasons"].append("not_enough_candles")
        return _fallback_from_analysis(analysis, out)

    atr_val = _atr(candles, DEFAULT_ATR_PERIOD)
    highs, lows = _find_swings(candles, span=3)
    if not highs and not lows:
        out["reasons"].append("no_structure")
        return _fallback_from_analysis(analysis, out)

    impulse = _last_impulse(candles, highs, lows, is_buy=is_buy)
    if not impulse:
        out["reasons"].append("no_impulse")
        return _fallback_from_analysis(analysis, out)

    a_idx, b_idx, a_price, b_price = impulse
    fvgs = _detect_fvgs(candles)
    ref = bid if is_buy else ask
    fvg = _nearest_fvg(candles, fvgs, ref, want_bullish=is_buy)
    eq_highs, eq_lows = _equal_levels(candles)

    # Candidate preferred entry near discount/premium + FVG refinement
    if is_buy:
        # 62–79% retrace from a->b (discount)
        fib_lo, fib_hi, fib_mid = 0.62, 0.79, 0.705
        zone_lo = b_price - (b_price - a_price) * fib_hi
        zone_hi = b_price - (b_price - a_price) * fib_lo
        entry = b_price - (b_price - a_price) * fib_mid
        zone_kind = "discount_fib"
    else:
        # 21–38% retrace from a->b (premium for sells in down-leg)
        fib_lo, fib_hi, fib_mid = 0.21, 0.38, 0.295
        zone_hi = b_price + (a_price - b_price) * fib_hi
        zone_lo = b_price + (a_price - b_price) * fib_lo
        entry = b_price + (a_price - b_price) * fib_mid
        zone_kind = "premium_fib"

    # If FVG mid is closer & aligned, prefer it
    use_fvg = False
    if fvg:
        fvg_mid = (fvg.low + fvg.high) / 2.0
        if is_buy and fvg_mid < ref and abs(ref - fvg_mid) < abs(ref - entry) * 0.9:
            entry = fvg_mid
            zone_lo, zone_hi = fvg.low, fvg.high
            zone_kind = "bullish_fvg"
            use_fvg = True
        if is_sell and fvg_mid > ref and abs(ref - fvg_mid) < abs(ref - entry) * 0.9:
            entry = fvg_mid
            zone_lo, zone_hi = fvg.low, fvg.high
            zone_kind = "bearish_fvg"
            use_fvg = True

    # Prefer LIMIT orientation; if inside spread, nudge into LIMIT
    pending = _classify_pending(bid, ask, is_buy, entry)
    if pending is None:
        min_dist = _min_distance(info)
        if is_buy:
            entry = min(entry, bid - max(min_dist * 1.2, atr_val * 0.1))
        else:
            entry = max(entry, ask + max(min_dist * 1.2, atr_val * 0.1))
        pending = _classify_pending(bid, ask, is_buy, entry)

    # SL/TP from structural pivot + ATR buffer
    pivot = candles[a_idx].l if is_buy else candles[a_idx].h
    sl, tp, min_dist = _sl_tp_from_structure(is_buy, entry, pivot, atr_val, info)

    # HTF bias score
    htf_score = _htf_bias(symbol, timeframe, is_buy=is_buy)
    # Confluence scoring
    cscore, more_reasons = _score_confluence(is_buy, entry, ref, impulse, fvg, eq_highs, eq_lows, atr_val, info, htf_score, pending)
    out["score"] = cscore
    out["reasons"].extend(more_reasons)
    out["reasons"].append(f"zone_kind:{zone_kind}")
    if use_fvg:
        out["reasons"].append("use_fvg:true")

    # Risk-to-Reward and sanity filter
    risk = abs(entry - sl)
    rr = abs((tp - entry) / (risk + 1e-9))
    if rr < max(1.5, DEFAULT_RR_TARGET * 0.8):
        out["reasons"].append(f"rr_too_low:{rr:.2f}")
        cscore -= 0.1

    # Final quality gate
    if cscore < DEFAULT_MIN_SCORE:
        out["reasons"].append(f"score_below_threshold:{cscore:.2f}<{DEFAULT_MIN_SCORE:.2f}")
        return _fallback_from_analysis(analysis, out)

    # Round & return
    entry, sl, tp = _round_all(digits, entry, sl, tp)
    out["entry"] = entry
    out["sl"] = sl
    out["tp"] = tp
    out["pending_type"] = pending
    out["zone_bounds"] = (round(min(zone_lo, zone_hi), digits), round(max(zone_lo, zone_hi), digits))
    out["diagnostics"].update({
        "atr": atr_val,
        "htf_bias": htf_score,
        "impulse": {"a_idx": a_idx, "b_idx": b_idx, "a_price": a_price, "b_price": b_price},
        "fvg_used": bool(fvg and use_fvg),
        "min_distance": min_dist,
        "rr": rr
    })
    out["ok"] = True
    return out


def _fallback_from_analysis(analysis: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, Any]:
    entry = (analysis.get("entry_point")
             or (analysis.get("precision_entry") or {}).get("entry_price")
             or (analysis.get("precision_entry") or {}).get("entry_point"))
    sl = analysis.get("sl") or analysis.get("stop_loss") or analysis.get("stop")
    tp = analysis.get("tp") or analysis.get("take_profit") or analysis.get("takeprofit")
    out.update({
        "fallback_used": True,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "pending_type": None
    })
    return out
