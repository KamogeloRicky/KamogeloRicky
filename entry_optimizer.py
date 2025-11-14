"""
Entry Optimizer for RickyFX — Robust Retest-Based Entries with Confluence Scoring

Purpose
- Refine raw analysis levels (entry, SL, TP) into realistic retest entries, avoiding mid-zone fills.
- Prefer LIMIT retests in discount/premium zones, optionally allow STOP (breakout) if configured.
- Enforce ATR-aware stop buffers and practical minimum distance heuristics.
- Use a confluence score (structure, impulse, FVGs, liquidity context, HTF bias) to gate quality.
- Fall back safely to original analysis values if data or confluence is insufficient.

Key Features
- Auto-initializes MetaTrader5 and makes the symbol visible (no credentials needed).
- Works even when symbol_info_tick is missing by falling back to symbol_info bid/ask.
- Safe scalarization: aggressively converts numpy arrays/lists/tuples/np.scalars to plain floats.
- No dependencies besides MetaTrader5 (numpy optional; not required).

Environment Flags (optional)
- ENTRY_OPTIMIZER_DISABLE=1        -> bypass optimizer (always fallback)
- ENTRY_OPTIMIZER_BARS=1000        -> number of candles to load
- ENTRY_OPTIMIZER_ATR_PERIOD=14    -> ATR lookback
- ENTRY_OPTIMIZER_RR_TARGET=1.8    -> minimum RR target used for TP
- ENTRY_OPTIMIZER_ATR_SL_BUF=0.5   -> SL buffer in ATR multiples beyond structure pivot
- ENTRY_OPTIMIZER_MIN_SCORE=0.6    -> confluence quality gate [0..1]
- ENTRY_OPTIMIZER_ALLOW_BREAKOUT=1 -> allow STOP entries (default prefers LIMIT)
- ENTRY_OPTIMIZER_SHOW=1           -> result includes reasons and diagnostics (default on)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import os
import math

try:
    import numpy as _np
except Exception:
    _np = None

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

# Defaults from env
_DEF_BARS = int(os.getenv("ENTRY_OPTIMIZER_BARS", "1000"))
_DEF_ATR_PERIOD = int(os.getenv("ENTRY_OPTIMIZER_ATR_PERIOD", "14"))
_DEF_RR_TARGET = float(os.getenv("ENTRY_OPTIMIZER_RR_TARGET", "1.8"))
_DEF_ATR_SL_BUF = float(os.getenv("ENTRY_OPTIMIZER_ATR_SL_BUF", "0.5"))
_DEF_MIN_SCORE = float(os.getenv("ENTRY_OPTIMIZER_MIN_SCORE", "0.6"))
_DEF_ALLOW_BREAKOUT = os.getenv("ENTRY_OPTIMIZER_ALLOW_BREAKOUT", "0") == "1"
_OPT_DISABLED = os.getenv("ENTRY_OPTIMIZER_DISABLE", "0") == "1"
_SHOW_OPT = os.getenv("ENTRY_OPTIMIZER_SHOW", "1") == "1"


def _scalar(x, fallback=None):
    try:
        if _np is not None and isinstance(x, _np.ndarray):
            if x.size == 0:
                return fallback
            return float(x.reshape(-1)[0])
        if isinstance(x, (list, tuple)):
            return _scalar(x[0], fallback) if x else fallback
        if hasattr(x, "item"):
            try:
                return float(x.item())
            except Exception:
                pass
        if x is None:
            return fallback
        return float(x)
    except Exception:
        return fallback


@dataclass
class Candle:
    t: int
    o: float
    h: float
    l: float
    c: float


@dataclass
class FVG:
    i0: int
    i2: int
    low: float
    high: float
    bullish: bool


def _tf_to_mt5(tf_str: str):
    if mt5 is None:
        return None
    s = (tf_str or "").lower()
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


def _ensure_attached_and_visible(symbol: str) -> bool:
    if mt5 is None:
        return False
    try:
        if mt5.terminal_info() is None:
            mt5.initialize()
        if mt5.terminal_info() is None:
            return False
        si = mt5.symbol_info(symbol)
        if si and not bool(getattr(si, "visible", True)):
            mt5.symbol_select(symbol, True)
        return mt5.symbol_info(symbol) is not None
    except Exception:
        return False


def _get_rates(symbol: str, timeframe: str, bars: int) -> List[Candle]:
    if mt5 is None:
        return []
    tf = _tf_to_mt5(timeframe)
    if tf is None:
        return []
    rows = mt5.copy_rates_from_pos(symbol, tf, 0, max(300, bars)) or []
    out: List[Candle] = []
    for r in rows:
        out.append(Candle(
            t=int(_scalar(r["time"], 0)),
            o=_scalar(r["open"], 0.0),
            h=_scalar(r["high"], 0.0),
            l=_scalar(r["low"], 0.0),
            c=_scalar(r["close"], 0.0),
        ))
    return out


def _fetch_quotes(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    if not _ensure_attached_and_visible(symbol):
        return None, None
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return _scalar(getattr(tick, "bid", None), None), _scalar(getattr(tick, "ask", None), None)
    except Exception:
        pass
    try:
        si = mt5.symbol_info(symbol)
        if si:
            return _scalar(getattr(si, "bid", None), None), _scalar(getattr(si, "ask", None), None)
    except Exception:
        pass
    return None, None


def _sma(vals: List[float], period: int) -> List[float]:
    out: List[float] = []
    s = 0.0
    for i, v in enumerate(vals):
        v = _scalar(v, 0.0)
        s += v
        if i >= period:
            s -= _scalar(vals[i - period], 0.0)
        out.append(s / period if i + 1 >= period else float("nan"))
    return out


def _atr(c: List[Candle], period: int) -> float:
    if len(c) < period + 1:
        return 0.0
    trs: List[float] = []
    for i in range(1, len(c)):
        h = _scalar(c[i].h, 0.0); l = _scalar(c[i].l, 0.0); pc = _scalar(c[i - 1].c, 0.0)
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return 0.0
    return sum(trs[-period:]) / period


def _find_swings(c: List[Candle], span: int = 3) -> Tuple[List[int], List[int]]:
    highs, lows = [], []
    n = len(c)
    for i in range(span, n - span):
        if all(_scalar(c[i].h, -1e18) >= _scalar(c[j].h, -1e18) for j in range(i - span, i + span + 1)):
            highs.append(i)
        if all(_scalar(c[i].l, +1e18) <= _scalar(c[j].l, +1e18) for j in range(i - span, i + span + 1)):
            lows.append(i)
    return highs, lows


def _last_impulse(c: List[Candle], highs: List[int], lows: List[int], is_buy: bool) -> Optional[Tuple[int, int, float, float]]:
    if not highs or not lows:
        return None
    if is_buy:
        for pivot in range(len(c) - 2, 3, -1):
            lows_before = [i for i in lows if i < pivot]
            if not lows_before:
                continue
            a = lows_before[-1]
            highs_after = [h for h in highs if h > a]
            if not highs_after:
                continue
            b = highs_after[-1]
            a_low = _scalar(c[a].l, 0.0); b_high = _scalar(c[b].h, 0.0)
            if b_high > a_low:
                return a, b, a_low, b_high
    else:
        for pivot in range(len(c) - 2, 3, -1):
            highs_before = [i for i in highs if i < pivot]
            if not highs_before:
                continue
            a = highs_before[-1]
            lows_after = [l for l in lows if l > a]
            if not lows_after:
                continue
            b = lows_after[-1]
            a_high = _scalar(c[a].h, 0.0); b_low = _scalar(c[b].l, 0.0)
            if a_high > b_low:
                return a, b, a_high, b_low
    return None


def _detect_fvgs(c: List[Candle]) -> List[FVG]:
    out: List[FVG] = []
    for i in range(2, len(c)):
        A, B, D = c[i - 2], c[i - 1], c[i]
        A_h, A_l = _scalar(A.h, 0.0), _scalar(A.l, 0.0)
        B_h, B_l = _scalar(B.h, 0.0), _scalar(B.l, 0.0)
        D_h, D_l = _scalar(D.h, 0.0), _scalar(D.l, 0.0)
        if A_h < D_l and B_l > A_h:
            out.append(FVG(i0=i - 2, i2=i, low=A_h, high=B_l, bullish=True))
        if A_l > D_h and B_h < A_l:
            out.append(FVG(i0=i - 2, i2=i, low=B_h, high=A_l, bullish=False))
    return out


def _nearest_fvg(fvgs: List[FVG], ref: float, want_bullish: bool) -> Optional[FVG]:
    ref = _scalar(ref, 0.0)
    best, best_d = None, None
    for f in fvgs:
        if bool(f.bullish) != bool(want_bullish):
            continue
        mid = (f.low + f.high) / 2.0
        if want_bullish and mid >= ref:
            continue
        if (not want_bullish) and mid <= ref:
            continue
        d = abs(ref - mid)
        if best_d is None or d < best_d:
            best, best_d = f, d
    return best


def _equal_levels(c: List[Candle], lookback: int = 30, tol: float = 0.0) -> Tuple[List[int], List[int]]:
    eq_highs, eq_lows = [], []
    n = len(c)
    L = max(5, min(lookback, n))
    for i in range(n - L, n - 2):
        if abs(_scalar(c[i].h, 0.0) - _scalar(c[i + 1].h, 0.0)) <= tol:
            eq_highs.append(i + 1)
        if abs(_scalar(c[i].l, 0.0) - _scalar(c[i + 1].l, 0.0)) <= tol:
            eq_lows.append(i + 1)
    return eq_highs[-3:], eq_lows[-3:]


def _point(info) -> float:
    return _scalar(getattr(info, "point", 0.00001), 0.00001) or 0.00001


def _round_price(x: float, digits: int) -> float:
    try:
        return float(f"{_scalar(x, 0.0):.{int(digits)}f}")
    except Exception:
        return _scalar(x, 0.0)


def _min_distance(info) -> float:
    p = _point(info)
    spread_points = _scalar(getattr(info, "spread", 0), 0.0)
    stops_level = _scalar(getattr(info, "trade_stops_level", 0), 0.0) * p
    return max(stops_level, spread_points * p * 0.25, p * 5)


def _classify_pending(bid: float, ask: float, is_buy: bool, entry: float) -> Optional[str]:
    bid = _scalar(bid, None); ask = _scalar(ask, None); entry = _scalar(entry, None)
    if None in (bid, ask, entry):
        return None
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


def _htf_bias(symbol: str, is_buy: bool) -> float:
    if mt5 is None:
        return 0.0
    cs = _get_rates(symbol, "4h", 400)
    if len(cs) < 120:
        return 0.0
    closes = [_scalar(x.c, 0.0) for x in cs]
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)
    last = _scalar(closes[-1], 0.0)
    s50 = _scalar(sma50[-1], last)
    s200 = _scalar(sma200[-1], s50 if not math.isnan(s50) else last)
    score = 0.0
    if last > s50 > s200:
        score += 0.6
    if last > s200:
        score += 0.2
    if len(sma50) >= 6 and not math.isnan(_scalar(sma50[-6], s50)):
        slope = _scalar(sma50[-1], s50) - _scalar(sma50[-6], s50)
        score += 0.2 if slope > 0 else -0.2
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
                      pending_type: Optional[str]) -> Tuple[float, List[str]]:
    reasons = []
    score = 0.0

    if pending_type and "LIMIT" in pending_type:
        score += 0.25
        reasons.append("prefer_limit_retest:+0.25")
    else:
        reasons.append("stop_or_inside_spread:+0.00")

    bias_delta = max(-0.3, min(0.3, _scalar(htf_score, 0.0) * 0.3))
    score += bias_delta
    reasons.append(f"htf_bias:{_scalar(htf_score,0.0):+.2f} => {bias_delta:+.2f}")

    a_idx, b_idx, a_price, b_price = impulse
    a_price = _scalar(a_price, 0.0); b_price = _scalar(b_price, 0.0)
    entry = _scalar(entry, 0.0); ref_price = _scalar(ref_price, 0.0)

    leg_len = abs(b_price - a_price) + 1e-9
    if is_buy:
        pos = (entry - a_price) / leg_len
        target_lo, target_hi = 0.21, 0.38
    else:
        pos = (entry - b_price) / leg_len
        target_lo, target_hi = 0.62, 0.79

    if pos < target_lo:
        dist = target_lo - pos
    elif pos > target_hi:
        dist = pos - target_hi
    else:
        dist = 0.0

    closeness = max(0.0, 1.0 - (dist / 0.5))
    delta = 0.25 * closeness
    score += delta
    reasons.append(f"impulse_retrace_closeness:{closeness:.2f} => {delta:+.2f}")

    if fvg is not None:
        score += 0.15
        reasons.append("fvg_alignment:+0.15")

    if is_buy and eq_lows:
        score += 0.10
        reasons.append("eq_lows_swept:+0.10")
    if (not is_buy) and eq_highs:
        score += 0.10
        reasons.append("eq_highs_swept:+0.10")

    atr_val = _scalar(atr_val, 0.0)
    risk_to_atr = min(3.0, max(0.0, abs(ref_price - entry) / (atr_val + 1e-9)))
    if risk_to_atr > 2.0:
        score -= 0.10
        reasons.append("risk_to_atr>2_penalty:-0.10")

    return score, reasons


def _sl_tp_from_structure(is_buy: bool,
                          entry: float,
                          pivot: float,
                          atr_val: float,
                          info) -> Tuple[float, float, float]:
    entry = _scalar(entry, 0.0); pivot = _scalar(pivot, entry); atr_val = _scalar(atr_val, 0.0)
    buf = atr_val * _DEF_ATR_SL_BUF
    min_dist = _min_distance(info)

    if is_buy:
        sl = min(entry - _point(info), pivot - buf)
        if (entry - sl) < min_dist:
            sl = entry - min_dist
        tp = entry + max(_DEF_RR_TARGET * (entry - sl), min_dist * 2)
    else:
        sl = max(entry + _point(info), pivot + buf)
        if (sl - entry) < min_dist:
            sl = entry + min_dist
        tp = entry - max(_DEF_RR_TARGET * (sl - entry), min_dist * 2)

    return sl, tp, min_dist


def optimize_entry(analysis: Dict[str, Any],
                   symbol: str,
                   timeframe: str = "1h",
                   overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

    try:
        if _OPT_DISABLED or mt5 is None:
            out["reasons"].append("optimizer_disabled_or_mt5_missing")
            return _fallback_from_analysis(analysis, out)

        # Config overrides
        bars = int((overrides or {}).get("bars", _DEF_BARS))
        atr_period = int((overrides or {}).get("atr_period", _DEF_ATR_PERIOD))
        rr_target = float((overrides or {}).get("rr_target", _DEF_RR_TARGET))
        atr_sl_buffer = float((overrides or {}).get("atr_sl_buffer", _DEF_ATR_SL_BUF))
        min_score = float((overrides or {}).get("min_score", _DEF_MIN_SCORE))
        allow_breakout = bool((overrides or {}).get("allow_breakout", _DEF_ALLOW_BREAKOUT))

        verdict = str(analysis.get("final_suggestion") or analysis.get("final") or analysis.get("suggestion") or "").upper()
        is_buy = verdict.startswith("BUY")
        is_sell = verdict.startswith("SELL")
        if not (is_buy or is_sell):
            out["reasons"].append("no_actionable_verdict")
            return _fallback_from_analysis(analysis, out)

        if not _ensure_attached_and_visible(symbol):
            out["reasons"].append("mt5_attach_or_symbol_visible_failed")
            return _fallback_from_analysis(analysis, out)

        info = mt5.symbol_info(symbol)
        if not info:
            out["reasons"].append("symbol_info_missing")
            return _fallback_from_analysis(analysis, out)

        bid, ask = _fetch_quotes(symbol)
        if bid is None or ask is None:
            out["reasons"].append("quotes_missing")
            return _fallback_from_analysis(analysis, out)

        digits = int(_scalar(getattr(info, "digits", 5), 5))

        candles = _get_rates(symbol, timeframe, bars)
        if len(candles) < max(150, atr_period + 10):
            out["reasons"].append("not_enough_candles")
            return _fallback_from_analysis(analysis, out)

        atr_val = _atr(candles, atr_period)
        highs, lows = _find_swings(candles, span=3)
        if not highs and not lows:
            out["reasons"].append("no_structure_found")
            return _fallback_from_analysis(analysis, out)

        impulse = _last_impulse(candles, highs, lows, is_buy=is_buy)
        if not impulse:
            out["reasons"].append("no_impulse_detected")
            return _fallback_from_analysis(analysis, out)

        a_idx, b_idx, a_price, b_price = impulse
        fvgs = _detect_fvgs(candles)
        ref = bid if is_buy else ask
        fvg = _nearest_fvg(fvgs, ref, want_bullish=is_buy)
        eq_highs, eq_lows = _equal_levels(candles)

        # Candidate entry via fib retrace + FVG refinement
        if is_buy:
            zone_lo = b_price - (b_price - a_price) * 0.79
            zone_hi = b_price - (b_price - a_price) * 0.62
            entry = b_price - (b_price - a_price) * 0.705
            zone_kind = "discount_fib"
        else:
            zone_hi = b_price + (a_price - b_price) * 0.38
            zone_lo = b_price + (a_price - b_price) * 0.21
            entry = b_price + (a_price - b_price) * 0.295
            zone_kind = "premium_fib"

        use_fvg = False
        if fvg:
            fvg_mid = (fvg.low + fvg.high) / 2.0
            if is_buy and fvg_mid < ref and abs(ref - fvg_mid) <= abs(ref - entry) * 0.9:
                entry = fvg_mid; zone_lo, zone_hi = fvg.low, fvg.high; zone_kind = "bullish_fvg"; use_fvg = True
            if is_sell and fvg_mid > ref and abs(ref - fvg_mid) <= abs(ref - entry) * 0.9:
                entry = fvg_mid; zone_lo, zone_hi = fvg.low, fvg.high; zone_kind = "bearish_fvg"; use_fvg = True

        pending = _classify_pending(bid, ask, is_buy, entry)
        if pending is None:
            md = _min_distance(info)
            if is_buy:
                entry = min(entry, bid - max(md * 1.2, atr_val * 0.1))
            else:
                entry = max(entry, ask + max(md * 1.2, atr_val * 0.1))
            pending = _classify_pending(bid, ask, is_buy, entry)

        if (not allow_breakout) and pending and "STOP" in pending:
            md = _min_distance(info)
            if is_buy:
                entry = bid - max(md * 1.6, atr_val * 0.12)
            else:
                entry = ask + max(md * 1.6, atr_val * 0.12)
            pending = _classify_pending(bid, ask, is_buy, entry)

        pivot = candles[a_idx].l if is_buy else candles[a_idx].h

        # Temporarily override globals for SL/TP calc (local effect)
        global _DEF_RR_TARGET, _DEF_ATR_SL_BUF
        prev_rr, prev_buf = _DEF_RR_TARGET, _DEF_ATR_SL_BUF
        _DEF_RR_TARGET, _DEF_ATR_SL_BUF = rr_target, atr_sl_buffer
        sl, tp, min_dist = _sl_tp_from_structure(is_buy, entry, pivot, atr_val, info)
        _DEF_RR_TARGET, _DEF_ATR_SL_BUF = prev_rr, prev_buf

        htf_score = _htf_bias(symbol, is_buy=is_buy)
        cscore, more_reasons = _score_confluence(is_buy, entry, ref, (a_idx, b_idx, a_price, b_price), fvg, eq_highs, eq_lows, atr_val, info, htf_score, pending)
        out["score"] = cscore
        out["reasons"].extend(more_reasons)
        out["reasons"].append(f"zone_kind:{zone_kind}")
        if use_fvg:
            out["reasons"].append("use_fvg:true")

        risk = max(1e-9, abs(entry - sl))
        rr = abs((tp - entry) / risk)
        if rr < max(1.5, rr_target * 0.8):
            out["reasons"].append(f"rr_too_low:{rr:.2f}")
            cscore -= 0.10

        if cscore < min_score:
            out["reasons"].append(f"score_below_threshold:{cscore:.2f}<{min_score:.2f}")
            return _fallback_from_analysis(analysis, out)

        digits = int(_scalar(getattr(info, "digits", 5), 5))
        entry, sl, tp = _round_price(entry, digits), _round_price(sl, digits), _round_price(tp, digits)

        out["entry"] = entry
        out["sl"] = sl
        out["tp"] = tp
        out["pending_type"] = pending
        out["zone_bounds"] = (_round_price(min(zone_lo, zone_hi), digits), _round_price(max(zone_lo, zone_hi), digits))
        out["diagnostics"].update({
            "atr": atr_val,
            "htf_bias": htf_score,
            "impulse": {"a_idx": a_idx, "b_idx": b_idx, "a_price": a_price, "b_price": b_price},
            "fvg_used": bool(fvg and use_fvg),
            "min_distance": min_dist,
            "rr": rr,
            "config": {
                "bars": bars, "atr_period": atr_period, "rr_target": rr_target,
                "atr_sl_buffer": atr_sl_buffer, "min_score": min_score, "allow_breakout": allow_breakout
            }
        })
        out["ok"] = True
        return out

    except Exception as e:
        out["reasons"].append(f"optimizer_exception:{e}")
        return _fallback_from_analysis(analysis, out)


def _fallback_from_analysis(analysis: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, Any]:
    entry = (analysis.get("entry_point")
             or (analysis.get("precision_entry") or {}).get("entry_price")
             or (analysis.get("precision_entry") or {}).get("entry_point"))
    sl = analysis.get("sl") or analysis.get("stop_loss") or analysis.get("stop")
    tp = analysis.get("tp") or analysis.get("take_profit") or analysis.get("takeprofit")
    out.update({
        "fallback_used": True,
        "entry": _scalar(entry, None),
        "sl": _scalar(sl, None),
        "tp": _scalar(tp, None),
        "pending_type": None
    })
    return out
