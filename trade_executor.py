"""
RickyFX Pending Order Executor (with Stop Orientation & Distance Validation + Autonomous Position Tracking)

NEW v5.0 GOAT EDITION:
- Automatically updates open_positions.json after every trade action via position_manager.py
- Full integration with Correlation Guard and Regime Sentinel
- No manual position tracking required

Improvements:
- Validates SL/TP orientation before send.
- Optional automatic correction & retry if AUTO_FIX_STOPS=1.
- Enforces a dynamic minimum distance (min_distance) for indices / instruments that silently require buffer.
- Retries once on retcode 10016 (Invalid stops) by expanding stop distances.
- NOW: Automatic position sync with MT5 after successful trade placement

Environment flags:
  TRADE_DEBUG=1      -> verbose debug_log
  AUTO_FIX_STOPS=1   -> auto-fix invalid orientation & distances, retry on 10016

Preserves:
- Lot size exact usage (unless previous volume normalization logic adjusts).
- Pending order classification (limit/stop).
- Variant symbol resolution, volume validation, root cause diagnostics.

Adds keys to result:
  original_sl, original_tp
  final_sl, final_tp
  stops_adjusted (bool)
  stop_fix_reason (list[str])
  retry_on_invalid_stops (bool)
  second_retcode (int|None)
  second_comment (str|None)
  position_tracking_status (str) - NEW: "synced" | "failed" | "skipped"
"""

import MetaTrader5 as mt5
import time
import re
import os
from math import isclose

# NEW: Import Position Manager for autonomous tracking
try:
    from position_manager import update_positions
    POSITION_MANAGER_AVAILABLE = True
except ImportError:
    POSITION_MANAGER_AVAILABLE = False
    print("⚠️ Trade Executor: position_manager.py not found. Position tracking will be disabled.")
    def update_positions():
        return []

_SUFFIXES = {"c", "m", "r", "x", "p", "i"}
_DEBUG = os.getenv("TRADE_DEBUG", "0") == "1"
AUTO_FIX_STOPS = os.getenv("AUTO_FIX_STOPS", "0") == "1"

RET_EXPLANATIONS = {
    getattr(mt5, "TRADE_RETCODE_DONE", 10009): "Order placed successfully",
    getattr(mt5, "TRADE_RETCODE_PLACED", 10008): "Order placed (pending)",
    getattr(mt5, "TRADE_RETCODE_REQUOTE", 10004): "Requote",
    getattr(mt5, "TRADE_RETCODE_PRICE_OFF", 10025): "Price off",
    getattr(mt5, "TRADE_RETCODE_INVALID_PRICE", 10027): "Invalid price",
    getattr(mt5, "TRADE_RETCODE_INVALID_STOPS", 10028): "Invalid SL/TP distance",
    getattr(mt5, "TRADE_RETCODE_OFFQUOTES", 10021): "Offquotes",
    getattr(mt5, "TRADE_RETCODE_REJECT", 10030): "Rejected",
    getattr(mt5, "TRADE_RETCODE_NOT_ENOUGH_MONEY", 10031): "Not enough margin",
    getattr(mt5, "TRADE_RETCODE_TOO_MANY_REQUESTS", 10033): "Too many requests",
    getattr(mt5, "TRADE_RETCODE_MARKET_CLOSED", 10024): "Market closed",
    getattr(mt5, "TRADE_RETCODE_TRADE_DISABLED", 10019): "Trading disabled",
    10016: "Invalid SL/TP distance (orientation or min distance)"  # explicit mapping
}

def _trace(msg, log):
    if _DEBUG:
        print(f"[TRADE_DEBUG] {msg}")
    log.append(msg)

def _round_price(value, digits):
    try:
        return float(f"{float(value):.{digits}f}")
    except Exception:
        return value

def _normalize_volume(volume, vol_min, vol_step, vol_max):
    try:
        v = float(volume)
    except Exception:
        return vol_min
    if v < vol_min:
        v = vol_min
    steps = int((v - vol_min + 1e-12) / max(vol_step, 1e-12))
    v = vol_min + steps * vol_step
    if vol_max and v > vol_max:
        v = vol_max
    return float(f"{v:.10g}")

def _preflight(symbol):
    out = {}
    try:
        ti = mt5.terminal_info()
        out["terminal_info"] = ti
        if ti:
            out["autotrading_allowed"] = getattr(ti, "trade_allowed", None)
            out["tradeapi_enabled"] = getattr(ti, "tradeapi_enabled", None)
    except Exception:
        out["terminal_info"] = None
    try:
        ai = mt5.account_info()
        out["account_info"] = ai
        if ai:
            out["account_trade_allowed"] = getattr(ai, "trade_allowed", None)
            out["margin_free"] = getattr(ai, "margin_free", None)
    except Exception:
        out["account_info"] = None
    try:
        si = mt5.symbol_info(symbol)
        out["raw_symbol_info"] = si._asdict() if si else None
        if si:
            out["symbol_trade_mode"] = getattr(si, "trade_mode", None)
            out["symbol_visible"] = getattr(si, "visible", None)
    except Exception:
        out["raw_symbol_info"] = None
    try:
        out["last_error"] = mt5.last_error()
    except Exception:
        out["last_error"] = None
    return out

def _ensure_connected(debug_log, root_cause, preflight):
    ti = preflight.get("terminal_info")
    if ti:
        _trace("MT5 already attached.", debug_log)
        return True
    _trace("Attempting mt5.initialize()...", debug_log)
    try:
        ok = mt5.initialize()
    except Exception as e:
        _trace(f"initialize exception: {e}", debug_log)
        root_cause.append(f"mt5_initialize_exception:{e}")
        return False
    if not ok:
        le = None
        try: le = mt5.last_error()
        except Exception: pass
        _trace(f"initialize failed last_error={le}", debug_log)
        root_cause.append("mt5_initialize_failed")
        return False
    _trace("mt5.initialize() succeeded.", debug_log)
    return True

def _generate_variant_candidates(symbol):
    s = symbol.strip()
    candidates = [s, s.upper()]
    if len(s) > 6 and s[-1].isalpha() and s[-1].lower() in _SUFFIXES:
        base = s[:-1]
        suffix = s[-1].lower()
        candidates += [base, base.upper(), f"{base}.{suffix}", f"{base.upper()}.{suffix}"]
        for alt in ("c", "m", "r", "x", "p"):
            cand = f"{base}{alt}"
            if cand not in candidates:
                candidates.append(cand)
    seen=set(); ordered=[]
    for c in candidates:
        if c not in seen:
            seen.add(c); ordered.append(c)
    return ordered

def _scan_all_symbol_variants(base_stem):
    try:
        all_syms = mt5.symbols_get() or []
    except Exception:
        all_syms = []
    base_up = base_stem.upper()
    names = []
    for s in all_syms:
        name = getattr(s, "name", "")
        if base_up in name.upper():
            names.append(name)
    return names

def _resolve_symbol_with_fallback(symbol, debug_log, root_cause):
    alt_suggestions = []
    tried_alts = False

    info = mt5.symbol_info(symbol)
    if info:
        if not info.visible:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        tm = getattr(info, "trade_mode", None)
        _trace(f"Primary symbol {symbol} trade_mode={tm}", debug_log)
        if tm not in (0, 3):
            return symbol, info, tried_alts, alt_suggestions
        root_cause.append("symbol_trade_mode_disabled_primary")
        tried_alts = True
    else:
        tried_alts = True
        _trace("Primary symbol not found; trying variants.", debug_log)

    base = symbol
    if len(symbol) > 6 and symbol[-1].isalpha():
        base = symbol[:-1]
    variants = _generate_variant_candidates(symbol)
    scanned = _scan_all_symbol_variants(base)
    for n in scanned:
        if n not in variants:
            variants.append(n)

    for cand in variants:
        ci = mt5.symbol_info(cand)
        if not ci:
            continue
        tm = getattr(ci, "trade_mode", None)
        alt_suggestions.append({
            "name": cand,
            "trade_mode": tm,
            "volume_min": getattr(ci, "volume_min", None),
            "volume_step": getattr(ci, "volume_step", None),
            "visible": getattr(ci, "visible", None),
        })

    alt_suggestions.sort(key=lambda x: (0 if (x["trade_mode"] not in (0,3) and x["trade_mode"] is not None) else 1, x["name"]))

    for sugg in alt_suggestions:
        if sugg["trade_mode"] not in (0,3) and sugg["trade_mode"] is not None:
            cand = sugg["name"]
            ci = mt5.symbol_info(cand)
            if ci and not ci.visible:
                mt5.symbol_select(cand, True)
                ci = mt5.symbol_info(cand)
            _trace(f"Using alternative tradable symbol: {cand} (trade_mode={sugg['trade_mode']})", debug_log)
            root_cause.append("alt_symbol_resolved")
            return cand, ci, tried_alts, alt_suggestions

    if tried_alts:
        root_cause.append("all_variants_disabled")

    return None, None, tried_alts, alt_suggestions

def _classify_root_causes(preflight, symbol_info, tick, entry, bid, ask, retcode, comment, error_tag, collected):
    causes = list(collected) if collected else []
    ti = preflight.get("terminal_info"); ai = preflight.get("account_info")

    if ti is None:
        causes.append("terminal_not_attached")
    else:
        if getattr(ti, "tradeapi_enabled", True) is False:
            causes.append("tradeapi_disabled_in_terminal")
        if getattr(ti, "trade_allowed", True) is False:
            causes.append("terminal_autotrading_disabled")

    if ai is None:
        causes.append("account_info_missing")
    else:
        if getattr(ai, "trade_allowed", True) is False:
            causes.append("account_trading_disabled")

    if symbol_info is None:
        causes.append("symbol_info_missing")
    else:
        tm = getattr(symbol_info, "trade_mode", None)
        if tm == 0:
            causes.append("symbol_trade_mode_disabled")
        elif tm == 3:
            causes.append("symbol_trade_mode_closeonly")

    if tick is None:
        causes.append("tick_missing")
    else:
        if (getattr(tick, "bid", 0) == 0) and (getattr(tick, "ask", 0) == 0):
            causes.append("bid_ask_zero")
        if bid and ask and entry is not None and (bid <= entry <= ask):
            causes.append("entry_inside_spread")

    if isinstance(retcode, int) and retcode not in (getattr(mt5, "TRADE_RETCODE_DONE", 10009), getattr(mt5, "TRADE_RETCODE_PLACED", 10008)):
        causes.append(f"retcode_issue:{retcode}")

    if comment:
        lc = str(comment).lower()
        if "not enough money" in lc:
            causes.append("margin_insufficient")
        if "disabled" in lc:
            causes.append("trading_disabled_comment")

    if error_tag and str(error_tag).startswith("retcode "):
        causes.append("generic_retcode_failure")

    if not causes:
        causes.append("ok")

    seen=set(); ordered=[]
    for c in causes:
        if c not in seen:
            seen.add(c); ordered.append(c)
    return ordered

def _compute_min_distance(info):
    point = getattr(info, "point", 0.00001)
    spread = getattr(info, "spread", 0) * point
    stops_level = getattr(info, "trade_stops_level", 0) * point
    return max(stops_level, spread * 0.2, point * 5)

def _validate_and_fix_stops(is_buy, entry, sl, tp, info, log):
    """
    Returns (final_sl, final_tp, stops_adjusted(bool), reasons[list], min_distance_used)
    """
    reasons = []
    adjusted = False
    min_dist = _compute_min_distance(info)
    point = getattr(info, "point", 0.00001)

    if is_buy:
        if sl >= entry:
            reasons.append("sl_above_entry_for_buy")
        if tp <= entry:
            reasons.append("tp_not_above_entry_for_buy")
    else:
        if sl <= entry:
            reasons.append("sl_below_entry_for_sell")
        if tp >= entry:
            reasons.append("tp_not_below_entry_for_sell")

    final_sl = sl
    final_tp = tp

    if reasons and not AUTO_FIX_STOPS:
        return final_sl, final_tp, adjusted, reasons, min_dist

    if reasons and AUTO_FIX_STOPS:
        adjusted = True
        dist_sl = abs(sl - entry)
        dist_tp = abs(tp - entry)
        if dist_sl < point * 0.5:
            dist_sl = min_dist
        if dist_tp < point * 0.5:
            dist_tp = min_dist

        if is_buy:
            final_sl = entry - max(dist_sl, min_dist)
            final_tp = entry + max(dist_tp, min_dist)
        else:
            final_sl = entry + max(dist_sl, min_dist)
            final_tp = entry - max(dist_tp, min_dist)
        reasons.append("auto_orientation_corrected")

    if AUTO_FIX_STOPS:
        if is_buy:
            if (entry - final_sl) < min_dist:
                final_sl = entry - min_dist
                adjusted = True
                reasons.append("sl_min_distance_adjusted")
            if (final_tp - entry) < min_dist:
                final_tp = entry + min_dist
                adjusted = True
                reasons.append("tp_min_distance_adjusted")
        else:
            if (final_sl - entry) < min_dist:
                final_sl = entry + min_dist
                adjusted = True
                reasons.append("sl_min_distance_adjusted")
            if (entry - final_tp) < min_dist:
                final_tp = entry - min_dist
                adjusted = True
                reasons.append("tp_min_distance_adjusted")

    digits = getattr(info, "digits", 5)
    final_sl = _round_price(final_sl, digits)
    final_tp = _round_price(final_tp, digits)

    return final_sl, final_tp, adjusted, reasons, min_dist

def place_pending_order(symbol, is_buy, entry_price, sl, tp, lot, magic=777, expiry_minutes=180):
    debug_log = []
    out = {
        "ok": False,
        "retcode": None,
        "pending_type": None,
        "comment": None,
        "error": None,
        "request": {},
        "resolved_symbol": None,
        "root_cause": [],
        "debug_log": debug_log,
        "preflight": {},
        "symbol_constraints": {},
        "volume_normalized": None,
        "alt_suggestions": [],
        "original_sl": sl,
        "original_tp": tp,
        "final_sl": None,
        "final_tp": None,
        "stops_adjusted": False,
        "stop_fix_reason": [],
        "retry_on_invalid_stops": False,
        "second_retcode": None,
        "second_comment": None,
        "position_tracking_status": "skipped"  # NEW: tracking status
    }

    _trace(f"Start place_pending_order(symbol={symbol}, is_buy={is_buy}, entry={entry_price}, sl={sl}, tp={tp}, lot={lot})", debug_log)

    pre = _preflight(symbol); out["preflight"] = pre
    rc_collect = []
    if not _ensure_connected(debug_log, rc_collect, pre):
        out["preflight"] = _preflight(symbol)
        out["error"] = "mt5_not_attached"
        out["root_cause"] = _classify_root_causes(out["preflight"], None, None, entry_price, None, None, None, None, out["error"], rc_collect)
        return out
    out["preflight"] = _preflight(symbol)

    resolved_symbol, info, tried_alts, suggestions = _resolve_symbol_with_fallback(symbol, debug_log, rc_collect)
    out["resolved_symbol"] = resolved_symbol
    out["alt_suggestions"] = suggestions

    if info is None:
        out["error"] = "symbol_not_found_or_disabled"
        out["comment"] = f"No enabled tradable variant for '{symbol}'."
        out["root_cause"] = _classify_root_causes(out["preflight"], None, None, entry_price, None, None, None, None, out["error"], rc_collect)
        return out

    vol_min = getattr(info, "volume_min", 0.01)
    vol_step = getattr(info, "volume_step", 0.01)
    vol_max = getattr(info, "volume_max", None)
    out["symbol_constraints"] = {"volume_min": vol_min, "volume_step": vol_step, "volume_max": vol_max}
    normalized = _normalize_volume(lot, vol_min, vol_step, vol_max)
    out["volume_normalized"] = normalized

    if lot < vol_min - 1e-12:
        out["error"] = "volume_invalid"
        out["comment"] = f"Lot {lot} < volume_min {vol_min}."
        rc_collect.append("volume_min_violation")
        out["root_cause"] = _classify_root_causes(out["preflight"], info, None, entry_price, None, None, None, out["comment"], out["error"], rc_collect)
        return out
    if vol_step > 0:
        steps_float = (lot - vol_min) / vol_step
        if not isclose(steps_float, round(steps_float), rel_tol=1e-9, abs_tol=1e-9):
            out["error"] = "volume_invalid"
            out["comment"] = f"Lot {lot} not aligned to step {vol_step} from min {vol_min}."
            rc_collect.append("volume_step_misaligned")
            out["root_cause"] = _classify_root_causes(out["preflight"], info, None, entry_price, None, None, None, out["comment"], out["error"], rc_collect)
            return out

    tm = getattr(info, "trade_mode", None)
    if tm in (0,3):
        out["error"] = "symbol_trade_mode_disabled"
        out["comment"] = f"Symbol '{resolved_symbol}' trade_mode={tm} blocks new orders."
        out["root_cause"] = _classify_root_causes(out["preflight"], info, None, entry_price, None, None, None, out["comment"], out["error"], rc_collect)
        return out

    tick = mt5.symbol_info_tick(resolved_symbol)
    if tick is None:
        out["error"] = "tick_unavailable"
        out["comment"] = f"No tick for '{resolved_symbol}'."
        out["root_cause"] = _classify_root_causes(out["preflight"], info, None, entry_price, None, None, None, out["comment"], out["error"], rc_collect)
        return out

    bid = tick.bid; ask = tick.ask
    digits = info.digits

    entry_r = _round_price(entry_price, digits)
    sl_r = _round_price(sl, digits)
    tp_r = _round_price(tp, digits)

    _trace(f"Rounded -> entry={entry_r}, sl={sl_r}, tp={tp_r}; bid={bid}, ask={ask}; digits={digits}", debug_log)

    if is_buy:
        if entry_r < bid:
            order_type = mt5.ORDER_TYPE_BUY_LIMIT; pending_label="BUY_LIMIT"
        elif entry_r > ask:
            order_type = mt5.ORDER_TYPE_BUY_STOP; pending_label="BUY_STOP"
        else:
            out["error"] = "entry_inside_spread_for_buy"
            out["comment"] = f"Entry {entry_r} inside spread [{bid}, {ask}]."
            out["root_cause"] = _classify_root_causes(out["preflight"], info, tick, entry_r, bid, ask, None, out["comment"], out["error"], rc_collect)
            return out
    else:
        if entry_r > ask:
            order_type = mt5.ORDER_TYPE_SELL_LIMIT; pending_label="SELL_LIMIT"
        elif entry_r < bid:
            order_type = mt5.ORDER_TYPE_SELL_STOP; pending_label="SELL_STOP"
        else:
            out["error"] = "entry_inside_spread_for_sell"
            out["comment"] = f"Entry {entry_r} inside spread [{bid}, {ask}]."
            out["root_cause"] = _classify_root_causes(out["preflight"], info, tick, entry_r, bid, ask, None, out["comment"], out["error"], rc_collect)
            return out

    out["pending_type"] = pending_label

    final_sl, final_tp, adjusted, reasons, min_dist = _validate_and_fix_stops(is_buy, entry_r, sl_r, tp_r, info, debug_log)
    if reasons and not AUTO_FIX_STOPS and adjusted is False:
        out["error"] = "invalid_stop_orientation"
        out["comment"] = f"Stop orientation invalid: {reasons}"
        out["stop_fix_reason"] = reasons
        out["final_sl"] = sl_r
        out["final_tp"] = tp_r
        out["stops_adjusted"] = False
        out["root_cause"] = _classify_root_causes(out["preflight"], info, tick, entry_r, bid, ask, None, out["comment"], out["error"], rc_collect) + ["stop_orientation_invalid"]
        return out

    out["final_sl"] = final_sl
    out["final_tp"] = final_tp
    out["stops_adjusted"] = adjusted
    out["stop_fix_reason"] = reasons

    expiration = int(time.time() + expiry_minutes * 60)
    req = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": resolved_symbol,
        "volume": lot,
        "type": order_type,
        "price": entry_r,
        "sl": final_sl,
        "tp": final_tp,
        "deviation": 20,
        "magic": magic,
        "comment": "RickyFX",
        "type_time": mt5.ORDER_TIME_SPECIFIED,
        "expiration": expiration,
        "type_filling": mt5.ORDER_FILLING_RETURN
    }
    out["request"] = req
    _trace(f"Sending request: {req}", debug_log)

    try:
        result = mt5.order_send(req)
    except Exception as e:
        out["error"] = f"order_send_exception:{e}"
        out["comment"] = str(e)
        out["root_cause"] = _classify_root_causes(out["preflight"], info, tick, entry_r, bid, ask, None, out["comment"], out["error"], rc_collect)
        _trace(f"order_send exception: {e}", debug_log)
        return out

    out["retcode"] = rc = result.retcode
    out["comment"] = getattr(result, "comment", None)
    _trace(f"MT5 retcode={rc}, comment={out['comment']}", debug_log)

    if rc == 10016 and AUTO_FIX_STOPS:
        out["retry_on_invalid_stops"] = True
        _trace("Retcode 10016: attempting widened distance retry.", debug_log)
        widen_factor = 1.5
        if is_buy:
            dist_sl = entry_r - final_sl
            dist_tp = final_tp - entry_r
            final_sl = _round_price(entry_r - dist_sl * widen_factor, digits)
            final_tp = _round_price(entry_r + dist_tp * widen_factor, digits)
        else:
            dist_sl = final_sl - entry_r
            dist_tp = entry_r - final_tp
            final_sl = _round_price(entry_r + dist_sl * widen_factor, digits)
            final_tp = _round_price(entry_r - dist_tp * widen_factor, digits)

        req_retry = dict(req)
        req_retry["sl"] = final_sl
        req_retry["tp"] = final_tp
        out["final_sl"] = final_sl
        out["final_tp"] = final_tp
        _trace(f"Retry request with widened stops: {req_retry}", debug_log)
        try:
            result2 = mt5.order_send(req_retry)
            out["second_retcode"] = result2.retcode
            out["second_comment"] = getattr(result2, "comment", None)
            _trace(f"Second retcode={result2.retcode}, comment={out['second_comment']}", debug_log)
            if result2.retcode == getattr(mt5, "TRADE_RETCODE_DONE", 10009):
                out["ok"] = True
                out["retcode"] = result2.retcode
                out["comment"] = out["second_comment"]
                out["error"] = None
            else:
                out["retcode"] = result2.retcode
                out["error"] = RET_EXPLANATIONS.get(result2.retcode) or f"retcode {result2.retcode}"
        except Exception as e:
            out["second_retcode"] = None
            out["second_comment"] = f"retry_exception:{e}"
            _trace(f"Retry send exception: {e}", debug_log)

    else:
        if rc == getattr(mt5, "TRADE_RETCODE_DONE", 10009):
            out["ok"] = True
            out["error"] = None
        else:
            out["error"] = RET_EXPLANATIONS.get(rc) or f"retcode {rc}"

    out["root_cause"] = _classify_root_causes(out["preflight"], info, tick, entry_r, bid, ask, out["retcode"], out["comment"], out["error"], rc_collect)

    # ============================================================
    # NEW: AUTONOMOUS POSITION TRACKING
    # ============================================================
    if out["ok"] and POSITION_MANAGER_AVAILABLE:
        try:
            _trace("Trade successful. Syncing positions with Position Manager...", debug_log)
            update_positions()
            out["position_tracking_status"] = "synced"
            _trace("✅ Position tracking synced successfully.", debug_log)
        except Exception as e:
            out["position_tracking_status"] = "failed"
            _trace(f"⚠️ Position tracking failed: {e}", debug_log)
    elif out["ok"] and not POSITION_MANAGER_AVAILABLE:
        out["position_tracking_status"] = "unavailable"
        _trace("⚠️ Position Manager not available. Positions not tracked.", debug_log)
    # ============================================================

    return out
