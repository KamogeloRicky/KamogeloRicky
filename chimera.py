"""
Project Chimera - Quantum Entry Module (v1.0)

This module operates independently to find ultra-high-probability "sniper" entries.
It is designed to be called BEFORE the main analysis engine as a high-conviction filter.

Core Logic:
1.  Fetches Higher Timeframe (HTF) and Lower Timeframe (LTF) data.
2.  Identifies HTF "Kill Zones" based on Fair Value Gaps (FVG) and Order Blocks (OB).
3.  Waits for price to enter a Kill Zone.
4.  Monitors the LTF (1-minute) for a confirming Change of Character (mCHOCH).
5.  If all conditions are met, it calculates a sniper limit entry, a protected stop loss,
    and a high-reward take profit.
6.  If any condition fails, it returns None, signaling the orchestrator to fall back
    to the standard analysis engine.
"""

import pandas as pd
from typing import Optional, Dict, Tuple, List, Any

# We need access to the robust data fetching and ATR calculation
try:
    from data import fetch_candles
    from analysis import compute_atr, _pips_for_pair, safe_round
except ImportError:
    print("❌ CHIMERA ERROR: Cannot import `fetch_candles` or `compute_atr`. Make sure data.py and analysis.py are available.")
    fetch_candles = None
    compute_atr = None
    _pips_for_pair = lambda p: 0.0001
    safe_round = lambda x, n=6: x


# --- Chimera Helper Functions ---

def _detect_fvg_zone_chimera(df: pd.DataFrame, direction: str) -> Optional[Tuple[float, float]]:
    """
    Finds the most recent Fair Value Gap (FVG) on the given dataframe.
    For a "Buy" signal, we expect a retracement down into a bullish FVG.
    For a "Sell" signal, we expect a retracement up into a bearish FVG.
    """
    try:
        if direction == "Buy":
            # Look for a bullish FVG (gap between candle i-1 high and i+1 low)
            for i in range(len(df) - 3, 0, -1):
                if df['high'].iloc[i-1] < df['low'].iloc[i+1]:
                    return (float(df['high'].iloc[i-1]), float(df['low'].iloc[i+1]))
        elif direction == "Sell":
            # Look for a bearish FVG (gap between candle i-1 low and i+1 high)
            for i in range(len(df) - 3, 0, -1):
                if df['low'].iloc[i-1] > df['high'].iloc[i+1]:
                    return (float(df['high'].iloc[i+1]), float(df['low'].iloc[i-1]))
    except Exception:
        return None
    return None


def _last_opposite_order_block_chimera(df: pd.DataFrame, direction: str) -> Optional[Tuple[float, float]]:
    """
    Finds the last valid Order Block (OB) to act as a point of interest.
    For a "Buy", it's the last down-candle before a strong up-move.
    For a "Sell", it's the last up-candle before a strong down-move.
    """
    try:
        for i in range(len(df) - 2, 0, -1):
            candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            if direction == "Buy" and candle['close'] < candle['open'] and next_candle['close'] > candle['high']:
                return (float(candle['low']), float(candle['high'])) # Return the body of the bearish OB
            
            if direction == "Sell" and candle['close'] > candle['open'] and next_candle['close'] < candle['low']:
                return (float(candle['low']), float(candle['high'])) # Return the body of the bullish OB
    except Exception:
        return None
    return None

def _get_swing_points_chimera(series: pd.Series, n: int = 3) -> Tuple[List[int], List[int]]:
    """Finds indices of swing highs and lows."""
    highs = (series == series.rolling(2 * n + 1, center=True, min_periods=1).max()).values
    lows = (series == series.rolling(2 * n + 1, center=True, min_periods=1).min()).values
    return [i for i, is_high in enumerate(highs) if is_high], [i for i, is_low in enumerate(lows) if is_low]


def _detect_micro_choch_chimera(df_micro: pd.DataFrame, direction: str, lookback: int = 40) -> Optional[Dict]:
    """Detects a Change of Character (mCHOCH) on the 1-minute timeframe."""
    try:
        segment = df_micro.iloc[-lookback:]
        if len(segment) < 15: return None
        
        swing_n = 3
        high_indices, low_indices = _get_swing_points_chimera(segment['high' if direction == "Buy" else 'low'], n=swing_n)

        if direction == "Buy":
            # Find the most recent confirmed swing high that isn't in the last few bars
            relevant_highs = [i for i in high_indices if i < len(segment) - (swing_n + 1)]
            if not relevant_highs: return None
            
            last_swing_high_idx = relevant_highs[-1]
            swing_high_price = segment.iloc[last_swing_high_idx]['high']
            
            # Check if this high was broken in the bars that followed
            if segment['high'].iloc[last_swing_high_idx:].max() > swing_high_price:
                break_candle_iloc = segment.index[segment['high'].iloc[last_swing_high_idx:].argmax() + last_swing_high_idx]
                break_candle_idx_in_segment = segment.index.get_loc(break_candle_iloc)

                # Find the lowest swing low that occurred before the break
                lows_before_break = [i for i in low_indices if i < break_candle_idx_in_segment]
                if not lows_before_break: return None
                
                sl_pivot_idx = lows_before_break[-1]
                return {
                    "type": "mCHOCH_Buy", 
                    "break_price": swing_high_price, 
                    "sl_pivot_price": segment.iloc[sl_pivot_idx]['low']
                }

        elif direction == "Sell":
            relevant_lows = [i for i in low_indices if i < len(segment) - (swing_n + 1)]
            if not relevant_lows: return None
            
            last_swing_low_idx = relevant_lows[-1]
            swing_low_price = segment.iloc[last_swing_low_idx]['low']

            if segment['low'].iloc[last_swing_low_idx:].min() < swing_low_price:
                break_candle_iloc = segment.index[segment['low'].iloc[last_swing_low_idx:].argmin() + last_swing_low_idx]
                break_candle_idx_in_segment = segment.index.get_loc(break_candle_iloc)

                highs_before_break = [i for i in high_indices if i < break_candle_idx_in_segment]
                if not highs_before_break: return None
                
                sl_pivot_idx = highs_before_break[-1]
                return {
                    "type": "mCHOCH_Sell", 
                    "break_price": swing_low_price, 
                    "sl_pivot_price": segment.iloc[sl_pivot_idx]['high']
                }
    except Exception:
        return None
    return None


def find_quantum_entry(symbol: str, htf: str = '15m', ltf: str = '1m') -> Optional[Dict]:
    """
    Main orchestrator for the Quantum Entry module. Attempts to find a sniper entry.
    Returns a complete trade dictionary if successful, otherwise None.
    """
    if fetch_candles is None: return None

    print(f"🔬 Chimera: Hunting for Quantum Entry on {symbol}...")

    # --- Step 1: Fetch Data ---
    df_htf = fetch_candles(symbol, htf, n=200)
    df_ltf = fetch_candles(symbol, ltf, n=500)

    if df_htf is None or df_ltf is None or df_htf.empty or df_ltf.empty:
        print("  -> Chimera: Data fetching failed. Aborting.")
        return None

    # --- Step 2: Determine HTF Directional Bias ---
    # Simple EMA cross on HTF to determine the dominant trend to follow
    ema_fast = df_htf['close'].ewm(span=21, adjust=False).mean().iloc[-1]
    ema_slow = df_htf['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    htf_direction = "Buy" if ema_fast > ema_slow else "Sell"
    print(f"  -> Chimera: HTF ({htf}) bias is {htf_direction}.")

    # --- Step 3: Identify HTF "Kill Zone" ---
    # A "Kill Zone" is a confluence of FVG and OB in the direction of the retracement.
    # For a Buy bias, we look for a retracement down into a Bullish FVG/OB.
    # So we search for the *opposite* patterns.
    fvg_zone = _detect_fvg_zone_chimera(df_htf, htf_direction)
    ob_zone = _last_opposite_order_block_chimera(df_htf, htf_direction)

    if not fvg_zone and not ob_zone:
        print("  -> Chimera: No HTF FVG or OB found to form a Kill Zone.")
        return None

    # Combine the zones to get the full Kill Zone
    zone_low = min(p[0] for p in [fvg_zone, ob_zone] if p)
    zone_high = max(p[1] for p in [fvg_zone, ob_zone] if p)
    print(f"  -> Chimera: HTF Kill Zone identified: {zone_low:.5f} - {zone_high:.5f}")

    # --- Step 4: Check for Price Interaction & LTF Confirmation ---
    last_ltf_price = df_ltf['close'].iloc[-1]
    if not (zone_low <= last_ltf_price <= zone_high):
        print("  -> Chimera: Price is not currently within the Kill Zone.")
        return None
    
    print("  -> Chimera: Price is in Kill Zone. Monitoring for mCHOCH...")
    mchoch = _detect_micro_choch_chimera(df_ltf, htf_direction)
    if not mchoch:
        print("  -> Chimera: No confirming mCHOCH on LTF. No entry.")
        return None

    print(f"  -> ✅ Chimera: {mchoch['type']} confirmed!")

    # --- Step 5: Calculate Sniper Entry, SL, and TP ---
    entry_reason = f"Quantum/{mchoch['type']}"
    
    # Sniper entry is at the OB on the LTF that caused the mCHOCH
    ltf_ob = _last_opposite_order_block_chimera(df_ltf, htf_direction, lookback=40)
    if ltf_ob:
        entry_price = ltf_ob[1] if htf_direction == "Buy" else ltf_ob[0]
        entry_reason += "+LTF_OB"
    else:
        entry_price = mchoch['break_price'] # Fallback
        
    sl_price = mchoch['sl_pivot_price']
    
    # Add a protective buffer to the SL
    ltf_atr = compute_atr(df_ltf, period=14)
    point_size = _pips_for_pair(symbol)
    sl_buffer = max(3 * point_size, (ltf_atr or abs(entry_price - sl_price) * 0.1) * 0.5)
    final_sl = sl_price - sl_buffer if htf_direction == "Buy" else sl_price + sl_buffer
    
    # Calculate a 3R Take Profit
    risk_dist = abs(entry_price - final_sl)
    if risk_dist == 0: return None # Avoid division by zero
    final_tp = entry_price + (3 * risk_dist) if htf_direction == "Buy" else entry_price - (3 * risk_dist)

    print("  -> ✅ Chimera: Quantum Entry locked and loaded.")

    return {
        "status": "Success",
        "final_suggestion": htf_direction,
        "entry_point": safe_round(entry_price),
        "sl": safe_round(final_sl),
        "tp": safe_round(final_tp),
        "confidence": 0.95, # Quantum entries are highest conviction
        "precision_reason": entry_reason,
        "quantum_details": {
            "htf_bias": htf_direction,
            "htf_kill_zone": (zone_low, zone_high),
            "mchoch_details": mchoch
        }
    }
