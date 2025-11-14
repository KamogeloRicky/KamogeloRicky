"""
Liquidity Hunter Module - The Trap Springer

This module actively stalks liquidity pools (PDH/PDL, equal highs/lows, Order Blocks)
and generates a "liquidity sweep entry" signal when:
1. Price sweeps a known liquidity level.
2. Price reverses with a rejection wick.
3. A pullback candle confirms the reversal.

This is a "Tier 1.5" signal (between Chimera and Context Engine).
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple

try:
    from data import fetch_candles
except ImportError:
    fetch_candles = None

try:
    from analysis import compute_atr, safe_round
except ImportError:
    def compute_atr(df, period=14): return None
    def safe_round(x, nd=6): return round(float(x), nd) if x else x


def _detect_liquidity_pools(df: pd.DataFrame, lookback: int = 100) -> Dict[str, List[float]]:
    """Identifies key liquidity levels: equal highs/lows and recent swing extremes."""
    pools = {"highs": [], "lows": []}
    try:
        segment = df.tail(lookback)
        
        # Equal highs (within 0.02% tolerance)
        highs = segment['high'].values
        for i in range(1, len(highs)):
            if abs(highs[i] - highs[i-1]) / highs[i] < 0.0002:
                pools["highs"].append(float(highs[i]))
        
        # Equal lows
        lows = segment['low'].values
        for i in range(1, len(lows)):
            if abs(lows[i] - lows[i-1]) / lows[i] < 0.0002:
                pools["lows"].append(float(lows[i]))
        
        # Add swing extremes
        pools["highs"].append(float(segment['high'].max()))
        pools["lows"].append(float(segment['low'].min()))
        
        # Deduplicate
        pools["highs"] = sorted(list(set(pools["highs"])))
        pools["lows"] = sorted(list(set(pools["lows"])))
    except Exception:
        pass
    return pools


def _detect_sweep_and_reversal(df_micro: pd.DataFrame, level: float, direction: str) -> Optional[Dict]:
    """
    Checks if price swept a liquidity level and reversed.
    Returns the index and price of the reversal candle if detected.
    """
    try:
        for i in range(len(df_micro) - 20, len(df_micro)):
            candle = df_micro.iloc[i]
            
            if direction == "Buy":  # Looking for sweep below, then reversal up
                if candle['low'] < level and candle['close'] > level:
                    # Sweep occurred, check for rejection wick
                    body = abs(candle['close'] - candle['open'])
                    lower_wick = min(candle['open'], candle['close']) - candle['low']
                    if lower_wick > body * 1.5:  # Strong rejection
                        return {
                            "sweep_type": "Buy",
                            "level_swept": level,
                            "reversal_candle_idx": i,
                            "reversal_price": float(candle['close']),
                            "rejection_wick": float(lower_wick)
                        }
            
            elif direction == "Sell":  # Looking for sweep above, then reversal down
                if candle['high'] > level and candle['close'] < level:
                    body = abs(candle['close'] - candle['open'])
                    upper_wick = candle['high'] - max(candle['open'], candle['close'])
                    if upper_wick > body * 1.5:
                        return {
                            "sweep_type": "Sell",
                            "level_swept": level,
                            "reversal_candle_idx": i,
                            "reversal_price": float(candle['close']),
                            "rejection_wick": float(upper_wick)
                        }
    except Exception:
        return None
    return None


def hunt_liquidity_sweep(symbol: str, timeframe: str = '1m') -> Optional[Dict]:
    """
    Main orchestrator for Liquidity Hunter.
    Returns a sweep entry signal if conditions are met.
    """
    if not fetch_candles:
        return None
    
    print(f"  🎯 Liquidity Hunter: Scanning {symbol} for sweep opportunities...")
    
    # Fetch HTF data to identify liquidity pools
    df_htf = fetch_candles(symbol, '15m', n=200)
    if df_htf is None or df_htf.empty:
        return None
    
    pools = _detect_liquidity_pools(df_htf)
    if not pools["highs"] and not pools["lows"]:
        print("  -> No liquidity pools identified.")
        return None
    
    print(f"  -> Found {len(pools['highs'])} high pools and {len(pools['lows'])} low pools.")
    
    # Fetch LTF data to monitor for sweeps
    df_ltf = fetch_candles(symbol, timeframe, n=500)
    if df_ltf is None or df_ltf.empty:
        return None
    
    current_price = float(df_ltf['close'].iloc[-1])
    
    # Check for Buy setup (sweep below support, reverse up)
    for level in pools["lows"]:
        if current_price < level * 1.005:  # Price is near the level
            sweep = _detect_sweep_and_reversal(df_ltf, level, "Buy")
            if sweep:
                atr = compute_atr(df_ltf)
                entry = sweep["reversal_price"]
                sl = level - (atr * 0.5 if atr else entry * 0.001)
                tp = entry + (abs(entry - sl) * 3.0)
                
                print(f"  ✅ LIQUIDITY SWEEP ENTRY FOUND: Buy at {entry}")
                return {
                    "status": "Success",
                    "final_suggestion": "Buy",
                    "entry_point": safe_round(entry),
                    "sl": safe_round(sl),
                    "tp": safe_round(tp),
                    "confidence": 0.88,
                    "precision_reason": f"LiquidityHunt/Sweep@{safe_round(level)}",
                    "sweep_details": sweep
                }
    
    # Check for Sell setup (sweep above resistance, reverse down)
    for level in pools["highs"]:
        if current_price > level * 0.995:
            sweep = _detect_sweep_and_reversal(df_ltf, level, "Sell")
            if sweep:
                atr = compute_atr(df_ltf)
                entry = sweep["reversal_price"]
                sl = level + (atr * 0.5 if atr else entry * 0.001)
                tp = entry - (abs(entry - sl) * 3.0)
                
                print(f"  ✅ LIQUIDITY SWEEP ENTRY FOUND: Sell at {entry}")
                return {
                    "status": "Success",
                    "final_suggestion": "Sell",
                    "entry_point": safe_round(entry),
                    "sl": safe_round(sl),
                    "tp": safe_round(tp),
                    "confidence": 0.88,
                    "precision_reason": f"LiquidityHunt/Sweep@{safe_round(level)}",
                    "sweep_details": sweep
                }
    
    print("  -> No active sweep setups detected.")
    return None
