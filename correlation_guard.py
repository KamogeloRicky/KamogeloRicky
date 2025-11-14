"""
Correlation Guard Module - Portfolio-Level Risk Management (v2.0 - Autonomous)

This module prevents the bot from taking highly correlated trades that would
result in hidden over-leverage.

Now automatically loads open positions from position_manager.py
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

try:
    from data import fetch_candles
except ImportError:
    fetch_candles = None

try:
    from position_manager import load_positions
except ImportError:
    def load_positions():
        return []


def calculate_correlation(symbol1: str, symbol2: str, bars: int = 50) -> Optional[float]:
    """Calculates the rolling correlation between two symbols."""
    if not fetch_candles:
        return None
    
    try:
        df1 = fetch_candles(symbol1, '1h', n=bars)
        df2 = fetch_candles(symbol2, '1h', n=bars)
        
        if df1 is None or df2 is None or df1.empty or df2.empty:
            return None
        
        # Calculate returns
        returns1 = df1['close'].pct_change().dropna()
        returns2 = df2['close'].pct_change().dropna()
        
        # Align
        min_len = min(len(returns1), len(returns2))
        returns1 = returns1.tail(min_len)
        returns2 = returns2.tail(min_len)
        
        # Correlation
        corr = returns1.corr(returns2)
        return float(corr) if not np.isnan(corr) else None
    except Exception:
        return None


def check_correlation_risk(proposed_symbol: str, 
                           proposed_direction: str,
                           threshold: float = 0.75) -> Dict:
    """
    Checks if the proposed trade would create correlated exposure.
    
    NOW AUTONOMOUS: Automatically loads open positions from position_manager.
    
    Args:
        proposed_symbol: Symbol for the new trade
        proposed_direction: "Buy" or "Sell"
        threshold: Correlation threshold (default 0.75)
    
    Returns:
        Dict with 'ok', 'action', 'correlated_with', 'correlation_value'
    """
    result = {
        "ok": True,
        "action": "allow",
        "correlated_with": None,
        "correlation_value": None,
        "recommendation": None
    }
    
    # Automatically load open positions
    open_positions = load_positions()
    
    if not open_positions:
        print("  🛡️ Correlation Guard: No open positions detected.")
        return result
    
    print(f"  🛡️ Correlation Guard: Checking {proposed_symbol} against {len(open_positions)} open positions...")
    
    for pos in open_positions:
        open_symbol = pos.get('symbol')
        open_direction = pos.get('direction')
        
        if not open_symbol or not open_direction:
            continue
        
        corr = calculate_correlation(proposed_symbol, open_symbol)
        
        if corr is None:
            continue
        
        # Check if same directional exposure on correlated pair
        same_direction = (proposed_direction == open_direction)
        high_corr = abs(corr) > threshold
        
        if same_direction and high_corr:
            result["ok"] = False
            result["correlated_with"] = open_symbol
            result["correlation_value"] = round(corr, 3)
            
            if abs(corr) > 0.85:
                result["action"] = "block"
                result["recommendation"] = f"BLOCK: {proposed_symbol} is {round(corr*100, 1)}% correlated with open position on {open_symbol}. This would create hidden over-leverage."
            else:
                result["action"] = "scale_down"
                result["recommendation"] = f"SCALE DOWN: {proposed_symbol} is {round(corr*100, 1)}% correlated with {open_symbol}. Reduce lot size by 50%."
            
            print(f"  ⚠️ Correlation Alert: {result['recommendation']}")
            return result
    
    print("  ✅ No correlation risk detected.")
    return result
