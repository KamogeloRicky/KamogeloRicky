"""
Divergence Detector Module - Momentum Validation

This module scans for hidden bullish/bearish divergences between price structure
and RSI/MACD at key structural levels (swing highs/lows, BOS points).

Divergence signals weakness in a trend and can prevent failed breakout entries.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict

try:
    import ta
except ImportError:
    ta = None


def _find_swing_points(series: pd.Series, n: int = 5) -> Dict[str, list]:
    """Finds swing highs and lows."""
    highs = []
    lows = []
    
    for i in range(n, len(series) - n):
        # Swing high
        if series.iloc[i] == series.iloc[i-n:i+n+1].max():
            highs.append(i)
        # Swing low
        if series.iloc[i] == series.iloc[i-n:i+n+1].min():
            lows.append(i)
    
    return {"highs": highs, "lows": lows}


def detect_divergence(df: pd.DataFrame, direction: str) -> Optional[Dict]:
    """
    Detects hidden divergence between price and RSI.
    
    For Buy: Look for higher lows in price but lower lows in RSI (bullish divergence)
    For Sell: Look for lower highs in price but higher highs in RSI (bearish divergence)
    """
    if ta is None or len(df) < 30:
        return None
    
    try:
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # Find swing points
        price_swings = _find_swing_points(df['close'])
        rsi_swings = _find_swing_points(rsi)
        
        if direction == "Buy":
            # Look for bullish divergence (higher lows in price, lower lows in RSI)
            price_lows = price_swings["lows"][-3:] if len(price_swings["lows"]) >= 3 else []
            rsi_lows = rsi_swings["lows"][-3:] if len(rsi_swings["lows"]) >= 3 else []
            
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                price_val1 = df['close'].iloc[price_lows[-2]]
                price_val2 = df['close'].iloc[price_lows[-1]]
                rsi_val1 = rsi.iloc[rsi_lows[-2]]
                rsi_val2 = rsi.iloc[rsi_lows[-1]]
                
                if price_val2 > price_val1 and rsi_val2 < rsi_val1:
                    return {
                        "type": "bullish_divergence",
                        "strength": "strong" if (rsi_val1 - rsi_val2) > 10 else "moderate",
                        "signal": "Buy",
                        "confidence_boost": 0.15
                    }
        
        elif direction == "Sell":
            # Look for bearish divergence (lower highs in price, higher highs in RSI)
            price_highs = price_swings["highs"][-3:] if len(price_swings["highs"]) >= 3 else []
            rsi_highs = rsi_swings["highs"][-3:] if len(rsi_swings["highs"]) >= 3 else []
            
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                price_val1 = df['close'].iloc[price_highs[-2]]
                price_val2 = df['close'].iloc[price_highs[-1]]
                rsi_val1 = rsi.iloc[rsi_highs[-2]]
                rsi_val2 = rsi.iloc[rsi_highs[-1]]
                
                if price_val2 < price_val1 and rsi_val2 > rsi_val1:
                    return {
                        "type": "bearish_divergence",
                        "strength": "strong" if (rsi_val2 - rsi_val1) > 10 else "moderate",
                        "signal": "Sell",
                        "confidence_boost": 0.15
                    }
    except Exception:
        return None
    
    return None


def check_for_negative_divergence(df: pd.DataFrame, proposed_direction: str) -> Dict:
    """
    Checks if there is a divergence AGAINST the proposed trade direction.
    Returns a penalty dict if negative divergence found.
    """
    result = {"has_negative_divergence": False, "confidence_penalty": 0.0, "reason": None}
    
    try:
        if ta is None:
            return result
        
        # Check opposite direction
        opposite_dir = "Sell" if proposed_direction == "Buy" else "Buy"
        div = detect_divergence(df, opposite_dir)
        
        if div and div["signal"] == opposite_dir:
            result["has_negative_divergence"] = True
            result["confidence_penalty"] = -0.20 if div["strength"] == "strong" else -0.10
            result["reason"] = f"{div['type']} detected against proposed {proposed_direction} direction"
    except Exception:
        pass
    
    return result
