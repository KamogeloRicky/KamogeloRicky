"""
Regime Sentinel Module - Mid-Trade Regime Monitoring (v2.0 - Autonomous)

This module continuously monitors the Hurst Exponent for symbols with open trades.
If the Hurst shifts significantly, it triggers defensive actions.

NOW AUTONOMOUS: Automatically loads open positions from position_manager.py
"""
import os
import json
from typing import List, Dict, Optional

try:
    from data import fetch_candles
except ImportError:
    fetch_candles = None

try:
    from market_psychology import calculate_hurst
except ImportError:
    calculate_hurst = None

try:
    from position_manager import load_positions
except ImportError:
    def load_positions():
        return []


def save_regime_state(symbol: str, hurst: float, filepath: str = "regime_state.json"):
    """Saves the current regime state for a symbol."""
    try:
        state = {}
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                state = json.load(f)
        
        state[symbol] = {"hurst": hurst}
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def load_regime_state(symbol: str, filepath: str = "regime_state.json") -> Optional[float]:
    """Loads the last known Hurst for a symbol."""
    try:
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r') as f:
            state = json.load(f)
        return state.get(symbol, {}).get("hurst")
    except Exception:
        return None


def monitor_regime_shift(symbol: str, shift_threshold: float = 0.15) -> Optional[Dict]:
    """
    Monitors a symbol for regime shift.
    Returns alert dict if shift detected, None otherwise.
    """
    if not fetch_candles or not calculate_hurst:
        return None
    
    df = fetch_candles(symbol, '1h', n=500)
    if df is None or df.empty:
        return None
    
    current_hurst = calculate_hurst(df['close'])
    if current_hurst is None:
        return None
    
    previous_hurst = load_regime_state(symbol)
    
    if previous_hurst is None:
        # First time monitoring, just save
        save_regime_state(symbol, current_hurst)
        return None
    
    shift = abs(current_hurst - previous_hurst)
    
    if shift > shift_threshold:
        alert = {
            "symbol": symbol,
            "previous_hurst": round(previous_hurst, 3),
            "current_hurst": round(current_hurst, 3),
            "shift": round(shift, 3),
            "action_recommended": None
        }
        
        # Determine recommended action
        if previous_hurst > 0.55 and current_hurst < 0.50:
            alert["action_recommended"] = "tighten_stops"
            alert["reason"] = "Market shifted from trending to choppy. Tighten stops by 50%."
        elif previous_hurst < 0.45 and current_hurst > 0.50:
            alert["action_recommended"] = "widen_stops"
            alert["reason"] = "Market shifted from mean-reverting to trending. Consider letting winners run."
        else:
            alert["action_recommended"] = "breakeven"
            alert["reason"] = "Significant regime shift detected. Move stop to breakeven."
        
        # Update saved state
        save_regime_state(symbol, current_hurst)
        
        return alert
    
    # No significant shift, update state
    save_regime_state(symbol, current_hurst)
    return None


def run_sentinel_scan():
    """Main function to scan all open positions for regime shifts. NOW AUTONOMOUS."""
    print("\n🔍 Regime Sentinel: Starting autonomous scan...")
    
    # Automatically load positions from Position Manager
    positions = load_positions()
    
    if not positions:
        print("  No open positions to monitor.")
        return
    
    print(f"  Monitoring {len(positions)} open position(s)...")
    
    alerts = []
    for pos in positions:
        symbol = pos.get('symbol')
        if not symbol:
            continue
        
        print(f"  Checking {symbol}...")
        alert = monitor_regime_shift(symbol)
        if alert:
            alerts.append(alert)
            print(f"  ⚠️ REGIME SHIFT DETECTED: {alert['reason']}")
    
    if not alerts:
        print("  ✅ All positions: No regime shifts detected.")
    else:
        print(f"\n🚨 {len(alerts)} regime shift(s) detected. Review recommended actions.")
        for alert in alerts:
            print(f"   {alert['symbol']}: {alert['action_recommended']} - {alert['reason']}")


if __name__ == "__main__":
    run_sentinel_scan()
