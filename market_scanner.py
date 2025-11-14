"""
Market Scanner & Confluence Engine ("The Hunter")

Purpose:
- Scans a user-defined watchlist of symbols across multiple timeframes (D1, H4, H1).
- Establishes a "Directional Bias" score for each symbol based on trend confluence.
- A high positive score indicates a strong bullish consensus; high negative is bearish.
- The CLI will use this to identify and rank high-probability trading opportunities.

Method:
- Uses simple moving averages (SMA 20/50/200) for trend assessment on each timeframe.
- Price > SMA50 > SMA200 = Bullish regime.
- Price < SMA50 < SMA200 = Bearish regime.
- Weights higher timeframes more heavily (D1 > H4 > H1) for a strategic view.
- Intelligently resolves broker-specific symbol suffixes (e.g., finding 'EURUSDm' from 'EURUSD').
"""

from typing import List, Dict, Optional
import time

try:
    import MetaTrader5 as mt5
    # Assumes entry_optimizer.py is available for its helpers, creating stubs if not.
    from entry_optimizer import _get_rates, _sma, _scalar, _ensure_attached_and_visible
except Exception:
    mt5 = None
    # Provide stubs if modules are missing, so scanner can be imported without error
    def _get_rates(*args, **kwargs): return []
    def _sma(*args, **kwargs): return []
    def _scalar(x, fallback=None): return x if x is not None else fallback
    def _ensure_attached_and_visible(*args, **kwargs): return False

# --- Configuration ---
# The bot's "hunting ground." Add any symbol you are interested in.
WATCHLIST = [
    # FX Majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF",
    # Key FX Crosses
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "EURAUD", "GBPAUD",
    # Commodities
    "XAUUSD", "XAGUSD",
    # Crude Oil (WTI)
    "USOUSD", "WTI", "OILUSD",
    # Major Indices
    "US500", "US100", "US30", "GER30", "GER40", "JP225", "STOXX50", "UK100", "DAX",
    # Major Cryptocurrencies
    "BTCUSD", "ETHUSD"
]

# Timeframes and their weight in the final bias score. D1 is most important.
BIAS_CONFIG = {
    "D1": {"weight": 0.5, "bars": 300},
    "H4": {"weight": 0.3, "bars": 300},
    "H1": {"weight": 0.2, "bars": 300},
}

def get_directional_bias(symbol: str) -> Dict:
    """
    Calculates a trend confluence score for a single symbol.
    Score ranges from approx -1.0 (strong bearish) to +1.0 (strong bullish).
    A score near 0 indicates ranging or conflicting signals.
    """
    if mt5 is None or not _ensure_attached_and_visible(symbol):
        return {"symbol": symbol, "bias_score": 0.0, "status": "mt5_unavailable_or_symbol_hidden", "details": {}}

    total_score = 0.0
    details = {}

    for tf, config in BIAS_CONFIG.items():
        weight = config["weight"]
        # Fetch more bars than needed for indicator stability
        candles = _get_rates(symbol, tf, config["bars"] + 10)
        if len(candles) < 210: # Need at least 200 bars for the longest SMA
            details[tf] = "not_enough_data"
            continue

        closes = [_scalar(c.c, 0.0) for c in candles]
        sma20 = _sma(closes, 20)
        sma50 = _sma(closes, 50)
        sma200 = _sma(closes, 200)

        # Ensure all indicators have values at the last bar
        if any(val is None or val != val for val in [closes[-1], sma20[-1], sma50[-1], sma200[-1]]):
            details[tf] = "indicator_calculation_failed"
            continue
            
        last_close = closes[-1]
        s20, s50, s200 = sma20[-1], sma50[-1], sma200[-1]
        
        tf_score = 0.0
        # Regime Scoring: A perfect trend gets a full score.
        if last_close > s20 > s50 > s200:
            tf_score = 1.0  # Perfect Bullish Alignment
        elif last_close < s20 < s50 < s200:
            tf_score = -1.0 # Perfect Bearish Alignment
        else:
            # Partial Scoring for less-perfect trends
            if last_close > s50: tf_score += 0.3
            if s50 > s200: tf_score += 0.4
            if last_close < s50: tf_score -= 0.3
            if s50 < s200: tf_score -= 0.4
        
        total_score += tf_score * weight
        details[tf] = {"score": round(tf_score, 2), "close": last_close, "sma50": round(s50, 4), "sma200": round(s200, 4)}

    return {
        "symbol": symbol,
        "bias_score": round(total_score, 3),
        "status": "ok",
        "details": details
    }

def scan_market(watchlist: Optional[List[str]] = None) -> List[Dict]:
    """
    Scans a watchlist and returns a list of symbols with their directional bias scores,
    sorted by the strongest trend alignment (absolute score).
    """
    if watchlist is None:
        watchlist = WATCHLIST
    
    print(f"📡 Scanning market for directional bias across {len(watchlist)} potential symbols...")
    results = []
    
    # Intelligently resolve broker-specific suffixes (e.g., '.c', '.m')
    resolved_watchlist = []
    if _ensure_attached_and_visible("EURUSD"): # Check connection once
        all_symbols = mt5.symbols_get()
        if all_symbols:
            all_names = {s.name for s in all_symbols}
            for sym in watchlist:
                found = False
                # Common suffixes to check for
                for suffix in ["", "c", "m", ".pro", ".ecn", "i", "_i"]:
                    candidate = sym.upper() + suffix
                    if candidate in all_names:
                        resolved_watchlist.append(candidate)
                        found = True
                        break
                if not found:
                    print(f"  - Warning: Could not resolve base symbol '{sym}' for this broker.")
        else:
            resolved_watchlist = watchlist # Fallback to raw list
    else:
        print("  - Critical Warning: Cannot connect to MT5 to resolve broker symbols. Using base names.")
        resolved_watchlist = watchlist

    print(f"  - Hunter operating on {len(resolved_watchlist)} resolved symbols. e.g., {', '.join(resolved_watchlist[:3])}...")
    
    for i, symbol in enumerate(resolved_watchlist):
        # Provide a live progress indicator
        print(f"  - Scanning [{i+1}/{len(resolved_watchlist)}] {symbol}", end='\r')
        bias_data = get_directional_bias(symbol)
        results.append(bias_data)
        time.sleep(0.05) # Be gentle with the API
    
    print("\n✅ Scan complete. Ranking opportunities...")
    
    # Sort by the most powerful trends (positive or negative)
    results.sort(key=lambda x: abs(x.get("bias_score", 0.0)), reverse=True)
    return results

if __name__ == "__main__":
    # Example of running the scanner directly to see the market state
    top_opportunities = scan_market()
    print("\n--- Top 10 Market Opportunities (by Absolute Bias Strength) ---")
    for item in top_opportunities[:10]:
        direction = "STRONG BULLISH" if item['bias_score'] > 0.5 else "BULLISH" if item['bias_score'] > 0.2 else "STRONG BEARISH" if item['bias_score'] < -0.5 else "BEARISH" if item['bias_score'] < -0.2 else "NEUTRAL/RANGING"
        print(f"Symbol: {item['symbol']:<15} | Score: {item['bias_score']:>+7.3f} | Bias: {direction}")
