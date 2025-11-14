"""
Data Module - Robust MT5 Data Fetcher with Auto-Reconnect

This module handles all data fetching from MetaTrader 5 with:
- Automatic connection retry
- Symbol resolution with variant matching
- Graceful fallbacks
- Connection state management

Uses hardcoded credentials from analysis.py for consistency.
"""
import os
import pandas as pd
from typing import Optional, List
from datetime import datetime, timedelta

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("❌ CRITICAL: MetaTrader5 library not installed. Run: pip install MetaTrader5")

# MT5 credentials (same as in analysis.py)
MT5_LOGIN = 211825028
MT5_PASSWORD = "RICKy@081"
MT5_SERVER = "Exness-MT5Trial9"

# Connection state
_MT5_INITIALIZED = False
_SYMBOL_CACHE = {}

def _ensure_mt5_connected() -> bool:
    """Ensures MT5 is connected. Auto-reconnects if needed."""
    global _MT5_INITIALIZED
    
    if not MT5_AVAILABLE:
        return False
    
    # Check if already connected
    if _MT5_INITIALIZED:
        try:
            # Verify connection is alive
            account_info = mt5.account_info()
            if account_info is not None:
                return True
        except Exception:
            _MT5_INITIALIZED = False
    
    # Attempt to initialize
    print("🔌 Connecting to MT5...")
    try:
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed. Error: {mt5.last_error()}")
            return False
        
        # Login with credentials
        authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        if not authorized:
            print(f"❌ MT5 login failed. Error: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        _MT5_INITIALIZED = True
        print(f"✅ Connected to MT5 | Account: {MT5_LOGIN} | Server: {MT5_SERVER}")
        return True
    
    except Exception as e:
        print(f"❌ MT5 connection exception: {e}")
        return False


def _resolve_symbol(symbol_query: str) -> Optional[str]:
    """
    Resolves a symbol query to an actual MT5 symbol.
    Handles variants like XAUUSDm, XAUUSD, GOLD, etc.
    """
    if not _ensure_mt5_connected():
        return None
    
    # Check cache first
    if symbol_query in _SYMBOL_CACHE:
        return _SYMBOL_CACHE[symbol_query]
    
    print(f"🔎 Resolving symbol: {symbol_query}")
    
    # Try exact match first
    try:
        info = mt5.symbol_info(symbol_query)
        if info is not None:
            # Enable symbol if hidden
            if not info.visible:
                mt5.symbol_select(symbol_query, True)
            _SYMBOL_CACHE[symbol_query] = symbol_query
            print(f"✅ Symbol resolved: {symbol_query}")
            return symbol_query
    except Exception:
        pass
    
    # Try to get all symbols and fuzzy match
    try:
        all_symbols = mt5.symbols_get()
        if all_symbols is None or len(all_symbols) == 0:
            print("⚠️ Could not retrieve symbol list from MT5.")
            return None
        
        # Generate variants
        base = symbol_query.upper()
        variants = [
            base,
            base.rstrip('m').rstrip('c'),  # Remove suffix
            f"{base}m",
            f"{base}c",
            f"{base}.a",
            f"{base}#",
        ]
        
        # Special cases
        if "XAU" in base or "GOLD" in base:
            variants.extend(["XAUUSD", "XAUUSDm", "XAUUSDc", "GOLD", "GOLDm", "XAUUSD.a"])
        
        # Search for match
        for variant in variants:
            for sym in all_symbols:
                if sym.name.upper() == variant.upper():
                    # Enable symbol if hidden
                    if not sym.visible:
                        mt5.symbol_select(sym.name, True)
                    _SYMBOL_CACHE[symbol_query] = sym.name
                    print(f"✅ Symbol resolved: {symbol_query} → {sym.name}")
                    return sym.name
        
        print(f"❌ Symbol '{symbol_query}' not found in MT5. Available GOLD symbols:")
        for sym in all_symbols:
            if "XAU" in sym.name.upper() or "GOLD" in sym.name.upper():
                print(f"   - {sym.name}")
        
        return None
    
    except Exception as e:
        print(f"❌ Symbol resolution failed: {e}")
        return None


def fetch_candles(symbol: str, timeframe: str, n: int = 500) -> Optional[pd.DataFrame]:
    """
    Fetches OHLCV candles from MT5.
    
    Args:
        symbol: Symbol to fetch (e.g., 'XAUUSD', 'EURUSD')
        timeframe: Timeframe string ('1m', '5m', '15m', '1h', '4h', 'D1')
        n: Number of bars to fetch
    
    Returns:
        DataFrame with columns: time, open, high, low, close, volume
        None if fetch fails
    """
    if not _ensure_mt5_connected():
        print("❌ Cannot fetch candles. MT5 not connected.")
        return None
    
    # Resolve symbol
    resolved_symbol = _resolve_symbol(symbol)
    if resolved_symbol is None:
        print(f"❌ Cannot fetch candles. Symbol '{symbol}' could not be resolved.")
        return None
    
    # Map timeframe string to MT5 constant
    timeframe_map = {
        '1m': mt5.TIMEFRAME_M1,
        '5m': mt5.TIMEFRAME_M5,
        '15m': mt5.TIMEFRAME_M15,
        '30m': mt5.TIMEFRAME_M30,
        '1h': mt5.TIMEFRAME_H1,
        '4h': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        '1D': mt5.TIMEFRAME_D1,
    }
    
    tf_constant = timeframe_map.get(timeframe)
    if tf_constant is None:
        print(f"❌ Invalid timeframe: {timeframe}")
        return None
    
    try:
        # Fetch rates
        rates = mt5.copy_rates_from_pos(resolved_symbol, tf_constant, 0, n)
        
        if rates is None or len(rates) == 0:
            print(f"❌ No data returned for {resolved_symbol} {timeframe}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Ensure required columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        if not all(col in df.columns for col in required_cols):
            print(f"❌ Missing required columns in fetched data")
            return None
        
        # Rename tick_volume to volume
        df = df.rename(columns={'tick_volume': 'volume'})
        
        # Set time as index
        df = df.set_index('time')
        
        print(f"✅ Fetched {len(df)} bars for {resolved_symbol} {timeframe}")
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    except Exception as e:
        print(f"❌ Error fetching candles: {e}")
        return None


def list_symbols() -> List[str]:
    """Returns a list of all available symbols in MT5."""
    if not _ensure_mt5_connected():
        return []
    
    try:
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        return [s.name for s in symbols]
    except Exception as e:
        print(f"❌ Error listing symbols: {e}")
        return []


def shutdown_mt5():
    """Gracefully shuts down MT5 connection."""
    global _MT5_INITIALIZED
    
    if MT5_AVAILABLE and _MT5_INITIALIZED:
        try:
            mt5.shutdown()
            _MT5_INITIALIZED = False
            print("🔌 MT5 connection closed.")
        except Exception as e:
            print(f"⚠️ Error during MT5 shutdown: {e}")


# Auto-connect on import
if MT5_AVAILABLE:
    _ensure_mt5_connected()
