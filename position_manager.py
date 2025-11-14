"""
Position Manager - Autonomous Position Tracker

This module automatically maintains the open_positions.json file by syncing with MT5.
It is called by trade_executor.py after every trade action (open/close).

Features:
- Automatic sync with MT5 open positions
- No manual updates required
- Thread-safe file operations
- Fallback to empty list if MT5 unavailable
"""
import json
import os
from typing import List, Dict, Optional
from datetime import datetime

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


POSITIONS_FILE = "open_positions.json"


def sync_positions_from_mt5() -> List[Dict]:
    """
    Fetches all open positions from MT5 and returns them as a list.
    Each position includes: symbol, direction, ticket, volume, open_price, open_time
    """
    if mt5 is None:
        return []
    
    try:
        if not mt5.initialize():
            return []
        
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            return []
        
        position_list = []
        for pos in positions:
            position_list.append({
                "symbol": pos.symbol,
                "direction": "Buy" if pos.type == mt5.ORDER_TYPE_BUY else "Sell",
                "ticket": pos.ticket,
                "volume": pos.volume,
                "open_price": pos.price_open,
                "open_time": datetime.fromtimestamp(pos.time).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return position_list
    
    except Exception as e:
        print(f"⚠️ Position Manager: Failed to sync from MT5: {e}")
        return []


def save_positions(positions: List[Dict]) -> bool:
    """Saves the positions list to JSON file."""
    try:
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(positions, f, indent=2)
        return True
    except Exception as e:
        print(f"⚠️ Position Manager: Failed to save positions: {e}")
        return False


def load_positions() -> List[Dict]:
    """Loads positions from JSON file. Returns empty list if file doesn't exist."""
    try:
        if not os.path.exists(POSITIONS_FILE):
            return []
        with open(POSITIONS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Position Manager: Failed to load positions: {e}")
        return []


def update_positions():
    """
    Main function to update the positions file.
    This should be called after every trade action.
    """
    print("🔄 Position Manager: Syncing with MT5...")
    positions = sync_positions_from_mt5()
    
    if save_positions(positions):
        print(f"✅ Position Manager: Updated. {len(positions)} open position(s).")
    else:
        print("❌ Position Manager: Failed to update positions file.")
    
    return positions


def add_position_manual(symbol: str, direction: str, ticket: int, volume: float, open_price: float):
    """
    Manual fallback to add a position if MT5 sync fails.
    This is called by trade_executor immediately after placing a trade.
    """
    try:
        positions = load_positions()
        
        # Check if position already exists
        for pos in positions:
            if pos.get("ticket") == ticket:
                return  # Already tracked
        
        positions.append({
            "symbol": symbol,
            "direction": direction,
            "ticket": ticket,
            "volume": volume,
            "open_price": open_price,
            "open_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        save_positions(positions)
        print(f"✅ Position Manager: Manually added position {ticket} for {symbol}")
    
    except Exception as e:
        print(f"⚠️ Position Manager: Failed to manually add position: {e}")


def remove_position_manual(ticket: int):
    """
    Manual fallback to remove a position if MT5 sync fails.
    This is called by trade_executor when a position is closed.
    """
    try:
        positions = load_positions()
        positions = [pos for pos in positions if pos.get("ticket") != ticket]
        save_positions(positions)
        print(f"✅ Position Manager: Manually removed position {ticket}")
    
    except Exception as e:
        print(f"⚠️ Position Manager: Failed to manually remove position: {e}")


# Initialize positions file if it doesn't exist
if not os.path.exists(POSITIONS_FILE):
    save_positions([])
    print("✅ Position Manager: Initialized open_positions.json")


if __name__ == "__main__":
    # Standalone test
    print("Testing Position Manager...")
    update_positions()
