"""
Strategy Core ("The Guardian") - FINAL VERSION

Purpose:
- Provides centralized, portfolio-level risk management.
- Acts as the single source of truth for strategic decisions and risk parameters.
- Prevents over-exposure to a single currency or correlated assets.
- Enforces master rules like max open trades and max total risk.
- **AUTO-TUNING UPGRADE:** Automatically detects and loads the 'optimized_profile.json'
  if it exists, ensuring the bot always uses the most evolved strategy.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import os

try:
    import MetaTrader5 as mt5
    # Assuming entry_optimizer or a similar helper module is available for MT5 connection
    # If not, this part needs a standalone MT5 initialization helper.
    def _ensure_attached_and_visible(symbol):
        if not mt5.initialize():
            return False
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return False
        if not sym_info.visible:
            return mt5.symbol_select(symbol, True)
        return True
except Exception:
    mt5 = None
    def _ensure_attached_and_visible(*args, **kwargs): return False

class ApprovalResponse:
    """A structured response from the Strategy Core's pre-trade check."""
    def __init__(self, approved: bool, reason: str, adjusted_lot: Optional[float] = None):
        self.approved = approved
        self.reason = reason
        self.adjusted_lot = adjusted_lot

    def __repr__(self):
        return f"ApprovalResponse(approved={self.approved}, reason='{self.reason}', adjusted_lot={self.adjusted_lot})"

class StrategyCore:
    """The central brain for risk and strategy management."""

    def __init__(self, base_profile_path: str = "strategy_profiles.json", optimized_profile_path: str = "optimized_profile.json"):
        self.config, profile_source = self._load_best_config(base_profile_path, optimized_profile_path)
        print(f"🧠 Guardian initialized. Loading profiles from: '{profile_source}'")

        # Determine which profile to activate
        if 'optimized_v1' in self.config.get("profiles", {}):
            self.active_profile_name = 'optimized_v1'
            print("  -> ✅ Auto-Tuned profile 'optimized_v1' detected and activated.")
        else:
            self.active_profile_name = "default"
            print("  -> No optimized profile found. Activating 'default' profile.")

        self.active_profile = self.config.get("profiles", {}).get(self.active_profile_name, {})
        self.currency_families = self.config.get("currency_families", {})

        if mt5 is None:
            raise ImportError("MetaTrader5 is not installed. StrategyCore cannot function.")

    def _load_best_config(self, base_path: str, optimized_path: str) -> Tuple[Dict, str]:
        """Loads the optimized config if it exists, otherwise falls back to the base config."""
        if os.path.exists(optimized_path):
            try:
                with open(optimized_path, 'r') as f:
                    return json.load(f), optimized_path
            except (json.JSONDecodeError) as e:
                print(f"CRITICAL: Optimized profile '{optimized_path}' is invalid. Falling back to base. Error: {e}")
        
        if os.path.exists(base_path):
            try:
                with open(base_path, 'r') as f:
                    return json.load(f), base_path
            except (json.JSONDecodeError) as e:
                print(f"CRITICAL: Base profile '{base_path}' is invalid. Using empty config. Error: {e}")
        
        # Fallback if no files are found
        return {"profiles": {"default": {}}, "currency_families": {}}, "None (Using empty config)"


    def _get_open_positions(self) -> List[Dict[str, Any]]:
        """Fetches and parses currently open positions from the MT5 terminal."""
        if not _ensure_attached_and_visible("EURUSD"): return []
        positions = mt5.positions_get()
        if positions is None: return []
        return [{"symbol": p.symbol, "lot": p.volume, "type": "BUY" if p.type == 0 else "SELL"} for p in positions]

    def _get_symbol_families(self, symbol: str) -> List[str]:
        """Identifies the currency families a symbol belongs to (e.g., EURUSD -> ['EUR', 'USD'])."""
        symbol_base = symbol.upper().replace("C", "").replace("M", "")
        # Handle FX pairs first for correctness
        if len(symbol_base) >= 6 and symbol_base[:3] in self.currency_families.get('FX', []) and symbol_base[3:6] in self.currency_families.get('FX', []):
            return [symbol_base[:3], symbol_base[3:6]]
        # Handle other families
        for family, members in self.currency_families.items():
            if any(member in symbol_base for member in members):
                return [family]
        return ["UNKNOWN"]

    def request_trade_approval(self, symbol: str, direction: str, lot_size: float) -> ApprovalResponse:
        """The main pre-trade check function against the active strategy profile."""
        print(f"🧠 Guardian assessing trade request against '{self.active_profile_name}' profile...")
        
        open_positions = self._get_open_positions()
        
        # Rule 1: Max total trades
        max_trades = self.active_profile.get("max_total_trades", 5)
        if len(open_positions) >= max_trades:
            return ApprovalResponse(False, f"REJECTED: Position limit reached ({len(open_positions)}/{max_trades}).")

        # --- Exposure Calculation ---
        family_exposure = defaultdict(float)
        for pos in open_positions:
            families = self._get_symbol_families(pos["symbol"])
            pos_direction = 1 if pos["type"] == "BUY" else -1
            if len(families) > 1: # Standard FX pair
                family_exposure[families[0]] += pos["lot"] * pos_direction
                family_exposure[families[1]] -= pos["lot"] * pos_direction
            else: # Single-family asset like Index or Metal
                family_exposure[families[0]] += pos["lot"] * pos_direction
        
        # Rule 2: Max family exposure
        new_trade_families = self._get_symbol_families(symbol)
        new_trade_direction = 1 if direction == "BUY" else -1
        max_family_lots = self.active_profile.get("max_family_exposure_lots", 2.0)
        
        # Project exposure for each affected family
        temp_exposure = family_exposure.copy()
        if len(new_trade_families) > 1:
            temp_exposure[new_trade_families[0]] += lot_size * new_trade_direction
            temp_exposure[new_trade_families[1]] -= lot_size * new_trade_direction
        else:
            temp_exposure[new_trade_families[0]] += lot_size * new_trade_direction

        for family, proj_exp in temp_exposure.items():
            if abs(proj_exp) > max_family_lots:
                return ApprovalResponse(False, f"REJECTED: Breaches max exposure for family '{family}'. (Projected: {proj_exp:.2f}, Max: {max_family_lots})")

        # Rule 3: Max total lot exposure
        max_total_lots = self.active_profile.get("max_total_exposure_lots", 4.0)
        current_total_lots = sum(p['lot'] for p in open_positions)
        if (current_total_lots + lot_size) > max_total_lots:
            return ApprovalResponse(False, f"REJECTED: Breaches max total lot exposure. (Current: {current_total_lots:.2f}, With New: {current_total_lots + lot_size:.2f}, Max: {max_total_lots})")
            
        print("✅ Guardian approves trade. All risk parameters within limits.")
        return ApprovalResponse(True, "APPROVED: All risk checks passed.")

if __name__ == "__main__":
    print("Running StrategyCore standalone test...")
    guardian = StrategyCore()
    print(f"Loaded active profile: '{guardian.active_profile_name}' with params: {guardian.active_profile}")
    approval = guardian.request_trade_approval(symbol="EURUSD", direction="BUY", lot_size=0.5)
    print(f"\n--- TEST APPROVAL REQUEST ---\nResponse: {approval}")
