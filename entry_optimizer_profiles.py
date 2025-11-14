"""
Entry Optimizer Profiles — per-instrument tuning presets.
Import this to adjust optimizer parameters dynamically by symbol family.

Example usage:
    from entry_optimizer_profiles import get_profile
    profile = get_profile("STOXX50m")
    # profile dict can be passed to an extended optimize_entry() signature if you add it.

Currently provides suggested defaults (documented only). The core optimizer reads env variables
for now; wire these profiles in if you want symbol-specific parameters at runtime.
"""

from typing import Dict

def get_profile(symbol: str) -> Dict:
    s = symbol.upper()
    # Defaults
    prof = dict(
        bars=1200,
        atr_period=14,
        rr_target=1.8,
        atr_sl_buffer=0.6,
        min_score=0.62,
        allow_breakout=False,
    )
    if any(k in s for k in ["DAX", "GER30", "DE40", "STOXX", "EU50"]):
        prof.update(rr_target=2.0, atr_sl_buffer=0.7, min_score=0.65)
    if any(k in s for k in ["JP225", "NIK", "NKY"]):
        prof.update(rr_target=2.0, atr_sl_buffer=0.7, min_score=0.65)
    if any(k in s for k in ["US30", "DJI", "WS30"]):
        prof.update(rr_target=2.0, atr_sl_buffer=0.7, min_score=0.65)
    if any(k in s for k in ["NAS", "US100", "NDX"]):
        prof.update(rr_target=1.9, atr_sl_buffer=0.6, min_score=0.62)
    if any(k in s for k in ["SPX", "US500", "SP500", "ES"]):
        prof.update(rr_target=1.9, atr_sl_buffer=0.6, min_score=0.62)
    if any(k in s for k in ["XAU", "GOLD"]):
        prof.update(rr_target=1.8, atr_sl_buffer=0.6, min_score=0.60)
    if any(k in s for k in ["XAG", "SILVER"]):
        prof.update(rr_target=1.8, atr_sl_buffer=0.6, min_score=0.60)
    if any(k in s for k in ["BTC", "ETH", "SOL", "XRP"]):
        prof.update(rr_target=1.8, atr_sl_buffer=0.7, min_score=0.66)
    return prof
