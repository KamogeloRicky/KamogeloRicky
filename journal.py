"""
The Journal ("The Scribe")

Purpose:
- Centralizes all logging operations for the trading bot.
- Provides a single, reliable interface for writing trade analysis and decisions to CSV files.
- Decouples the core analysis logic from the data storage implementation, making the system cleaner.
- Ensures consistent data schemas for all log files.
"""

import os
import csv
from datetime import datetime
from typing import Dict, Any

# Define the schemas for our log files to ensure consistency
# This schema matches the final, most detailed GOAT+ pass in your analysis.py
GOAT_PLUS_SCHEMA = [
    "timestamp_sast", "pair", "timeframe", "label_in", "label_out",
    "fresh_ok", "age_min", "spread_to_atr", "hurst", "entropy", "trend_slope", "trend_persist",
    "congestion", "clusters", "sr_min_dist_r", "dd", "dd_state", "exp_family", "exp_open", "blocked_reasons"
]

def _get_sast_timestamp() -> str:
    """Returns the current UTC time formatted as a SAST string."""
    try:
        from analysis import _to_sast
        return _to_sast(datetime.utcnow())
    except Exception:
        return datetime.utcnow().isoformat() + " (UTC)"

def log_trade_decision(analysis_result: Dict[str, Any]):
    """
    Logs the result of a GOAT+ analysis pass to the main journal CSV.
    This is the primary logging function for post-analysis results.
    """
    if not isinstance(analysis_result, dict):
        return

    goat_plus_data = analysis_result.get("goat_plus")
    if not isinstance(goat_plus_data, dict):
        return # Don't log if there's no GOAT+ pass data

    log_file = os.getenv("ANALYSIS_HISTORY_GOATPLUS_CSV", "analysis_history_goat_plus.csv")
    
    try:
        # Prepare the data row according to the schema
        row_data = {
            "timestamp_sast": _get_sast_timestamp(),
            "pair": analysis_result.get("pair"),
            "timeframe": analysis_result.get("timeframe"),
            "label_in": goat_plus_data.get("incoming_label"),
            "label_out": goat_plus_data.get("decision_label"),
            "blocked_reasons": "|".join(goat_plus_data.get("blocked_reasons", [])),
            # Unpack nested dictionaries for flat CSV structure
            "fresh_ok": (goat_plus_data.get("freshness") or {}).get("ok"),
            "age_min": (goat_plus_data.get("freshness") or {}).get("age_min"),
            "spread_to_atr": (goat_plus_data.get("spread_to_atr") or {}).get("ratio"),
            "hurst": goat_plus_data.get("hurst"),
            "entropy": goat_plus_data.get("entropy"),
            "trend_slope": (goat_plus_data.get("trend_health") or {}).get("slope"),
            "trend_persist": (goat_plus_data.get("trend_health") or {}).get("persistence"),
            "congestion": (goat_plus_data.get("sr_congestion") or {}).get("congestion"),
            "clusters": (goat_plus_data.get("sr_congestion") or {}).get("clusters"),
            "sr_min_dist_r": goat_plus_data.get("sr_min_dist_r"),
            "dd": (goat_plus_data.get("dd_throttle") or {}).get("dd"),
            "dd_state": (goat_plus_data.get("dd_throttle") or {}).get("state"),
            "exp_family": (goat_plus_data.get("exposure_guard") or {}).get("family"),
            "exp_open": (goat_plus_data.get("exposure_guard") or {}).get("open_risk")
        }

        file_exists = os.path.exists(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=GOAT_PLUS_SCHEMA)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row_data.get(k, "") for k in GOAT_PLUS_SCHEMA})

    except Exception as e:
        print(f"JOURNAL ERROR: Failed to write to log file '{log_file}'. Reason: {e}")
