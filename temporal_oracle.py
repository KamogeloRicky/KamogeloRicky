"""
temporal_oracle.py
------------------
TEO (Temporal Execution Oracle)

Purpose:
    - Always provide a directional bias (Buy/Sell/Neutral).
    - When PREX says "not safe now", tell:
        * how many candles to wait (approx),
        * approximate time in minutes,
        * structural condition to wait for,
        * invalidation level/condition.
    - When PREX allows trading, still provide the same guidance as meta-data
      (how long the setup is likely valid, etc.).

Integration:
    - Import and call from cli.py after run_prex().
    - Pass in any directional hints from your existing engines as `engines_bias`.
    - Use TEOResult in your final JSON (for GUI) and in quantum_execution.

This module does NOT place trades. It only describes timing and structure.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import statistics


@dataclass
class TEOResult:
    direction_bias: str            # "Buy", "Sell", "Neutral"
    bias_strength: float           # 0.0 - 1.0
    wait_candles: int              # how many bars to wait (estimate)
    estimated_wait_minutes: int    # wait_candles * TF minutes (approx)
    wait_condition: str            # human-readable condition
    invalid_if: str                # human-readable invalidation condition
    pattern: str                   # e.g. "PostShock_VReversal", "NormalPullback"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _tf_to_minutes(tf: str) -> int:
    tf = tf.lower()
    if tf.endswith("m"):
        return int(tf[:-1] or "5")
    if tf.endswith("h"):
        return int(tf[:-1] or "1") * 60
    if tf == "d1" or tf == "1d":
        return 1440
    return 5  # default fallback


def _simple_direction_from_closes(closes: List[float]) -> str:
    if len(closes) < 5:
        return "Neutral"
    # simple slope check
    first = statistics.mean(closes[:3])
    last = statistics.mean(closes[-3:])
    if last > first * 1.001:
        return "Buy"
    if last < first * 0.999:
        return "Sell"
    return "Neutral"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_teo(
    symbol: str,
    timeframe: str,
    prex_regime: str,
    ohlcv: Dict[str, List[float]],
    engines_bias: Optional[str] = None,
) -> TEOResult:
    """
    Main entrypoint.

    Args:
        symbol: e.g. "USDJPYm"
        timeframe: e.g. "5m"
        prex_regime: from PrexVerdict.regime
        ohlcv: dict with lists: {"open","high","low","close","volume"}
        engines_bias: optional bias from your existing engines ("Buy"/"Sell"/"Neutral")

    Returns:
        TEOResult with direction bias + timing + structural guidance.
    """

    closes = ohlcv.get("close", [])
    highs = ohlcv.get("high", [])
    lows = ohlcv.get("low", [])

    # 1. Direction bias
    auto_bias = _simple_direction_from_closes(closes)
    bias = engines_bias if engines_bias in ("Buy", "Sell", "Neutral") else auto_bias

    # If still neutral, default to neutral bias with low strength
    if bias not in ("Buy", "Sell"):
        bias = "Neutral"
        bias_strength = 0.2
    else:
        bias_strength = 0.7

    tf_min = _tf_to_minutes(timeframe)

    # 2. Pattern-based waiting logic
    pattern = "Normal"
    wait_candles = 0
    wait_condition = "Setup is actionable immediately."
    invalid_if = "Bias invalidated if opposite momentum dominates for several bars."

    if prex_regime == "VOLATILITY_SHOCK":
        pattern = "PostShock_VReversal"
        wait_candles = 5
        wait_condition = (
            "Environment is in volatility shock. "
            "Wait for 3–5 smaller candles forming a base after this spike, "
            "with shrinking range and no new extremes."
        )
        invalid_if = (
            "If price makes a new extreme beyond the spike's high/low, "
            "consider this bias invalid and wait for fresh structure."
        )
    elif prex_regime == "LOW_LIQUIDITY":
        pattern = "LowLiquidity"
        wait_candles = 4
        wait_condition = (
            "Spread is abnormally wide. Wait for spread to compress back into its "
            "normal range and at least 3 candles with decent volume before entering."
        )
        invalid_if = (
            "If wide spreads persist beyond 2× the usual time, avoid trading this symbol."
        )
    elif prex_regime == "SESSION_OPEN_RISK":
        pattern = "SessionOpen"
        wait_candles = 2
        wait_condition = (
            "Session opening volatility is elevated. "
            "Wait for the first 2 candles to close, then reassess structure."
        )
        invalid_if = (
            "If after the first two candles the direction violently flips twice, "
            "treat the environment as random and wait for clearer trend."
        )
    else:
        # NORMAL regime – provide mild guidance, not delay
        pattern = "NormalPullback"
        wait_candles = 0
        if bias == "Buy":
            wait_condition = (
                "Look for a modest pullback or small consolidation before entering long; "
                "avoid chasing the exact spike high."
            )
            invalid_if = "If price breaks the most recent swing low with momentum, re-evaluate the bias."
        elif bias == "Sell":
            wait_condition = (
                "Look for a modest retrace or base before entering short; "
                "avoid selling directly into exhaustion."
            )
            invalid_if = "If price breaks the most recent swing high with momentum, re-evaluate the bias."
        else:
            # Truly neutral
            wait_candles = 3
            wait_condition = (
                "Bias is neutral. Wait for 3 candles to establish clearer trend or a range boundary sweep."
            )
            invalid_if = "If no structure appears after 3–5 candles, consider skipping this instrument."

    estimated_wait_minutes = wait_candles * tf_min

    return TEOResult(
        direction_bias=bias,
        bias_strength=bias_strength,
        wait_candles=wait_candles,
        estimated_wait_minutes=estimated_wait_minutes,
        wait_condition=wait_condition,
        invalid_if=invalid_if,
        pattern=pattern,
    )
