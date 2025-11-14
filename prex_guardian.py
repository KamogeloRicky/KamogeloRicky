"""
prex_guardian.py
----------------
PREX (Pre-Execution Guardian)

Purpose:
    - Run BEFORE heavy orchestration.
    - Decide whether it is globally safe to consider trades *right now*.
    - Detect volatility shocks, V-reversals, session/news danger zones.
    - Provide microstructure constraints (tick size, min stop, spread bands).
    - Provide a risk multiplier for downstream sizing (e.g. 0.5 in risky regimes).

Integration:
    - Import and call from cli.py at the very top of run_orchestrated_analysis().
    - Use PrexVerdict.allow to decide whether to proceed with full analysis.
    - Pass PrexVerdict.microstructure and risk_multiplier into the rest of the pipeline.

Notes:
    - This module is conservative but NOT timid:
      It blocks only truly hostile environments (e.g. USDJPYm-style spikes),
      while letting Quantum Execution (quantum_execution.py) handle edge
      optimization within allowed regimes.

Data sources:
    - OHLCV: data.fetch_candles (your existing function).
    - Symbol info: MetaTrader5.symbol_info if available, otherwise robust defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import statistics
from datetime import datetime, timezone

# Try to import your existing candle fetcher
try:
    from data import fetch_candles as _fetch_candles_for_prex  # type: ignore
except Exception:
    _fetch_candles_for_prex = None

# Try MetaTrader5 for symbol info (optional)
try:
    import MetaTrader5 as mt5  # type: ignore
    _MT5_AVAILABLE = True
except Exception:
    mt5 = None  # type: ignore
    _MT5_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MicrostructureProfile:
    entry_price_side_buy_limit: str  # "ask" or "bid"
    entry_price_side_sell_limit: str # "bid" or "ask"
    min_stop_points: float           # min SL distance in raw price units
    spread_normal: float             # typical spread (price units)
    spread_max_safe: float           # maximum spread we consider safe
    tick_size: float                 # symbol point / tick size
    max_leverage_hint: float         # optional hint (not enforced here)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrexVerdict:
    allow: bool
    reason: str
    regime: str                       # e.g. "NORMAL", "VOLATILITY_SHOCK", "LOW_LIQUIDITY"
    risk_multiplier: float
    microstructure: MicrostructureProfile
    cooldown_bars: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["microstructure"] = self.microstructure.to_dict()
        return d


# ---------------------------------------------------------------------------
# Symbol configuration (tunable per-broker)
# ---------------------------------------------------------------------------

DEFAULT_SYMBOL_PROFILES: Dict[str, Dict[str, Any]] = {
    "XAU": {  # Gold
        "min_stop_points_factor": 3.0,     # ATR multiple for min stop
        "max_spread_factor": 6.0,          # ATR multiple for "unsafe" spread
        "risk_multiplier_shock": 0.5,
        "shock_atr_multiple": 6.0,         # bar range > this x ATR = shock
    },
    "JPY": {  # JPY pairs
        "min_stop_points_factor": 2.5,
        "max_spread_factor": 6.0,
        "risk_multiplier_shock": 0.5,
        "shock_atr_multiple": 7.0,
    },
    "BTC": {  # Crypto
        "min_stop_points_factor": 4.0,
        "max_spread_factor": 10.0,
        "risk_multiplier_shock": 0.4,
        "shock_atr_multiple": 8.0,
    },
    "INDEX": {  # Indices like US30m, USTECm
        "min_stop_points_factor": 3.5,
        "max_spread_factor": 8.0,
        "risk_multiplier_shock": 0.5,
        "shock_atr_multiple": 7.0,
    },
    "FX": {  # generic FX
        "min_stop_points_factor": 2.0,
        "max_spread_factor": 5.0,
        "risk_multiplier_shock": 0.6,
        "shock_atr_multiple": 6.0,
    },
}


def classify_symbol_family(symbol: str) -> str:
    s = symbol.upper()
    if "XAU" in s or "GOLD" in s:
        return "XAU"
    if "BTC" in s:
        return "BTC"
    if "US30" in s or "USTEC" in s or "NAS" in s or "DJ" in s:
        return "INDEX"
    if s.endswith("JPY") or "JPY" in s:
        return "JPY"
    return "FX"


# ---------------------------------------------------------------------------
# Helper calculations
# ---------------------------------------------------------------------------

def _compute_atr(ranges: List[float]) -> float:
    """Simple ATR proxy from list of (high-low) ranges."""
    if not ranges:
        return 0.0
    try:
        return statistics.mean(ranges[-20:])
    except statistics.StatisticsError:
        return statistics.mean(ranges)


def _detect_volatility_shock(last_range: float, atr: float, cfg: Dict[str, Any]) -> bool:
    if atr <= 0:
        return False
    shock_multiple = cfg.get("shock_atr_multiple", 6.0)
    return last_range > shock_multiple * atr


def _detect_v_reversal_no_base(
    closes: List[float],
    highs: List[float],
    lows: List[float],
) -> bool:
    """
    Lightweight spike + V-reversal + no base heuristic.

    We assume:
      - last two bars are opposite direction and very large,
      - previous 3 bars are also large, meaning no 'calm base'.
    """
    if len(closes) < 5:
        return False

    c0, c1 = closes[-1], closes[-2]
    o0, o1 = closes[-2], closes[-3]  # approx opens
    r0 = abs(highs[-1] - lows[-1])
    r1 = abs(highs[-2] - lows[-2])

    dir0 = 1 if c0 > o0 else -1
    dir1 = 1 if c1 > o1 else -1
    if dir0 * dir1 >= 0:
        return False

    ranges = [h - l for h, l in zip(highs, lows)]
    atr = _compute_atr(ranges)
    if atr <= 0:
        return False

    if r0 > 4.5 * atr and r1 > 4.5 * atr:
        prev_ranges = ranges[-5:-2]
        if all(r > 2.5 * atr for r in prev_ranges):
            return True

    return False


def _is_risky_session_open(symbol: str, now_utc: datetime) -> bool:
    """
    Rough heuristic: avoid first minutes of major sessions for sensitive symbols.
    """

    hour = now_utc.hour
    family = classify_symbol_family(symbol)

    # Asia open (dangerous for JPY & XAU)
    if family in ("JPY", "XAU", "FX"):
        if 23 <= hour or hour < 1:
            return True

    # London open
    if 7 <= hour < 8:
        return True

    # NY open
    if 12 <= hour < 13:
        return True

    return False


# ---------------------------------------------------------------------------
# Data access implementations (NO stubs)
# ---------------------------------------------------------------------------

def fetch_ohlcv(symbol: str, timeframe: str, bars: int = 200) -> Dict[str, List[float]]:
    """
    Concrete implementation for PREX: proxy to data.fetch_candles.

    Assumes:
        - data.fetch_candles(symbol, timeframe, n) returns a pandas DataFrame
          with columns ['open', 'high', 'low', 'close'] and optionally 'volume'.
    """
    if _fetch_candles_for_prex is None:
        raise RuntimeError("data.fetch_candles is not available for PREX.")

    df = _fetch_candles_for_prex(symbol, timeframe, n=bars)
    if df is None or df.empty:
        raise RuntimeError(f"fetch_candles returned no data for {symbol} {timeframe}")

    return {
        "open": df["open"].tolist(),
        "high": df["high"].tolist(),
        "low": df["low"].tolist(),
        "close": df["close"].tolist(),
        "volume": (df["volume"] if "volume" in df.columns else df["close"] * 0).tolist(),
    }


def fetch_symbol_info(symbol: str) -> Dict[str, Any]:
    """
    Concrete implementation for PREX.
    Tries MetaTrader5 first; if unavailable or symbol not found, falls back to reasonable defaults.
    """
    # 1) Try MetaTrader5 if available
    if _MT5_AVAILABLE:
        try:
            if not mt5.initialize():
                raise RuntimeError("MetaTrader5 initialization failed for PREX symbol info.")

            info = mt5.symbol_info(symbol)
            if info is None:
                raise RuntimeError(f"MetaTrader5.symbol_info returned None for {symbol}")

            tick_size = float(info.point)
            # trade_stops_level is in "points"; convert to price units
            min_stop_distance_points = float(info.trade_stops_level * info.point)
            if min_stop_distance_points <= 0:
                min_stop_distance_points = tick_size * 5.0

            typical_spread = float(info.spread * info.point) if info.spread is not None else tick_size * 5.0
            if typical_spread <= 0:
                typical_spread = tick_size * 2.0

            # If you have access to current Bid/Ask, you can refine current_spread; otherwise use typical_spread
            current_spread = typical_spread

            return {
                "tick_size": tick_size,
                "min_stop_distance_points": min_stop_distance_points,
                "typical_spread": typical_spread,
                "max_leverage_hint": 100.0,
                "current_spread": current_spread,
            }
        except Exception:
            # fall through to robust defaults
            pass

    # 2) Robust defaults by symbol family
    family = classify_symbol_family(symbol)

    if family == "BTC":
        tick = 1.0
        return {
            "tick_size": tick,
            "min_stop_distance_points": 50.0,
            "typical_spread": 20.0,
            "max_leverage_hint": 10.0,
            "current_spread": 20.0,
        }
    if family in ("XAU", "INDEX"):
        tick = 0.1
        return {
            "tick_size": tick,
            "min_stop_distance_points": 1.0,
            "typical_spread": 0.5,
            "max_leverage_hint": 50.0,
            "current_spread": 0.5,
        }
    # generic FX
    tick = 0.0001
    return {
        "tick_size": tick,
        "min_stop_distance_points": 0.0005,
        "typical_spread": 0.0002,
        "max_leverage_hint": 100.0,
        "current_spread": 0.0002,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_prex(symbol: str, timeframe: str, balance: float) -> PrexVerdict:
    """
    Main entrypoint.

    Args:
        symbol: e.g. "XAUUSDm"
        timeframe: e.g. "5m", "1m", "15m"
        balance: account balance (optional, not strongly used here)

    Returns:
        PrexVerdict:
            - allow: whether to proceed with full analysis & potential execution.
            - reason/regime: textual explanation and environment classification.
            - risk_multiplier: factor for downstream position sizing.
            - microstructure: constraints for order/price construction.
    """
    family = classify_symbol_family(symbol)
    family_cfg = DEFAULT_SYMBOL_PROFILES.get(family, DEFAULT_SYMBOL_PROFILES["FX"])

    # Fetch data
    ohlcv = fetch_ohlcv(symbol, timeframe, bars=200)
    highs = ohlcv["high"]
    lows = ohlcv["low"]
    closes = ohlcv["close"]

    symbol_info = fetch_symbol_info(symbol)

    # Basic microstructure
    tick = float(symbol_info["tick_size"])
    typical_spread = float(symbol_info["typical_spread"])
    min_stop_raw = float(symbol_info["min_stop_distance_points"])

    ranges = [h - l for h, l in zip(highs, lows)]
    atr = _compute_atr(ranges) if ranges else 0.0
    last_range = ranges[-1] if ranges else 0.0

    # Derive dynamic min stop & spread caps
    dyn_min_stop = max(
        min_stop_raw,
        family_cfg["min_stop_points_factor"] * atr if atr > 0 else min_stop_raw,
    )
    spread_max_safe = max(
        typical_spread * 3.0,
        family_cfg["max_spread_factor"] * atr if atr > 0 else typical_spread * 3.0,
    )

    micro = MicrostructureProfile(
        entry_price_side_buy_limit="ask",
        entry_price_side_sell_limit="bid",
        min_stop_points=dyn_min_stop,
        spread_normal=typical_spread,
        spread_max_safe=spread_max_safe,
        tick_size=tick,
        max_leverage_hint=float(symbol_info.get("max_leverage_hint", 100.0)),
    )

    # Environment classification
    regime = "NORMAL"
    reason = "Normal regime"
    risk_mult = 1.0
    cooldown_bars = 0

    now_utc = datetime.now(timezone.utc)

    # Spread / liquidity check
    current_spread = float(symbol_info.get("current_spread", typical_spread))
    if current_spread > micro.spread_max_safe:
        regime = "LOW_LIQUIDITY"
        reason = f"Spread too wide: {current_spread:.6f} > safe {micro.spread_max_safe:.6f}"
        risk_mult = 0.4
        cooldown_bars = 2
        return PrexVerdict(
            allow=False,
            reason=reason,
            regime=regime,
            risk_multiplier=risk_mult,
            microstructure=micro,
            cooldown_bars=cooldown_bars,
        )

    # Volatility shock detection
    is_shock = _detect_volatility_shock(last_range, atr, family_cfg)
    is_vrev = _detect_v_reversal_no_base(closes, highs, lows)

    if is_shock or is_vrev:
        regime = "VOLATILITY_SHOCK"
        reason = "Volatility shock / V-reversal with no base"
        risk_mult = family_cfg["risk_multiplier_shock"]
        cooldown_bars = 4
        return PrexVerdict(
            allow=False,
            reason=reason,
            regime=regime,
            risk_multiplier=risk_mult,
            microstructure=micro,
            cooldown_bars=cooldown_bars,
        )

    # Risky session openings
    if _is_risky_session_open(symbol, now_utc):
        regime = "SESSION_OPEN_RISK"
        reason = "Session open window (potentially unstable)"
        risk_mult = 0.7
        cooldown_bars = 1
        # We still allow, but with reduced risk
        return PrexVerdict(
            allow=True,
            reason=reason,
            regime=regime,
            risk_multiplier=risk_mult,
            microstructure=micro,
            cooldown_bars=cooldown_bars,
        )

    # Default: allow
    return PrexVerdict(
        allow=True,
        reason=reason,
        regime=regime,
        risk_multiplier=risk_mult,
        microstructure=micro,
        cooldown_bars=cooldown_bars,
    )
