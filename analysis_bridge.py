"""
analysis_bridge.py
Use contextual analysis without modifying analysis.py.

Compatibility:
- Works whether analysis.analyze_pair supports `df_override` or not.
- If `df_override` is not supported, it temporarily monkey-patches fetch_candles
  used by analysis/analyze_pair so the call uses your provided DataFrame.

Rescue logic:
- If analyze_pair returns success=False but the bridge has a valid context DataFrame,
  it re-runs analyze_pair using that DF (via monkey-patch) to "rescue" the base result.

Environment toggles:
- ANALYSIS_ENABLE_CONTEXT=1|0
- ANALYSIS_BLEND_CONTEXT=1|0
- ANALYSIS_HTF="4h"
- ANALYSIS_CONTEXT_BARS=2000
- ANALYSIS_NEWS_CSV=/path/news.csv
"""

from __future__ import annotations

import os
import inspect
from contextlib import contextmanager
from typing import Any, Dict, Optional

import pandas as pd

from data import fetch_candles
from analysis import analyze_pair  # do not modify analysis.py
from context_engine import contextual_analysis
from news_loader import get_news_events


@contextmanager
def _patch_fetch_candles_for_analysis(df: pd.DataFrame):
    """
    Monkey-patch fetch_candles in both `analysis` and `data` modules so that
    analyze_pair uses the provided df without requiring a df_override parameter.
    Fully restored afterwards.
    """
    import analysis as _analysis_mod  # module, not function
    import data as _data_mod

    had_analysis_fc = hasattr(_analysis_mod, "fetch_candles")
    orig_analysis_fc = getattr(_analysis_mod, "fetch_candles", None)
    orig_data_fc = getattr(_data_mod, "fetch_candles", None)

    def _fake_fetch(symbol: str, timeframe: str, n: Optional[int] = None, *args, **kwargs):
        return df.copy()

    try:
        if had_analysis_fc:
            setattr(_analysis_mod, "fetch_candles", _fake_fetch)
        if orig_data_fc is not None:
            setattr(_data_mod, "fetch_candles", _fake_fetch)
        yield
    finally:
        if had_analysis_fc:
            setattr(_analysis_mod, "fetch_candles", orig_analysis_fc)
        if orig_data_fc is not None:
            setattr(_data_mod, "fetch_candles", orig_data_fc)


def _call_analyze_pair(pair: str,
                       timeframe: str,
                       df_override: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """
    Call analyze_pair in a way that's compatible with both signatures.
    If df_override is provided but analyze_pair doesn't accept it,
    we patch fetch_candles during the call.
    """
    try:
        sig = inspect.signature(analyze_pair)
        supports_df_override = "df_override" in sig.parameters
    except Exception:
        supports_df_override = False

    if supports_df_override and df_override is not None:
        try:
            return analyze_pair(pair, timeframe, df_override=df_override)
        except Exception:
            return None

    if df_override is not None:
        try:
            with _patch_fetch_candles_for_analysis(df_override):
                return analyze_pair(pair, timeframe)
        except Exception:
            return None

    try:
        return analyze_pair(pair, timeframe)
    except Exception:
        return None


def analyze_with_context(pair: str,
                         timeframe: str,
                         df_override: Optional[pd.DataFrame] = None,
                         n: Optional[int] = None,
                         blend: Optional[bool] = None) -> Dict[str, Any]:
    """
    Wrapper that preserves analyze_pair behavior and adds context as a separate block.
    Never mutates input DataFrame, never modifies analysis.py.
    """
    # 1) Run original analyzer
    base = _call_analyze_pair(pair, timeframe, df_override=None)  # first try without overrides

    # 2) Prepare context DF (prefer supplied override)
    ctx_df = df_override
    if ctx_df is None:
        bars = int(os.getenv("ANALYSIS_CONTEXT_BARS", str(n or 2000)))
        try:
            ctx_df = fetch_candles(pair, timeframe, n=bars)
        except Exception:
            ctx_df = None

    # 3) Rescue path: if base failed but we have a context DF, re-run analyze_pair using that DF
    if (not isinstance(base, dict)) or (not base.get("success")):
        if isinstance(ctx_df, pd.DataFrame) and not ctx_df.empty:
            rescue = _call_analyze_pair(pair, timeframe, df_override=ctx_df)
            if isinstance(rescue, dict) and rescue.get("success"):
                base = rescue  # promote rescued base

    # Base is now either the original, rescued, or None -> normalize to dict
    if not isinstance(base, dict):
        base = {"success": False, "pair": pair, "timeframe": timeframe}

    # 4) Optional context run (can work even if base failed)
    if os.getenv("ANALYSIS_ENABLE_CONTEXT", "1") == "0":
        return {
            "success": bool(base.get("success", True)),
            "base": base,
            "context": None,
            "final_suggestion": base.get("final_suggestion"),
            "confidence": base.get("confidence"),
            "entry_plan": None,
        }

    # Determine price for context (fallback to last close if base has no price)
    price_for_ctx = base.get("price")
    if price_for_ctx is None and isinstance(ctx_df, pd.DataFrame) and "close" in ctx_df.columns and not ctx_df.empty:
        try:
            price_for_ctx = float(pd.to_numeric(ctx_df["close"], errors="coerce").dropna().iloc[-1])
        except Exception:
            price_for_ctx = None

    # 5) Run contextual analysis
    ctx = None
    try:
        if isinstance(ctx_df, pd.DataFrame) and not ctx_df.empty:
            ctx = contextual_analysis(
                ctx_df,
                pair=pair,
                timeframe=timeframe,
                price=price_for_ctx,
                fetcher=fetch_candles,                      # HTF bias sampling
                htf=os.getenv("ANALYSIS_HTF", "4h"),
                news_events=(get_news_events() or [])
            )
    except Exception:
        ctx = None

    # 6) Blend confidence (lightly) if requested
    do_blend = (os.getenv("ANALYSIS_BLEND_CONTEXT", "1") != "0") if blend is None else bool(blend)
    confidence = base.get("confidence")
    if do_blend and isinstance(ctx, dict):
        try:
            base_conf = float(confidence if confidence is not None else 0.5)
            ctx_conf = float(ctx.get("confidence", 0.5))
            confidence = max(0.05, min(0.97, 0.7 * base_conf + 0.3 * ctx_conf))
        except Exception:
            pass

    # 7) Compute wrapper success: true if either base succeeded or we have a valid context
    wrapper_success = bool(base.get("success", False) or (isinstance(ctx, dict) and ctx))

    # 8) Build final wrapper result
    entry_plan = ctx.get("entry_plan") if isinstance(ctx, dict) else None

    return {
        "success": wrapper_success,
        "base": base,
        "context": ctx,
        "final_suggestion": base.get("final_suggestion"),
        "confidence": confidence,
        "entry_plan": entry_plan,
    }


if __name__ == "__main__":
    # Quick smoke test
    pair = os.getenv("BRIDGE_PAIR", "EURUSDm")
    timeframe = os.getenv("BRIDGE_TF", "1h")
    res = analyze_with_context(pair, timeframe)
    print(res)
