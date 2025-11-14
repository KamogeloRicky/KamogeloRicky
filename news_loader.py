"""
news_loader.py
- Loads news events from CSV into the format expected by context_engine.contextual_analysis.
- CSV must have columns: time (ISO string) and impact (High/Medium/Low).
- Paths: set ANALYSIS_NEWS_CSV or BACKTEST_NEWS_CSV env var to point to your CSV.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
import pandas as pd


def get_news_events() -> List[Dict[str, Any]]:
    path = os.getenv("ANALYSIS_NEWS_CSV", os.getenv("BACKTEST_NEWS_CSV", ""))
    if not path or not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        # normalize columns
        tcol = None
        for cand in ("time", "timestamp", "date"):
            if cand in df.columns:
                tcol = cand; break
        if tcol is None:
            return []
        icol = "impact" if "impact" in df.columns else ("importance" if "importance" in df.columns else None)
        if icol is None:
            df["impact"] = "Medium"
            icol = "impact"
        out: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            t = row[tcol]
            imp = row[icol]
            try:
                ts = pd.to_datetime(str(t))
            except Exception:
                continue
            out.append({"time": ts, "impact": str(imp)})
        return out
    except Exception:
        return []
