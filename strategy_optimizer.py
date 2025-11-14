"""
Strategy Optimizer (signature-agnostic, walk-forward CV, Monte Carlo, parallel hyperopt)

Compatibility:
- Works whether analysis.analyze_pair supports `df_override` or not.
- If it doesn't, we temporarily patch fetch_candles in analysis/data to feed the partial DataFrame.

Other features:
- Walk-forward cross-validation
- Monte Carlo bootstrap robustness
- Multi-objective fitness for hyperopt
"""

from __future__ import annotations

import os
import json
import math
import random
import statistics
import multiprocessing as mp
import datetime as dt
import inspect
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data import fetch_candles
from analysis import analyze_pair  # unchanged analysis.py

OPTIMIZATION_LOG = os.getenv("OPTIMIZATION_LOG", "optimizer_log.json")
RANDOM_SEED = int(os.getenv("OPTIMIZER_SEED", "42"))
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ------------- Compatibility layer for analyze_pair -------------
def _supports_df_override() -> bool:
    try:
        sig = inspect.signature(analyze_pair)
        return "df_override" in sig.parameters
    except Exception:
        return False


@contextmanager
def _patch_fetch_candles_for_analysis(df: pd.DataFrame):
    import analysis as _analysis_mod
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


def _call_analyzer(pair: str, timeframe: str, df_override: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    if _supports_df_override() and df_override is not None:
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


# ------------------------- Helpers -------------------------
def _bars_per_day(timeframe: str) -> Optional[int]:
    tf = str(timeframe).lower().strip()
    mapping = {"1m":1440,"5m":288,"15m":96,"30m":48,"1h":24,"2h":12,"4h":6,"1d":1}
    val = mapping.get(tf)
    return val if val and val > 0 else None


def _calc_lookback_bars(timeframe: str, lookback_days: int, hard_cap: int = 5000) -> int:
    bpd = _bars_per_day(timeframe)
    if bpd:
        est = bpd * max(1, int(lookback_days))
        return max(160, min(hard_cap, int(est)))
    return min(hard_cap, max(160, lookback_days * 100))


def _safe_fetch_candles(pair: str, timeframe: str, n: int) -> Optional[pd.DataFrame]:
    for _ in range(2):
        df = fetch_candles(pair, timeframe, n=n)
        if isinstance(df, pd.DataFrame) and not df.empty:
            for c in ("open", "high", "low", "close"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"])
            if not df.empty:
                return df
    return None


def _atomic_write_json(path: str, data: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)


def _load_json_array(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path): return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception:
        try: os.replace(path, f"{path}.bak")
        except Exception: pass
        return []


def save_optimization_result(result: Dict[str, Any]) -> None:
    data = _load_json_array(OPTIMIZATION_LOG)
    data.append(result)
    if len(data) > 5000:
        data = data[-5000:]
    try:
        _atomic_write_json(OPTIMIZATION_LOG, data)
    except Exception:
        with open(OPTIMIZATION_LOG, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


# ------------------------- Core evaluators -------------------------
def _simulate_tp_sl_outcome(df: pd.DataFrame, i: int, suggestion: str, tp: float, sl: float, horizon: int = 20) -> str:
    next_moves = df.iloc[i:i + horizon]
    if len(next_moves) < 5: return "neutral"
    if suggestion == "Buy":
        hit_tp = bool(next_moves["high"].max() >= tp)
        hit_sl = bool(next_moves["low"].min() <= sl)
    else:
        hit_tp = bool(next_moves["low"].min() <= tp)
        hit_sl = bool(next_moves["high"].max() >= sl)
    if hit_tp and not hit_sl: return "win"
    if hit_sl and not hit_tp: return "loss"
    return "neutral"


def _walk_forward_once(pair: str, timeframe: str, df: pd.DataFrame, step: int = 15, warmup: int = 120, horizon: int = 20) -> Tuple[int,int,int]:
    wins = losses = 0
    for i in range(warmup, len(df) - 1, step):
        sample_df = df.iloc[:i].copy()
        r = _call_analyzer(pair, timeframe, df_override=sample_df)
        if not isinstance(r, dict) or not r.get("success"):
            continue
        sugg = str(r.get("final_suggestion") or "Neutral")
        tp = r.get("tp"); sl = r.get("sl")
        try:
            tp = float(tp) if tp is not None else None
            sl = float(sl) if sl is not None else None
        except Exception:
            tp = sl = None
        if sugg not in ("Buy", "Sell") or tp is None or sl is None or not math.isfinite(tp) or not math.isfinite(sl):
            continue
        outcome = _simulate_tp_sl_outcome(df, i, sugg, tp, sl, horizon)
        if outcome == "win": wins += 1
        elif outcome == "loss": losses += 1
    return wins, losses, (wins + losses)


def _walk_forward_folds(pair: str, timeframe: str, df: pd.DataFrame, folds: int = 3) -> Dict[str, Any]:
    L = len(df)
    if L < 600 or folds < 2:
        w,l,t = _walk_forward_once(pair, timeframe, df)
        acc = (w / t * 100.0) if t > 0 else 0.0
        return {"folds": 1, "accuracies": [acc], "wins": w, "losses": l, "trades": t}
    seg = np.array_split(np.arange(L), folds)
    accs = []; W=L_=T=0
    for idxs in seg:
        sub = df.iloc[idxs[0]: idxs[-1]+1]
        w,l,t = _walk_forward_once(pair, timeframe, sub)
        accs.append((w / t * 100.0) if t > 0 else 0.0)
        W += w; L_ += l; T += t
    return {"folds": folds, "accuracies": accs, "wins": W, "losses": L_, "trades": T}


def _mc_bootstrap(trade_outcomes: List[int], r_multiple_win: float = 1.0, r_multiple_loss: float = -1.0, paths: int = 300) -> Dict[str, Any]:
    if not trade_outcomes:
        return {"paths": 0, "mean_R": 0.0, "p05_R": 0.0, "p95_R": 0.0, "risk_of_ruin": None}
    arr = np.array(trade_outcomes, dtype=int)
    N = len(arr)
    totals = []
    for _ in range(paths):
        samp = np.random.choice(arr, size=N, replace=True)
        R = np.where(samp == 1, r_multiple_win, r_multiple_loss).sum()
        totals.append(R)
    totals = np.array(totals, dtype=float)
    mean_R = float(np.mean(totals))
    p05 = float(np.percentile(totals, 5))
    p95 = float(np.percentile(totals, 95))
    ruin = float(np.mean(totals < (-0.5 * N)))
    return {"paths": paths, "mean_R": mean_R, "p05_R": p05, "p95_R": p95, "risk_of_ruin": ruin}


# ------------------------- Public runners -------------------------
def run_backtest(pair: str, timeframe: str, lookback_days: int = 30) -> Dict[str, Any]:
    print(f"🔁 Running backtest for {pair} ({timeframe})...")
    n_bars = _calc_lookback_bars(timeframe, lookback_days, hard_cap=5000)
    df = _safe_fetch_candles(pair, timeframe, n=n_bars)
    if df is None or df.empty or len(df) < 160:
        err = "No or insufficient data"
        print(f"⚠️ {err}")
        return {"success": False, "error": err, "pair": pair, "timeframe": timeframe}

    stats = _walk_forward_folds(pair, timeframe, df, folds=3)
    wins, losses, trades = stats["wins"], stats["losses"], stats["trades"]
    accuracy = (wins / trades * 100.0) if trades > 0 else 0.0

    outcomes = [1]*wins + [0]*losses
    mc = _mc_bootstrap(outcomes, r_multiple_win=1.0, r_multiple_loss=-1.0, paths=300)

    results = {
        "success": True,
        "pair": pair,
        "timeframe": timeframe,
        "wins": int(wins),
        "losses": int(losses),
        "accuracy": float(round(accuracy, 2)),
        "total_trades": int(trades),
        "folds": int(stats["folds"]),
        "fold_accuracies": [float(round(a, 2)) for a in stats["accuracies"]],
        "mc": mc,
        "timestamp": dt.datetime.utcnow().isoformat(),
    }
    print(f"📈 Backtest {pair} {timeframe}: {results['accuracy']}% (trades={trades}) folds={results['folds']} MC meanR={round(mc['mean_R'],2)} p05={round(mc['p05_R'],2)}")
    save_optimization_result(results)
    return results


def run_forward_test(pair: str, timeframe: str, recent_bars: int = 200) -> Dict[str, Any]:
    n = max(120, int(recent_bars))
    df = _safe_fetch_candles(pair, timeframe, n=n)
    if df is None or df.empty or len(df) < 60:
        err = "No or insufficient data"
        print(f"⚠️ {err}")
        return {"success": False, "error": err, "pair": pair, "timeframe": timeframe}

    print(f"⏩ Running forward test on {pair} ({timeframe})...")
    suggestions: List[str] = []
    for i in range(50, len(df), 10):
        partial = df.iloc[:i].copy()
        r = _call_analyzer(pair, timeframe, df_override=partial)
        if isinstance(r, dict) and r.get("success"):
            suggestions.append(str(r.get("final_suggestion") or "Neutral"))
    total = len(suggestions)
    buy_ratio = (suggestions.count("Buy") / total) if total else 0.0
    sell_ratio = (suggestions.count("Sell") / total) if total else 0.0
    neutral_ratio = (suggestions.count("Neutral") / total) if total else 0.0

    forward_stats = {
        "success": True,
        "pair": pair,
        "timeframe": timeframe,
        "buy_ratio": float(round(buy_ratio * 100.0, 2)),
        "sell_ratio": float(round(sell_ratio * 100.0, 2)),
        "neutral_ratio": float(round(neutral_ratio * 100.0, 2)),
        "total_samples": total,
        "timestamp": dt.datetime.utcnow().isoformat(),
    }
    print(f"🔮 Forward summary: {forward_stats}")
    save_optimization_result(forward_stats)
    return forward_stats


# ------------------------- Hyperopt (parallel) -------------------------
def _trial_worker(args) -> Dict[str, Any]:
    pair, timeframe, env, lookback_days = args
    try:
        # Set overrides for this worker
        old = {k: os.getenv(k) for k in env}
        for k, v in env.items():
            os.environ[k] = str(v)
        try:
            res = run_backtest(pair, timeframe, lookback_days=lookback_days)
        finally:
            # restore env
            for k, v in old.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v
        if not res.get("success"):
            return {"ok": False}
        return {"ok": True, "env": env, "accuracy": res.get("accuracy", 0.0), "trades": res.get("total_trades", 0), "mc": res.get("mc", {})}
    except Exception:
        return {"ok": False}


def optimize_hyperparameters(pair: str,
                             timeframe: str,
                             trials: int = 10,
                             overrides: Optional[Dict[str, Sequence[Any]]] = None,
                             lookback_days: int = 30,
                             parallel: bool = True) -> Dict[str, Any]:
    if not overrides:
        print("No overrides provided; skipping optimization.")
        return {"success": False, "error": "no_overrides"}

    keys = list(overrides.keys())
    grid: List[Dict[str, str]] = []
    seen = set()
    while len(grid) < trials:
        c = {k: str(random.choice(list(overrides[k]))) for k in keys}
        key_tuple = tuple(sorted(c.items()))
        if key_tuple in seen: continue
        seen.add(key_tuple); grid.append(c)

    jobs = [(pair, timeframe, env, lookback_days) for env in grid]
    results: List[Dict[str, Any]] = []

    if parallel:
        with mp.Pool(processes=min(max(1, mp.cpu_count()-1), len(jobs))) as pool:
            results = pool.map(_trial_worker, jobs)
    else:
        results = [_trial_worker(j) for j in jobs]

    ok_res = [r for r in results if r.get("ok")]
    if not ok_res:
        return {"success": False, "error": "no_valid_trials"}

    def score_fun(r: Dict[str, Any]) -> float:
        acc = float(r.get("accuracy", 0.0))
        trades = int(r.get("trades", 0))
        ruin = float(r.get("mc", {}).get("risk_of_ruin") or 0.0)
        return acc + 0.02*trades - 10.0*ruin

    ok_res.sort(key=score_fun, reverse=True)
    best_env = ok_res[0]["env"]
    best = {
        "success": True,
        "pair": pair,
        "timeframe": timeframe,
        "best_env": best_env,
        "best_accuracy": float(ok_res[0].get("accuracy", 0.0)),
        "best_trades": int(ok_res[0].get("trades", 0)),
        "trial_count": len(ok_res),
        "timestamp": dt.datetime.utcnow().isoformat(),
    }
    save_optimization_result({"type": "hyperopt", **best})
    print("🏆 Best hyperparameters:", best)
    return best


# ------------------------- Universe Benchmark -------------------------
def benchmark_universe(pairs: Sequence[str],
                       timeframes: Sequence[str],
                       lookback_days: int = 30) -> List[Dict[str, Any]]:
    table: List[Dict[str, Any]] = []
    for p in pairs:
        for tf in timeframes:
            res = run_backtest(p, tf, lookback_days=lookback_days)
            if res.get("success"): table.append(res)
    if table:
        accs = [float(x["accuracy"]) for x in table]
        mean_acc = statistics.mean(accs)
        std_acc = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        print(f"🏁 Universe mean accuracy: {round(mean_acc,2)}% (σ={round(std_acc,2)}) over {len(table)} runs")
    table.sort(key=lambda x: (x.get("accuracy", 0.0), x.get("total_trades", 0)), reverse=True)
    save_optimization_result({"type": "benchmark", "results": table, "timestamp": dt.datetime.utcnow().isoformat()})
    return table


# ------------------------- Auto Optimize Hook -------------------------
def auto_optimize_parameters() -> None:
    data = _load_json_array(OPTIMIZATION_LOG)
    if not data:
        print("⚙️ No optimization history yet."); return
    acc = [float(x.get("accuracy", 0.0)) for x in data[-10:] if isinstance(x, dict) and "accuracy" in x]
    avg_accuracy = float(np.mean(acc)) if acc else 0.0
    print(f"📊 Avg recent backtest accuracy (last {len(acc)}): {round(avg_accuracy, 2)}%")
    if avg_accuracy < 65.0:
        print("⚙️ Accuracy low → Adjust ATR/EMA multipliers, confluence gates, or session filters.")
    else:
        print("✅ Strategy stable; no optimization required.")


# ------------------------- CLI -------------------------
if __name__ == "__main__":
    pair = os.getenv("OPT_PAIR", "EURUSDm")
    timeframe = os.getenv("OPT_TIMEFRAME", "1h")

    print(f"== Strategy Optimizer ==\nPair: {pair} | TF: {timeframe}")
    try:
        res_bt = run_backtest(pair, timeframe, lookback_days=int(os.getenv("OPT_LOOKBACK_DAYS", "30")))
        res_ft = run_forward_test(pair, timeframe, recent_bars=int(os.getenv("OPT_RECENT_BARS", "200")))
        if os.getenv("OPT_HYPEROPT", "0") == "1":
            overrides = {
                "ANALYSIS_PARTIAL_TP_R": ["0.8", "1.0", "1.2"],
                "ANALYSIS_TRAIL_AFTER_R": ["1.2", "1.5", "2.0"],
                "ANALYSIS_TRAIL_ATR_MULT": ["0.8", "1.0", "1.2"],
                "ANALYSIS_USE_4H": ["0", "1"],
            }
            optimize_hyperparameters(
                pair, timeframe,
                trials=int(os.getenv("OPT_TRIALS","8")),
                overrides=overrides,
                lookback_days=int(os.getenv("OPT_LOOKBACK_DAYS","30")),
                parallel=True
            )
        auto_optimize_parameters()
    except Exception as e:
        print("Optimizer runtime error:", e)
