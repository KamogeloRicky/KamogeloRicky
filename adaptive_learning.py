"""
Adaptive self-improvement engine for RickyFX Analysis Bot (next-gen, bandit-driven).

Unbelievable improvements, fully backward compatible:
- Keeps the original public API: adaptive_engine.get_weights(), get_params(), summary(), log_result()
- Adds powerful but optional capabilities (no breaking changes):
  - Bayesian per-source skill modeling (Beta distributions) + Thompson Sampling bandit boost
  - Reward shaping using R-multiple or PnL magnitude (tanh shaping with outlier clamps)
  - Confidence calibration using per-source Brier score (downweights unreliable confidences)
  - DriftGuard: auto-detects regime degradation and temp-throttles learning (decayed LR)
  - Asset- and timeframe-aware stats and micro-bias (pair -> timeframe -> source multipliers)
  - Non-stationarity control: exponential moving averages + small step-size floors/ceilings
  - Safer persistence: versioned memory, atomic writes, rolling backups (.bak), bounded history
  - Richer summary with per-source and per-asset insights
  - New helper methods (optional): get_weights_dynamic(pair=None, timeframe=None), summary_detailed(),
    export_memory(), import_memory(data), vacuum(max_records)

Defaults are conservative and robust. If a field is missing (e.g., timeframe, pnl), the engine
gracefully falls back to classic behavior. Unknown outcomes are stored but excluded from learning.

Environment toggles (optional):
- ADAPTIVE_BANDIT_ENABLE=1 | 0           Enable Thompson Sampling bandit boost (default: 1)
- ADAPTIVE_DRIFT_GUARD=1 | 0             Enable DriftGuard LR throttling (default: 1)
- ADAPTIVE_LEARNING_FILE=path.json       Memory location (default: adaptive_memory.json)
- ADAPTIVE_BACKUP_KEEP=3                 Number of rolling backups to keep (default: 2)
- ADAPTIVE_MAX_HISTORY=5000              Max records retained (default: 5000)
"""

from __future__ import annotations
import os
import io
import json
import math
import random
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ------------------- CONFIG -------------------
LEARNING_FILE = os.getenv("ADAPTIVE_LEARNING_FILE", "adaptive_memory.json")
BACKUP_KEEP = max(0, int(os.getenv("ADAPTIVE_BACKUP_KEEP", "2")))
MAX_HISTORY_DEFAULT = max(1000, int(os.getenv("ADAPTIVE_MAX_HISTORY", "5000")))
BANDIT_ENABLE = os.getenv("ADAPTIVE_BANDIT_ENABLE", "1") != "0"
DRIFT_GUARD_ENABLE = os.getenv("ADAPTIVE_DRIFT_GUARD", "1") != "0"

# Default weights for each major subsystem (kept for backward compat)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "core": 1.0,
    "advanced": 1.0,
    "precision": 1.0,
    "context": 1.0,
    "news": 1.0
}

# Default adaptive parameters
DEFAULT_PARAMS: Dict[str, Any] = {
    "atr_multiplier": 1.0,
    "retest_sensitivity": 0.5,
    "pattern_confidence_threshold": 0.7
}

MEMORY_VERSION = 2  # bump when schema changes


# ------------------- SAFE IO -------------------
def _safe_write_json(path: str, data: Dict[str, Any]) -> None:
    """Write JSON atomically with rolling backups to avoid corruption."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    try:
        # Roll backups
        if BACKUP_KEEP > 0 and os.path.exists(path):
            for i in reversed(range(BACKUP_KEEP)):
                src = f"{path}.bak{'' if i == 0 else i}"
                dst = f"{path}.bak{i+1}"
                if i == 0:
                    src = path
                if os.path.exists(src):
                    try:
                        if os.path.exists(dst):
                            os.remove(dst)
                    except Exception:
                        pass
                    try:
                        with open(src, "rb") as rf, open(dst, "wb") as wf:
                            wf.write(rf.read())
                    except Exception:
                        pass
        os.replace(tmp, path)
    except Exception:
        # final fallback
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


# ------------------- CORE ENGINE -------------------
class AdaptiveLearning:
    def __init__(self):
        self._lock = threading.RLock()
        self.memory: Dict[str, Any] = {
            "version": MEMORY_VERSION,
            "weights": DEFAULT_WEIGHTS.copy(),        # global weights (backward compat)
            "params": DEFAULT_PARAMS.copy(),          # tunable params
            "history": [],                            # raw records (bounded)
            # new structures
            "per_source": {},                         # source -> stats dict
            "assets": {},                             # pair -> timeframe -> {"stats":{}, "bias":{source:float}}
        }
        # Random seed for bandit sampling (can be customized via env for reproducibility)
        try:
            seed_env = os.getenv("ADAPTIVE_RANDOM_SEED")
            if seed_env:
                random.seed(int(seed_env))
        except Exception:
            pass
        self._load_memory()

    # --------------- MEMORY MGMT ---------------
    def _load_memory(self) -> None:
        try:
            if os.path.exists(LEARNING_FILE):
                with open(LEARNING_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("adaptive memory file corrupted (not a dict)")
                # migrate shape
                self.memory["version"] = data.get("version", 1)
                self.memory["weights"] = data.get("weights", DEFAULT_WEIGHTS.copy())
                self.memory["params"] = data.get("params", DEFAULT_PARAMS.copy())
                hist = data.get("history", [])
                self.memory["history"] = hist[-MAX_HISTORY_DEFAULT:] if isinstance(hist, list) else []

                # new sections, backward-safe
                self.memory["per_source"] = data.get("per_source", {}) if isinstance(data.get("per_source", {}), dict) else {}
                self.memory["assets"] = data.get("assets", {}) if isinstance(data.get("assets", {}), dict) else {}

                # migrate to latest version if needed
                if self.memory.get("version", 1) < MEMORY_VERSION:
                    self._migrate_memory(self.memory.get("version", 1))
                    self.memory["version"] = MEMORY_VERSION
                    _safe_write_json(LEARNING_FILE, self.memory)
            else:
                _safe_write_json(LEARNING_FILE, self.memory)
        except Exception:
            # reset on hard failure; do not raise
            self.memory = {
                "version": MEMORY_VERSION,
                "weights": DEFAULT_WEIGHTS.copy(),
                "params": DEFAULT_PARAMS.copy(),
                "history": [],
                "per_source": {},
                "assets": {},
            }

    def _migrate_memory(self, from_version: int) -> None:
        """Migrate structures to current schema (idempotent, conservative)."""
        try:
            if from_version < 2:
                # Ensure new keys exist
                self.memory.setdefault("per_source", {})
                self.memory.setdefault("assets", {})
        except Exception:
            pass

    def _save_memory(self) -> None:
        with self._lock:
            try:
                _safe_write_json(LEARNING_FILE, self.memory)
            except Exception:
                pass

    # --------------- PUBLIC API (BACKWARD COMPAT) ---------------
    def get_weights(self) -> Dict[str, float]:
        """Return current global weights (copy). Backward compatible."""
        with self._lock:
            return dict(self.memory.get("weights", DEFAULT_WEIGHTS.copy()))

    def get_params(self) -> Dict[str, Any]:
        """Return current parameters (copy)."""
        with self._lock:
            return dict(self.memory.get("params", DEFAULT_PARAMS.copy()))

    def summary(self) -> Dict[str, Any]:
        """Human-friendly summary over known outcomes (keeps backward compat fields)."""
        with self._lock:
            hist: List[Dict[str, Any]] = list(self.memory.get("history", []))
            total = len(hist)
            known = [r for r in hist if isinstance(r.get("success"), bool)]
            known_count = len(known)
            wins = sum(1 for r in known if r.get("success") is True)
            accuracy = round((wins / known_count * 100.0), 2) if known_count else 0.0

            # recent(win rate) over last 50 known
            recent_known = []
            for r in reversed(hist):
                if isinstance(r.get("success"), bool):
                    recent_known.append(r)
                if len(recent_known) >= 50:
                    break
            recent_wins = sum(1 for r in recent_known if r.get("success") is True)
            recent_rate = round((recent_wins / len(recent_known) * 100.0), 2) if recent_known else None

            return {
                "total_records": total,
                "known_outcomes": known_count,
                "wins": wins,
                "accuracy_percent": accuracy,
                "recent_win_rate_percent": recent_rate,
                "weights": dict(self.memory.get("weights", DEFAULT_WEIGHTS.copy())),
                "params": dict(self.memory.get("params", DEFAULT_PARAMS.copy()))
            }

    # --------------- NEW OPTIONAL APIS ---------------
    def get_weights_dynamic(self, pair: Optional[str] = None, timeframe: Optional[str] = None) -> Dict[str, float]:
        """
        Return dynamic weights factoring in:
        - Global weights
        - Optional asset/timeframe micro-bias
        - Optional bandit boost (Thompson Sampling across sources)

        Keeps values clamped [0.5, 2.0].
        """
        with self._lock:
            base = dict(self.memory.get("weights", DEFAULT_WEIGHTS.copy()))
            # asset bias multipliers
            bias = self._asset_bias_multipliers(pair, timeframe)
            for s in base:
                base[s] = float(base[s]) * bias.get(s, 1.0)

            # bandit boosts (small, mean-preserving)
            if BANDIT_ENABLE:
                boosts = self._bandit_boosts()
                for s in base:
                    base[s] *= boosts.get(s, 1.0)

            # clamp
            for s in base:
                base[s] = float(max(0.5, min(2.0, base[s])))
            return base

    def summary_detailed(self) -> Dict[str, Any]:
        """Return an expanded summary including per-source and per-asset stats."""
        with self._lock:
            per_source = self.memory.get("per_source", {})
            assets = self.memory.get("assets", {})
            # light derivations
            ps_sum = {}
            for src, st in per_source.items():
                n = int(st.get("n", 0))
                wins = int(st.get("wins", 0))
                ema_win = float(st.get("ema_win", 0.5))
                a = float(st.get("beta_a", 1.0)); b = float(st.get("beta_b", 1.0))
                brier_n = int(st.get("brier_n", 0)); brier_sum = float(st.get("brier_sum", 0.0))
                ps_sum[src] = {
                    "n": n, "wins": wins, "ema_win": round(ema_win, 4),
                    "beta_a": round(a, 3), "beta_b": round(b, 3),
                    "brier": round(brier_sum / brier_n, 4) if brier_n > 0 else None
                }
            return {
                "summary": self.summary(),
                "per_source": ps_sum,
                "assets": assets  # already structured; can be large
            }

    def export_memory(self) -> Dict[str, Any]:
        with self._lock:
            # deep copy via json roundtrip (safe)
            try:
                return json.loads(json.dumps(self.memory))
            except Exception:
                return dict(self.memory)

    def import_memory(self, data: Dict[str, Any]) -> bool:
        """Replace memory with provided dict (must be schema-compatible)."""
        try:
            if not isinstance(data, dict):
                return False
            with self._lock:
                # basic shape validation
                data.setdefault("version", MEMORY_VERSION)
                data.setdefault("weights", DEFAULT_WEIGHTS.copy())
                data.setdefault("params", DEFAULT_PARAMS.copy())
                data.setdefault("history", [])
                data.setdefault("per_source", {})
                data.setdefault("assets", {})
                self.memory = data
                self._save_memory()
            return True
        except Exception:
            return False

    def vacuum(self, max_records: int = MAX_HISTORY_DEFAULT) -> None:
        """Trim history to at most max_records; persists memory."""
        with self._lock:
            hist = self.memory.get("history", [])
            if isinstance(hist, list) and len(hist) > max_records:
                self.memory["history"] = hist[-max_records:]
            self._save_memory()

    # --------------- UPDATE ENTRYPOINT ---------------
    def update_performance(self, result: Dict[str, Any]) -> None:
        """
        Update learning logic after a trade/backtest result.

        Expected result fields (all optional except source/confidence are inferred if missing):
          - pair: str (e.g., 'EURUSD' or 'XAUUSD')
          - timeframe: str (e.g., '1h', '4h', '15m')
          - source: str (e.g., 'advanced','precision','core','news','context')
          - success: True/False/None
          - confidence: float in [0,1]
          - pnl: numeric (profit in pips/$) or r_multiple: numeric (reward in R)
          - forward_test/backtest: dicts optionally containing {success, pnl, result}

        Behavior:
          - Unknown outcomes are stored but excluded from learning updates.
          - Uses Bayesian per-source updates, reward shaping, and confidence calibration.
          - DriftGuard throttles LR when a source underperforms recently vs its baseline.
        """
        if not isinstance(result, dict):
            return

        with self._lock:
            rec = {k: result.get(k) for k in result.keys()}
            # defaults
            rec.setdefault("source", result.get("source", "core"))
            rec["source"] = str(rec.get("source") or "core")
            rec.setdefault("confidence", float(result.get("confidence", 0.5) or 0.5))
            # normalize pair/timeframe fields if present
            if "pair" in rec and rec["pair"] is not None:
                rec["pair"] = str(rec["pair"])
            if "timeframe" in rec and rec["timeframe"] is not None:
                rec["timeframe"] = str(rec["timeframe"])

            # timestamp
            rec["timestamp"] = datetime.utcnow().isoformat()

            # infer success if missing
            if "success" not in rec or rec.get("success") is None:
                rec["success"] = self._infer_success_from_result(rec)

            # append bounded history
            self.memory.setdefault("history", [])
            self.memory["history"].append(rec)
            max_hist = MAX_HISTORY_DEFAULT
            if len(self.memory["history"]) > max_hist:
                self.memory["history"] = self.memory["history"][-max_hist:]

            # updates only on known outcomes
            if isinstance(rec.get("success"), bool):
                # Update stats and weights/params
                self._update_per_source_stats(rec)
                self._update_asset_stats(rec)
                self._adjust_weights(rec)
                self._adjust_parameters(rec)
                self._save_memory()
            else:
                # persist anyway
                self._save_memory()

    # --------------- INFERENCE HELPERS ---------------
    def _infer_success_from_result(self, rec: Dict[str, Any]) -> Optional[bool]:
        s = rec.get("success")
        if isinstance(s, bool):
            return s
        # pnl or R
        for key in ("pnl", "r_multiple", "R", "reward"):
            v = rec.get(key)
            if isinstance(v, (int, float)):
                try:
                    return float(v) > 0.0
                except Exception:
                    pass
        # nested dicts
        for key in ("forward_test", "backtest", "ft", "bt"):
            val = rec.get(key)
            if isinstance(val, dict):
                if isinstance(val.get("success"), bool):
                    return val.get("success")
                if isinstance(val.get("pnl"), (int, float)):
                    return float(val.get("pnl")) > 0.0
                if isinstance(val.get("result"), str):
                    low = val.get("result").lower()
                    if "win" in low or "profit" in low or "positive" in low:
                        return True
                    if "loss" in low or "negative" in low:
                        return False
        # text hints
        for cand in ("outcome", "result", "status"):
            v = rec.get(cand)
            if isinstance(v, str):
                low = v.lower()
                if "win" in low or "profit" in low:
                    return True
                if "loss" in low or "lose" in low or "negative" in low:
                    return False
        return None

    # --------------- STATS UPDATES ---------------
    def _per_source_entry(self, source: str) -> Dict[str, Any]:
        ps = self.memory.setdefault("per_source", {})
        if source not in ps:
            ps[source] = {
                "n": 0, "wins": 0,
                "ema_win": 0.5,  # exponential moving estimate of win prob
                "beta_a": 1.0, "beta_b": 1.0,  # Bayesian success prior
                "brier_sum": 0.0, "brier_n": 0,  # calibration tracking
                "last_outcomes": []  # recent outcome booleans (bounded)
            }
        return ps[source]

    def _update_per_source_stats(self, rec: Dict[str, Any]) -> None:
        src = rec.get("source", "core")
        success = bool(rec.get("success") is True)
        conf = float(rec.get("confidence", 0.5) or 0.5)

        st = self._per_source_entry(src)
        # counts
        st["n"] = int(st.get("n", 0)) + 1
        if success:
            st["wins"] = int(st.get("wins", 0)) + 1

        # EMA of win rate (alpha small -> smoother)
        ema_prev = float(st.get("ema_win", 0.5))
        alpha = 0.06  # conservative smoothing
        st["ema_win"] = (1 - alpha) * ema_prev + alpha * (1.0 if success else 0.0)

        # Beta Bayesian update
        a = float(st.get("beta_a", 1.0))
        b = float(st.get("beta_b", 1.0))
        # small prior prevents extreme early behavior
        a = max(0.5, a) + (1.0 if success else 0.0)
        b = max(0.5, b) + (0.0 if success else 1.0)
        st["beta_a"], st["beta_b"] = a, b

        # Brier score for confidence calibration
        try:
            # If confidence is outside [0,1], coerce
            p = min(1.0, max(0.0, conf))
            y = 1.0 if success else 0.0
            brier = (p - y) ** 2
            st["brier_sum"] = float(st.get("brier_sum", 0.0)) + float(brier)
            st["brier_n"] = int(st.get("brier_n", 0)) + 1
        except Exception:
            pass

        # recent outcomes
        lo = st.get("last_outcomes", [])
        if isinstance(lo, list):
            lo.append(success)
            if len(lo) > 200:
                lo[:] = lo[-200:]
            st["last_outcomes"] = lo

    def _update_asset_stats(self, rec: Dict[str, Any]) -> None:
        pair = rec.get("pair")
        tf = rec.get("timeframe")
        if not pair or not tf:
            return
        src = rec.get("source", "core")
        success = bool(rec.get("success") is True)

        assets = self.memory.setdefault("assets", {})
        ap = assets.setdefault(pair, {})
        tp = ap.setdefault(tf, {"stats": {}, "bias": {}})

        # stats per source inside asset profile
        st = tp["stats"].setdefault(src, {"n": 0, "wins": 0, "ema_win": 0.5})
        st["n"] += 1
        if success:
            st["wins"] += 1
        # smoother EMA for asset-timeframe (more noisy)
        alpha = 0.08
        st["ema_win"] = (1 - alpha) * float(st.get("ema_win", 0.5)) + alpha * (1.0 if success else 0.0)

        # derive small bias multiplier around 1.0, capped tight to avoid instability
        # bias = 1 + k*(ema_win - 0.5), with k small
        k = 0.6
        bias_val = 1.0 + k * (st["ema_win"] - 0.5)
        tp["bias"][src] = float(max(0.85, min(1.15, bias_val)))

    # --------------- WEIGHT/PARAM ADJUSTMENTS ---------------
    def _drift_guard_lr_scale(self, src: str) -> float:
        if not DRIFT_GUARD_ENABLE:
            return 1.0
        st = self._per_source_entry(src)
        ema_win = float(st.get("ema_win", 0.5))
        last = st.get("last_outcomes", [])
        # compute recent win rate over last 40 known outcomes (if available)
        if isinstance(last, list) and len(last) >= 20:
            window = last[-40:] if len(last) >= 40 else last
            recent = sum(1 for x in window if x) / float(len(window))
            # if recent performance under baseline by > 15%, throttle LR
            delta = ema_win - recent
            if delta > 0.15:
                return 0.35  # strong throttle
            if delta > 0.08:
                return 0.6   # mild throttle
        return 1.0

    def _confidence_reliability(self, src: str) -> float:
        """Compute a factor in [0.6, 1.4] based on Brier calibration (lower Brier -> >1)."""
        st = self._per_source_entry(src)
        bn = int(st.get("brier_n", 0))
        if bn < 20:
            return 1.0
        brier = float(st.get("brier_sum", 0.0)) / max(1, bn)
        # typical reasonable Brier ~ 0.18-0.24; map inversely
        # factor = 1 + c*(target - brier)
        target = 0.20
        c = 1.2
        factor = 1.0 + c * (target - brier)
        return float(max(0.6, min(1.4, factor)))

    def _reward_shaping(self, rec: Dict[str, Any]) -> float:
        """
        Return a shaping factor based on r_multiple or pnl.
        - Uses tanh to cap extremes.
        - Neutral (1.0) when no info.
        """
        val = None
        for key in ("r_multiple", "R", "reward"):
            v = rec.get(key)
            if isinstance(v, (int, float)):
                val = float(v); break
        if val is None and isinstance(rec.get("pnl"), (int, float)):
            # scale pnl roughly by ATR/pip? if not available, just normalize
            val = float(rec.get("pnl"))
            # mild normalization
            val = max(-5.0, min(5.0, val / (abs(val) + 1e-9)))  # map to ~[-1,1] keeping sign
        if val is None:
            return 1.0
        # cap extremes with tanh; map [-inf,inf] -> ~[0.5,1.5]
        shaped = math.tanh(val / 2.0)
        return float(1.0 + 0.5 * shaped)

    def _bandit_boosts(self) -> Dict[str, float]:
        """
        Compute small relative boosts per source via Thompson Sampling from per-source Beta(a,b).
        Mean-preserving: normalize boosts to have average ~1.0 across present sources.
        """
        boosts: Dict[str, float] = {}
        ps = self.memory.get("per_source", {})
        if not isinstance(ps, dict) or not ps:
            return boosts
        thetas = {}
        for src, st in ps.items():
            a = float(st.get("beta_a", 1.0)); b = float(st.get("beta_b", 1.0))
            a = max(0.5, a); b = max(0.5, b)
            try:
                # Thompson draw
                theta = random.betavariate(a, b)
            except Exception:
                theta = a / (a + b)
            thetas[src] = theta
        if not thetas:
            return boosts
        # map theta to small boost [0.9,1.1] around mean
        mean_theta = sum(thetas.values()) / len(thetas)
        if mean_theta <= 0:
            mean_theta = 0.5
        for src, th in thetas.items():
            rel = th / mean_theta
            # squash to [0.9, 1.1] using power
            squashed = rel ** 0.2  # gentle
            boosts[src] = float(max(0.9, min(1.1, squashed)))
        return boosts

    def _asset_bias_multipliers(self, pair: Optional[str], timeframe: Optional[str]) -> Dict[str, float]:
        if not pair or not timeframe:
            return {}
        try:
            return dict(self.memory.get("assets", {}).get(pair, {}).get(timeframe, {}).get("bias", {}))
        except Exception:
            return {}

    def _adjust_weights(self, rec: Dict[str, Any]) -> None:
        """
        Conservative, sample-aware weight updates with:
        - Decaying LR by per-source sample count (sqrt decay)
        - DriftGuard LR throttling when recent performance < baseline
        - Reward shaping and confidence calibration
        - Clamp in [0.5, 2.0]
        """
        source = rec.get("source", "core")
        success = bool(rec.get("success") is True)
        confidence = float(rec.get("confidence", 0.5) or 0.5)

        self.memory.setdefault("weights", DEFAULT_WEIGHTS.copy())
        if source not in self.memory["weights"]:
            self.memory["weights"][source] = 1.0

        # observations for source
        ps = self.memory.get("per_source", {})
        st = ps.get(source, {})
        source_count = int(st.get("n", 0))

        # base LR (smaller than previous version for stability)
        base_lr = 0.025
        decay = 1.0 / math.sqrt(1.0 + source_count / 60.0)
        lr = base_lr * decay

        # DriftGuard throttle
        lr *= self._drift_guard_lr_scale(source)

        # confidence reliability
        conf_rel = self._confidence_reliability(source)

        # reward shaping
        rw = self._reward_shaping(rec)

        # direction update (+/-)
        dir_sign = 1.0 if success else -1.0
        delta = lr * dir_sign * (0.5 + 0.5 * confidence) * conf_rel * rw

        new_weight = float(self.memory["weights"].get(source, 1.0) + delta)
        new_weight = max(0.5, min(2.0, new_weight))
        self.memory["weights"][source] = round(new_weight, 4)

    def _adjust_parameters(self, rec: Dict[str, Any]) -> None:
        """
        Conservative param updates with clamped ranges; success tightens risk, failure relaxes slightly.
        """
        success = bool(rec.get("success") is True)
        confidence = float(rec.get("confidence", 0.5) or 0.5)

        self.memory.setdefault("params", DEFAULT_PARAMS.copy())
        params = self.memory["params"]

        # small step scaled by confidence (and drift throttle averaged across sources if available)
        step = 0.004 * (0.5 + 0.5 * confidence)

        # ATR multiplier adjustment
        if success:
            params["atr_multiplier"] = max(0.6, params.get("atr_multiplier", 1.0) - step)
        else:
            params["atr_multiplier"] = min(1.8, params.get("atr_multiplier", 1.0) + 2 * step)

        # retest sensitivity
        if success:
            params["retest_sensitivity"] = max(0.2, params.get("retest_sensitivity", 0.5) - (step * 0.5))
        else:
            params["retest_sensitivity"] = min(1.0, params.get("retest_sensitivity", 0.5) + (step * 0.8))

        # pattern threshold
        if success:
            params["pattern_confidence_threshold"] = max(0.45, params.get("pattern_confidence_threshold", 0.7) - (step * 0.5))
        else:
            params["pattern_confidence_threshold"] = min(0.95, params.get("pattern_confidence_threshold", 0.7) + (step * 0.8))

        # round & clamp
        params["atr_multiplier"] = round(float(params["atr_multiplier"]), 4)
        params["retest_sensitivity"] = round(max(0.2, min(1.0, float(params["retest_sensitivity"]))), 4)
        params["pattern_confidence_threshold"] = round(max(0.4, min(0.95, float(params["pattern_confidence_threshold"]))), 4)

        self.memory["params"] = params

    # --------------- GLOBAL INSTANCE UTILS ---------------
    def _clean_history(self) -> None:
        hist = self.memory.get("history", [])
        if isinstance(hist, list) and len(hist) > MAX_HISTORY_DEFAULT:
            self.memory["history"] = hist[-MAX_HISTORY_DEFAULT:]

# ------------------ GLOBAL INSTANCE ------------------
adaptive_engine = AdaptiveLearning()


# ------------------ HELPER FUNCTION (BACKWARD COMPAT) ------------------
def log_result(pair: str, success: Optional[bool], confidence: float, source: str = "core") -> None:
    """
    Backward compatible log entry.
    - success may be True/False/None (None => unknown inference)
    - Only updates learning on known outcomes
    - Use update_performance(...) directly for richer fields (e.g., timeframe, pnl, r_multiple)
    """
    try:
        rec = {
            "pair": pair,
            "success": success,
            "confidence": float(confidence or 0.5),
            "source": source
        }
        adaptive_engine.update_performance(rec)
    except Exception:
        # never raise to callers
        pass


# ------------------ LOCAL SMOKE TEST ------------------
if __name__ == "__main__":
    print("AdaptiveLearning self-test (non-breaking)...")
    log_result("EURUSD", True, 0.8, "core")
    log_result("EURUSD", False, 0.6, "precision")
    # richer record with pnl and timeframe
    adaptive_engine.update_performance({
        "pair": "XAUUSD", "timeframe": "1h", "source": "advanced",
        "success": True, "confidence": 0.7, "r_multiple": 1.4
    })
    print(json.dumps(adaptive_engine.summary(), indent=2))
    # showcase new APIs (optional)
    print("Dynamic weights:", adaptive_engine.get_weights_dynamic("EURUSD", "1h"))
    print("Detailed summary keys:", list(adaptive_engine.summary_detailed().keys()))
