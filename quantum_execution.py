"""
quantum_execution.py
--------------------
QEF (Quantum Execution Field)

Purpose:
    - Given a base entry idea from your existing pipeline, and PREX+TEO context,
      explore a small 'field' of nearby candidate entries and pick the one with
      maximum *expected edge*, not just maximum safety.
    - Avoid:
        * Entries that look good but have poor fill probability (BTC case).
        * 'Too safe' entries that give away most of the move (overly timid).
    - Provide:
        * best_entry (price, sl, tp, edge, label),
        * alt_entry (more conservative),
        * candidate list for debugging/visualization.

Integration:
    - Import and call from cli.py AFTER you have:
        * PrexVerdict (microstructure + risk_multiplier)
        * TEOResult (direction bias, pattern, wait horizon)
        * base_entry suggestion from your engines (if any).
    - If you have no base entry, you can still call this with a rough zone
      or skip and only use TEO.

This module is intentionally approximate but structured to be improved over time.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import math

from prex_guardian import MicrostructureProfile
from temporal_oracle import TEOResult


@dataclass
class QEntryCandidate:
    price: float
    sl: float
    tp: float
    edge: float
    label: str
    fill_probability: float
    win_probability: float
    reward_r: float
    risk_r: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QEFResult:
    best_entry: Optional[QEntryCandidate]
    alt_entry: Optional[QEntryCandidate]
    entry_candidates: List[QEntryCandidate]
    expected_wait_candles: int
    expected_wait_minutes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_entry": self.best_entry.to_dict() if self.best_entry else None,
            "alt_entry": self.alt_entry.to_dict() if self.alt_entry else None,
            "entry_candidates": [c.to_dict() for c in self.entry_candidates],
            "expected_wait_candles": self.expected_wait_candles,
            "expected_wait_minutes": self.expected_wait_minutes,
        }


# ---------------------------------------------------------------------------
# Helper approximations
# ---------------------------------------------------------------------------

def _approx_win_probability(
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    recent_trend_slope: float,
) -> float:
    """
    Very rough probability-of-win proxy:
      - Favors entries aligned with slope.
      - Rewards better RR, but avoids absurd distance.
    """

    if sl <= 0 or tp <= 0 or entry <= 0:
        return 0.5

    # R/R
    if direction == "Buy":
        reward = max(tp - entry, 0.0)
        risk = max(entry - sl, 1e-9)
    else:
        reward = max(entry - tp, 0.0)
        risk = max(sl - entry, 1e-9)

    rr = reward / risk if risk > 0 else 1.0

    # Base probability
    p = 0.5

    # Nudge by trend slope
    p += 0.1 * math.tanh(recent_trend_slope * 5.0)  # squash

    # Nudge by RR (penalize extreme RR values)
    if rr > 0 and rr < 4.0:
        p += 0.1 * math.tanh((rr - 1.0))  # prefer >1R but not insane
    elif rr >= 4.0:
        p -= 0.05  # extremely far TP is less likely in practice

    # clamp
    return max(0.1, min(0.9, p))


def _approx_fill_probability(
    direction: str,
    current_price: float,
    entry: float,
    micro: MicrostructureProfile,
) -> float:
    """
    Approximate probability that a pending entry will actually fill.
    - If entry is very far from current_price, probability decreases.
    - If entry is inside spread region and we'll likely use market, treat as high.
    """

    spread = micro.spread_normal
    eps = micro.tick_size

    dist = abs(entry - current_price)
    if dist <= spread * 0.5:
        return 0.95  # essentially inside/in-touch zone
    if dist <= spread * 2.0:
        return 0.85
    if dist <= spread * 5.0:
        return 0.7
    if dist <= spread * 10.0:
        return 0.5
    return 0.3  # far away


def _compute_edge(
    p_win: float,
    p_fill: float,
    reward_r: float,
    risk_r: float,
    exec_penalty: float,
) -> float:
    """
    QEdge = P(fill) * (P(win) * Reward - (1 - P(win)) * Risk) - ExecPenalty
    """
    ev = p_fill * (p_win * reward_r - (1.0 - p_win) * risk_r)
    return ev - exec_penalty


def _snap_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(price / tick) * tick


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_qef(
    symbol: str,
    timeframe: str,
    direction: str,
    current_price: float,
    base_entry: Optional[Dict[str, float]],
    micro: MicrostructureProfile,
    teo: TEOResult,
    recent_trend_slope: float = 0.0,
) -> QEFResult:
    """
    Main entrypoint.

    Args:
        symbol: e.g. "XAUUSDm"
        timeframe: e.g. "5m"
        direction: "Buy" / "Sell" (if "Neutral", QEF will not propose entries)
        current_price: latest mid or appropriate reference price
        base_entry: dict with keys:
            {
              "entry": float,
              "sl": float,
              "tp": float
            }
          from your existing pipeline. If None, QEF still returns timing info only.
        micro: MicrostructureProfile from PREX.
        teo: TEOResult from Temporal Oracle.
        recent_trend_slope: optional slope measure (positive up, negative down).

    Returns:
        QEFResult with chosen best_entry, alt_entry, and candidate list.
    """

    # Default: no entries, just propagate waiting info
    if direction not in ("Buy", "Sell") or base_entry is None:
        return QEFResult(
            best_entry=None,
            alt_entry=None,
            entry_candidates=[],
            expected_wait_candles=teo.wait_candles,
            expected_wait_minutes=teo.estimated_wait_minutes,
        )

    entry0 = float(base_entry["entry"])
    sl0 = float(base_entry["sl"])
    tp0 = float(base_entry["tp"])

    # Candidate generation: slight perturbations around base entry
    # We scale by a fraction of min_stop_points to avoid going too far.
    step = max(micro.min_stop_points * 0.3, micro.tick_size * 2.0)

    candidates_raw: List[Dict[str, float]] = []

    if direction == "Buy":
        # base, slightly better price, slightly worse / faster
        candidates_raw.append({"price": entry0,           "label": "base"})
        candidates_raw.append({"price": entry0 - step,    "label": "conservative"})
        candidates_raw.append({"price": entry0 + step*0.5,"label": "aggressive"})
    else:  # Sell
        candidates_raw.append({"price": entry0,           "label": "base"})
        candidates_raw.append({"price": entry0 + step,    "label": "conservative"})
        candidates_raw.append({"price": entry0 - step*0.5,"label": "aggressive"})

    # Build QEntryCandidates
    q_candidates: List[QEntryCandidate] = []
    for raw in candidates_raw:
        price = _snap_to_tick(raw["price"], micro.tick_size)
        # Adjust SL/TP relative to new price keeping R/R ratio roughly intact
        if direction == "Buy":
            base_risk = entry0 - sl0
            base_reward = tp0 - entry0
        else:
            base_risk = sl0 - entry0
            base_reward = entry0 - tp0

        if base_risk <= 0:
            base_risk = micro.min_stop_points

        rr = base_reward / base_risk if base_risk > 0 else 1.5
        risk = max(micro.min_stop_points, base_risk)  # ensure at least min stop

        if direction == "Buy":
            sl = price - risk
            tp = price + rr * risk
        else:
            sl = price + risk
            tp = price - rr * risk

        # approximate risk & reward in R units:
        risk_r = 1.0
        reward_r = rr

        p_win = _approx_win_probability(direction, price, sl, tp, recent_trend_slope)
        p_fill = _approx_fill_probability(direction, current_price, price, micro)

        # Exec penalty: far away from current_price + too conservative/wide
        distance = abs(price - current_price)
        # penalty grows with distance and with extremely large RR
        exec_penalty = (distance / max(micro.min_stop_points, micro.tick_size)) * 0.05
        if rr > 3.0:
            exec_penalty += 0.05

        edge = _compute_edge(p_win, p_fill, reward_r, risk_r, exec_penalty)

        q_candidates.append(
            QEntryCandidate(
                price=price,
                sl=sl,
                tp=tp,
                edge=edge,
                label=raw["label"],
                fill_probability=p_fill,
                win_probability=p_win,
                reward_r=reward_r,
                risk_r=risk_r,
            )
        )

    # Pick best and conservative alternative
    if not q_candidates:
        return QEFResult(
            best_entry=None,
            alt_entry=None,
            entry_candidates=[],
            expected_wait_candles=teo.wait_candles,
            expected_wait_minutes=teo.estimated_wait_minutes,
        )

    # Sort by edge descending
    q_candidates_sorted = sorted(q_candidates, key=lambda c: c.edge, reverse=True)
    best_entry = q_candidates_sorted[0]

    # conservative: best candidate that is not labelled "aggressive"
    conservative = None
    for c in q_candidates_sorted:
        if c.label != "aggressive":
            conservative = c
            break

    return QEFResult(
        best_entry=best_entry,
        alt_entry=conservative if conservative and conservative != best_entry else None,
        entry_candidates=q_candidates_sorted,
        expected_wait_candles=teo.wait_candles,
        expected_wait_minutes=teo.estimated_wait_minutes,
    )
