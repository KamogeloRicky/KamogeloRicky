# Entry Optimizer for RickyFX

This module improves entry placement quality by:
- Using structure (swing highs/lows), impulse legs, FVGs, and liquidity sweeps.
- Preferring LIMIT retests in discount/premium zones, not random mid-zones.
- Applying ATR-aware SL buffers and minimum distance heuristics.
- Scoring confluence and rejecting low-quality setups.

## Install

Place `entry_optimizer.py` (and optional `entry_optimizer_profiles.py`) next to your GUI and executor.

## Environment Flags

- `ENTRY_OPTIMIZER_DISABLE=1` to bypass optimizer and use analysis values.
- `ENTRY_OPTIMIZER_MIN_SCORE=0.6` to raise/lower the quality gate.
- `ENTRY_OPTIMIZER_BARS=1000`, `ENTRY_OPTIMIZER_ATR_PERIOD=14` to tune data depth.
- `ENTRY_OPTIMIZER_RR_TARGET=1.8`, `ENTRY_OPTIMIZER_ATR_SL_BUF=0.5` to tune exits.
- `ENTRY_OPTIMIZER_ALLOW_BREAKOUT=1` to allow STOP entries on breakouts (default retest LIMIT).
- `ENTRY_OPTIMIZER_SHOW=1` to always include diagnostics in the result map.

## GUI Integration (one-liner)

In your Confirm Trade flow, before calling `place_pending_order`, add:

```python
from entry_optimizer import optimize_entry

# analysis_dict: the dict you already render
refined = optimize_entry(analysis_dict, symbol=symbol, timeframe=self.combo_tf.currentText())
entry_used = refined["entry"] if refined.get("ok") else entry_from_analysis
sl_used = refined["sl"] if refined.get("ok") else sl_from_analysis
tp_used = refined["tp"] if refined.get("ok") else tp_from_analysis
```

Then pass `entry_used`, `sl_used`, `tp_used` to your executor.

Append an “Optimization Summary” block in your execution summary:
- `refined["ok"]`, `refined["score"]`
- `refined["pending_type"]`, `refined["zone_bounds"]`
- `refined["reasons"]`, `refined["diagnostics"]`

## Tuning

- Increase `ENTRY_OPTIMIZER_MIN_SCORE` to be more selective (fewer but higher-quality trades).
- Increase `ENTRY_OPTIMIZER_ATR_SL_BUF` on volatile symbols (e.g., crypto).
- Raise `ENTRY_OPTIMIZER_RR_TARGET` for stronger risk:reward but fewer fills.
- For indices, a higher `min_score` and `rr_target` is recommended.

## Limitations

- No optimizer can guarantee 95%+ accuracy across markets. This aims to filter weak entries and focus on realistic retests with confluence, materially improving win rate and expectancy.
- Always forward-test on demo, and iterate the thresholds/profiles per instrument group.
