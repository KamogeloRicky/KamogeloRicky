# Forex Analysis Bot (Extreme Accuracy Edition)

## Features
- CLI-based forex pair analysis (15m, 30m, 1h, real MT5/Exness)
- Ensemble of indicators: EMA, RSI, MACD, Stochastic, Bollinger Bands, ADX, Ichimoku, Parabolic SAR (step=0.02, max_step=0.2)
- Multi-timeframe trend confirmation
- ATR-based dynamic stop-loss/take-profit suggestion
- Logging of all analyses for review
- Modular, easily extensible for ML, auto-trading, and notifications

## Quickstart

1. Download and install MetaTrader 5 from [Exness](https://www.exness.com/mt5/)
2. Create a real or demo account on Exness and note your login, password, and server (e.g., "Exness-Real")
3. Install requirements:

```bash
pip install -r requirements.txt
```

4. Edit `analysis.py` and set your Exness login, password, and server at the top.
5. Run:

```bash
python main.py
```

## Usage
- Enter a forex pair (e.g., EURUSD)
- Choose a timeframe (15m, 30m, 1h)
- View the detailed ensemble analysis

---

**Warning:**  
No system can be 100% accurate. Use demo first, manage risk, and understand all trades are at your own risk.

---
