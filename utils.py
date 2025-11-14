import ta
import pandas as pd
import MetaTrader5 as mt5
import numpy as np

SUPPORTED_TIMEFRAMES = ["15m", "30m", "1h"]
HIGHER_TIMEFRAME = "1h"

TIMEFRAME_TO_MT5 = {
 "15m": "TIMEFRAME_M15",
 "30m": "TIMEFRAME_M30",
 "1h": "TIMEFRAME_H1"
}

def indicator_signals(df):
 signals = {}
 if df is None or df.empty or "close" not in df.columns:
  return {"error": "Invalid DataFrame"}

 df = df.copy().dropna()
 try:
  rsi = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
  signals["RSI"] = "Buy" if rsi.iloc[-1] < 30 else "Sell" if rsi.iloc[-1] > 70 else "Neutral"
 except Exception as e:
  signals["RSI"] = "Neutral"

 try:
  macd = ta.trend.MACD(df["close"])
  macd_diff = macd.macd_diff().iloc[-1]
  signals["MACD"] = "Buy" if macd_diff > 0 else "Sell" if macd_diff < 0 else "Neutral"
 except Exception as e:
  signals["MACD"] = "Neutral"

 try:
  ema_fast = ta.trend.EMAIndicator(df["close"], window=12).ema_indicator()
  ema_slow = ta.trend.EMAIndicator(df["close"], window=26).ema_indicator()
  signals["EMA"] = "Buy" if ema_fast.iloc[-1] > ema_slow.iloc[-1] else "Sell" if ema_fast.iloc[-1] < ema_slow.iloc[-1] else "Neutral"
 except Exception as e:
  signals["EMA"] = "Neutral"

 try:
  bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
  bb_signal = "Buy" if df["close"].iloc[-1] < bb.bollinger_lband().iloc[-1] else "Sell" if df["close"].iloc[-1] > bb.bollinger_hband().iloc[-1] else "Neutral"
  signals["Bollinger"] = bb_signal
 except Exception as e:
  signals["Bollinger"] = "Neutral"

 try:
  adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx().iloc[-1]
  adx_trend = "Strong" if adx > 25 else "Weak"
  signals["ADX_strength"] = adx_trend
 except Exception as e:
  signals["ADX_strength"] = "Weak"

 try:
  stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
  k = stoch.stoch().iloc[-1]
  d = stoch.stoch_signal().iloc[-1]
  stoch_sig = "Buy" if k < 20 and k > d else "Sell" if k > 80 and k < d else "Neutral"
  signals["Stochastic"] = stoch_sig
 except Exception as e:
  signals["Stochastic"] = "Neutral"

 return signals

def ensemble_signal(signals, trend):
 if not isinstance(signals, dict):
  return "Neutral"
 score = 0
 weights = {"RSI": 1, "MACD": 1, "EMA": 1, "Bollinger": 1, "Stochastic": 1}
 for key, val in signals.items():
  w = weights.get(key, 1)
  if val == "Buy": score += w
  elif val == "Sell": score -= w
 if trend == "Buy": score += 2
 elif trend == "Sell": score -= 2
 if score > 1: return "Buy"
 if score < -1: return "Sell"
 return "Neutral"

def multi_timeframe_trend(pair):
 tf = mt5.TIMEFRAME_H1
 if not mt5.initialize():
  raise Exception("MT5 not initialized in multi_timeframe_trend")
 sym = mt5.symbol_info(pair)
 if sym is None:
  raise Exception(f"Symbol '{pair}' not found")
 if not sym.visible:
  mt5.symbol_select(pair, True)
 rates = mt5.copy_rates_from_pos(pair, tf, 0, 100)
 if rates is None or len(rates) < 20:
  raise Exception("Insufficient data for higher timeframe trend")
 df = pd.DataFrame(rates)
 df["time"] = pd.to_datetime(df["time"], unit="s")
 df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=12).ema_indicator()
 df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=26).ema_indicator()
 df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
 if df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]:
  trend = "Buy"
 elif df["ema_fast"].iloc[-1] < df["ema_slow"].iloc[-1]:
  trend = "Sell"
 else:
  trend = "Neutral"
 if df["adx"].iloc[-1] < 20:
  trend = "Neutral"
 mt5.shutdown()
 return trend
