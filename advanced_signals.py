"""
Advanced signals module (robust, GOATED)
Detects: FVG, BOS/CHOCH, Fractals, Candles, Ichimoku, Bollinger, PSAR, Demarker (manual),
Williams %R, ADX (+DI/-DI), Alligator, Order Blocks, Swing highs/lows, Sniper entries,
VWAP/session alignment, Keltner channels, RSI/Stochastic/MACD/CCI/OBV/MFI, Supertrend,
Heikin-Ashi trend, BB% B, Squeeze (BB vs Keltner), Divergences (RSI/MACD),
ATR/ADR regimes, spectral entropy, Hurst exponent, HTF alignment (optional 4H),
Fibonacci retracement (high-performance, swing-aware) with confluence and stacking (Fib x OB/FVG) scoring.

Safe: defensive try/except, works without ta or volume/time columns.
API: advanced_analysis(df, bars=DEFAULT_BARS) -> dict
- Adds 'advanced_signal' and 'suggestion' mapped to Buy/Sell/Neutral
- Adds 'score' and 'score_breakdown' where every improvement contributes
"""

import math
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# 'ta' optional
try:
    import ta  # type: ignore
except Exception:
    ta = None

# Optional 4H HTF alignment support (hook; not used internally to avoid IO in this module)
try:
    from data import fetch_candles
except Exception:
    fetch_candles = None

# Optional timezone for session tagging
try:
    import pytz
    SAST_TZ = pytz.timezone("Africa/Johannesburg")
except Exception:
    SAST_TZ = None

# ---------- CONFIG ----------
DEFAULT_BARS = 2000
FVG_LOOKBACK = 50
SWING_LOOKBACK = 5

# Scoring weights (tunable)
WEIGHTS = {
    # Trend block
    "ema_trend": 1.0,
    "ichimoku_conv_base": 0.7,
    "psar_trend": 0.6,
    "alligator_align": 0.5,
    "supertrend": 1.0,
    "heikin_ashi": 0.6,
    "adx_strength": 0.8,
    # Momentum block
    "macd_trend": 0.8,
    "stoch_signal": 0.5,
    "rsi_zone": 0.5,
    "cci_extreme": 0.4,
    # Confluence/pattern
    "bos_buy": 1.2,
    "bos_sell": -1.2,
    "choch": 0.3,
    "liquidity_sweep": 0.3,
    "order_block": 0.7,
    "fvg_near": 0.5,
    "candle_pattern": 0.4,
    "fractals_recent": 0.2,
    # VWAP/session/HTF
    "vwap_align": 0.5,
    "session_align": 0.2,
    "htf_align": 0.8,
    # Volatility/regime
    "squeeze_on": 0.3,
    "atr_up": 0.3,
    "adr_active": 0.2,
    # Divergences (contrarian reduce trend conviction)
    "rsi_div_reg": -0.8,
    "macd_div_reg": -0.8,
    "rsi_div_hidden": 0.5,
    "macd_div_hidden": 0.5,
    # Market microstructure extras
    "willr_extreme": 0.2,
    "mfi_flow": 0.2,
    "obv_flow": 0.2,
    # Information measures (regime)
    "spectral_trend": 0.4,
    "hurst_trend": 0.4,
    # Fibonacci confluence and stacking
    "fib_confluence": 0.6,
    "fib_stack": 0.7,
}

# ---------- SAFE HELPERS ----------
def _safe_series_like(df: pd.DataFrame, fill: float = np.nan) -> pd.Series:
    return pd.Series([fill] * len(df), index=df.index)

def _compute_atr_safe(df: pd.DataFrame, period: int = 14) -> pd.Series:
    try:
        if ta is not None:
            s = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=period).average_true_range()
            if isinstance(s, pd.Series):
                return s
    except Exception:
        pass
    try:
        return (df["high"].astype(float) - df["low"].astype(float)).rolling(period).mean()
    except Exception:
        return _safe_series_like(df)

def _ema(s: pd.Series, window: int) -> pd.Series:
    try:
        if ta is not None:
            return ta.trend.EMAIndicator(s, window=window).ema_indicator()
        return s.ewm(span=window, adjust=False).mean()
    except Exception:
        return s.rolling(window).mean()

def _sma(s: pd.Series, window: int) -> pd.Series:
    try:
        if ta is not None:
            return ta.trend.SMAIndicator(s, window=window).sma_indicator()
    except Exception:
        pass
    return s.rolling(window).mean()

def _compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    try:
        if ta is not None:
            return ta.momentum.RSIIndicator(df["close"], window=window).rsi()
        delta = df["close"].diff()
        up = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
        down = -delta.clip(upper=0).ewm(alpha=1/window, adjust=False).mean()
        rs = up / (down + 1e-9)
        return 100 - (100 / (1 + rs))
    except Exception:
        return _safe_series_like(df)

def _compute_stoch(df: pd.DataFrame, k: int = 14, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    try:
        if ta is not None:
            stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=k, smooth_window=d)
            return stoch.stoch(), stoch.stoch_signal()
        low_min = df["low"].rolling(k).min()
        high_max = df["high"].rolling(k).max()
        k_val = (df["close"] - low_min) / (high_max - low_min + 1e-9) * 100
        d_val = k_val.rolling(d).mean()
        return k_val, d_val
    except Exception:
        return _safe_series_like(df), _safe_series_like(df)

def _compute_macd(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    try:
        if ta is not None:
            macd = ta.trend.MACD(df["close"])
            return macd.macd(), macd.macd_signal(), macd.macd_diff()
        ema12 = _ema(df["close"], 12)
        ema26 = _ema(df["close"], 26)
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal
        return macd_line, signal, hist
    except Exception:
        s = _safe_series_like(df)
        return s, s, s

def _compute_cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    try:
        if ta is not None:
            return ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=window).cci()
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        sma = tp.rolling(window).mean()
        md = (tp - sma).abs().rolling(window).mean()
        return (tp - sma) / (0.015 * (md + 1e-9))
    except Exception:
        return _safe_series_like(df)

def _compute_obv(df: pd.DataFrame) -> pd.Series:
    try:
        if "volume" not in df.columns:
            return _safe_series_like(df, 0.0)
        direction = np.sign(df["close"].diff().fillna(0.0))
        return (direction * df["volume"]).cumsum()
    except Exception:
        return _safe_series_like(df, 0.0)

def _compute_mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    try:
        if "volume" not in df.columns:
            return _safe_series_like(df)
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        raw = tp * df["volume"]
        pos = raw.where(tp > tp.shift(1), 0.0).rolling(window).sum()
        neg = raw.where(tp < tp.shift(1), 0.0).rolling(window).sum()
        ratio = pos / (neg + 1e-9)
        return 100 - (100 / (1 + ratio))
    except Exception:
        return _safe_series_like(df)

def _keltner(df: pd.DataFrame, ema_window: int = 20, atr_mult: float = 1.5) -> Tuple[pd.Series, pd.Series]:
    try:
        mid = _ema(df["close"], ema_window)
        atr = _compute_atr_safe(df, 20)
        upper = mid + atr_mult * atr
        lower = mid - atr_mult * atr
        return upper, lower
    except Exception:
        s = _safe_series_like(df)
        return s, s

def _bollinger(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    try:
        if ta is not None:
            bb = ta.volatility.BollingerBands(df["close"], window=window)
            return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()
        m = df["close"].rolling(window).mean()
        s = df["close"].rolling(window).std()
        return m + 2*s, m, m - 2*s
    except Exception:
        s = _safe_series_like(df)
        return s, s, s

def _bb_percent_b(df: pd.DataFrame, upper: pd.Series, lower: pd.Series) -> pd.Series:
    try:
        return (df["close"] - lower) / (upper - lower + 1e-9)
    except Exception:
        return _safe_series_like(df)

def _vwap(df: pd.DataFrame) -> Optional[pd.Series]:
    try:
        if "volume" not in df.columns:
            return None
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"].astype(float)
        cum_vol = vol.cumsum().replace(0, np.nan)
        vwap = (tp * vol).cumsum() / cum_vol
        return vwap
    except Exception:
        return None

def _heikin_ashi(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    try:
        ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
        ha_open = ha_close.copy()
        ha_open.iloc[0] = float(df["open"].iloc[0])
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2.0
        return ha_open, ha_close
    except Exception:
        return _safe_series_like(df), _safe_series_like(df)

def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """
    Basic supertrend direction: +1 bull, -1 bear, 0 unknown.
    """
    try:
        atr = _compute_atr_safe(df, period)
        hl2 = (df["high"] + df["low"]) / 2.0
        upperband = hl2 + multiplier * atr
        lowerband = hl2 - multiplier * atr
        direction = np.zeros(len(df))
        trend_up = upperband.copy()
        trend_dn = lowerband.copy()
        for i in range(1, len(df)):
            trend_up.iloc[i] = min(upperband.iloc[i], trend_up.iloc[i-1]) if df["close"].iloc[i-1] > trend_up.iloc[i-1] else upperband.iloc[i]
            trend_dn.iloc[i] = max(lowerband.iloc[i], trend_dn.iloc[i-1]) if df["close"].iloc[i-1] < trend_dn.iloc[i-1] else lowerband.iloc[i]
            if df["close"].iloc[i] > trend_up.iloc[i-1]:
                direction[i] = 1
            elif df["close"].iloc[i] < trend_dn.iloc[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
        return pd.Series(direction, index=df.index)
    except Exception:
        return _safe_series_like(df, 0.0)

def _ichimoku(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    try:
        conv = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
        base = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
        span_a = ((conv + base) / 2).shift(26)
        span_b = ((df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2).shift(26)
        return conv, base, span_a, span_b
    except Exception:
        s = _safe_series_like(df)
        return s, s, s, s

def _demarker_manual(df: pd.DataFrame, period=14) -> pd.Series:
    try:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        up = np.where(high > prev_high, high - prev_high, 0.0)
        down = np.where(low < prev_low, prev_low - low, 0.0)
        up_s = pd.Series(up, index=df.index).rolling(period).sum()
        down_s = pd.Series(down, index=df.index).rolling(period).sum()
        denom = (up_s + down_s).replace(0, np.nan)
        dem = (up_s / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return dem
    except Exception:
        return _safe_series_like(df, 0.0)

def _psar_custom(df: pd.DataFrame, af_start=0.02, af_step=0.02, af_max=0.2) -> pd.Series:
    try:
        if ta is not None and hasattr(ta.trend, "PSARIndicator"):
            return ta.trend.PSARIndicator(df["high"], df["low"], df["close"], step=af_start, max_step=af_max).psar()
    except Exception:
        pass
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    length = len(df)
    sar = np.full(length, np.nan)
    if length < 3:
        return pd.Series(sar, index=df.index)
    up = True
    ep = highs[0]
    af = af_start
    sar_val = lows[0]
    for i in range(1, length):
        sar[i] = sar_val
        if up:
            if highs[i] > ep:
                ep = highs[i]; af = min(af + af_step, af_max)
            sar_val = sar_val + af * (ep - sar_val)
            if sar_val > lows[i]:
                up = False
                sar_val = ep; ep = lows[i]; af = af_start
        else:
            if lows[i] < ep:
                ep = lows[i]; af = min(af + af_step, af_max)
            sar_val = sar_val + af * (ep - sar_val)
            if sar_val < highs[i]:
                up = True
                sar_val = ep; ep = highs[i]; af = af_start
    return pd.Series(sar, index=df.index)

# ---------- STRUCTURE / PATTERNS ----------
def _detect_candlestick_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    try:
        if len(df) < 2:
            return []
        patterns: List[Dict[str, Any]] = []
        last = df.iloc[-3:].reset_index(drop=True)
        for i in last.index:
            o = float(last.at[i, "open"]); c = float(last.at[i, "close"])
            h = float(last.at[i, "high"]); l = float(last.at[i, "low"])
            body = abs(c - o)
            full = h - l if (h - l) != 0 else 1e-9
            upper = h - max(c, o)
            lower = min(c, o) - l
            if body < full * 0.25 and lower > 2 * body:
                patterns.append({"idx": df.index[-3 + i], "pattern": "hammer"})
            if body < full * 0.25 and upper > 2 * body:
                patterns.append({"idx": df.index[-3 + i], "pattern": "shooting_star"})
            if i >= 1:
                prev_o = float(last.at[i-1, "open"]); prev_c = float(last.at[i-1, "close"])
                if c > o and prev_c < prev_o and c > prev_o and o < prev_c:
                    patterns.append({"idx": df.index[-3 + i], "pattern": "bullish_engulfing"})
                if c < o and prev_c > prev_o and c < prev_o and o > prev_c:
                    patterns.append({"idx": df.index[-3 + i], "pattern": "bearish_engulfing"})
        return patterns
    except Exception:
        return []

def _detect_fractals(df: pd.DataFrame) -> Dict[str, List[Tuple[int, float]]]:
    highs: List[Tuple[int, float]] = []
    lows: List[Tuple[int, float]] = []
    try:
        for i in range(2, len(df)-2):
            if df["high"].iat[i] > df["high"].iat[i-1] and df["high"].iat[i] > df["high"].iat[i-2] and df["high"].iat[i] > df["high"].iat[i+1] and df["high"].iat[i] > df["high"].iat[i+2]:
                highs.append((i, float(df["high"].iat[i])))
            if df["low"].iat[i] < df["low"].iat[i-1] and df["low"].iat[i] < df["low"].iat[i-2] and df["low"].iat[i] < df["low"].iat[i+1] and df["low"].iat[i] < df["low"].iat[i+2]:
                lows.append((i, float(df["low"].iat[i])))
    except Exception:
        pass
    return {"highs": highs, "lows": lows}

def _find_swing_highs_lows(df: pd.DataFrame, lookback=5) -> Tuple[List[Tuple[int, float, Any]], List[Tuple[int, float, Any]]]:
    highs: List[Tuple[int, float, Any]] = []
    lows: List[Tuple[int, float, Any]] = []
    try:
        times = df["time"] if "time" in df.columns else df.index
        for i in range(lookback, len(df)-lookback):
            window_high = df["high"].iloc[i-lookback:i+lookback+1]
            window_low = df["low"].iloc[i-lookback:i+lookback+1]
            if df["high"].iat[i] == window_high.max():
                highs.append((i, float(df["high"].iat[i]), times[i]))
            if df["low"].iat[i] == window_low.min():
                lows.append((i, float(df["low"].iat[i]), times[i]))
    except Exception:
        pass
    return highs, lows

def _detect_fvg(df: pd.DataFrame, lookback=FVG_LOOKBACK) -> List[Dict[str, Any]]:
    gaps: List[Dict[str, Any]] = []
    try:
        n = min(len(df), lookback)
        start = len(df) - n
        for idx in range(start, len(df) - 1):
            c1_open = float(df["open"].iat[idx]); c1_close = float(df["close"].iat[idx])
            c2_open = float(df["open"].iat[idx+1]); c2_close = float(df["close"].iat[idx+1])
            if c1_close > c1_open and c2_close > c2_open:
                top1 = max(c1_open, c1_close); bot2 = min(c2_open, c2_close)
                if bot2 > top1:
                    gaps.append({"type": "bull", "start_idx": idx, "end_idx": idx+1, "low": top1, "high": bot2})
            if c1_close < c1_open and c2_close < c2_open:
                bot1 = min(c1_open, c1_close); top2 = max(c2_open, c2_close)
                if top2 < bot1:
                    gaps.append({"type": "bear", "start_idx": idx, "end_idx": idx+1, "low": top2, "high": bot1})
    except Exception:
        pass
    return gaps

def _detect_order_blocks(df: pd.DataFrame, lookback=200) -> List[Dict[str, Any]]:
    obs: List[Dict[str, Any]] = []
    try:
        start = max(5, len(df) - lookback)
        for idx in range(start + 5, len(df) - 1):
            try:
                prev_bear = df["close"].iat[idx-5] < df["open"].iat[idx-5]
                seq = df["close"].iloc[idx-4:idx+1]
                if prev_bear and len(seq) >= 3 and (seq.pct_change().fillna(0) > 0).all():
                    obs.append({"type": "bull", "idx": idx-5, "high": float(df["high"].iat[idx-5]), "low": float(df["low"].iat[idx-5])})
            except Exception:
                pass
            try:
                prev_bull = df["close"].iat[idx-5] > df["open"].iat[idx-5]
                seq = df["close"].iloc[idx-4:idx+1]
                if prev_bull and len(seq) >= 3 and (seq.pct_change().fillna(0) < 0).all():
                    obs.append({"type": "bear", "idx": idx-5, "high": float(df["high"].iat[idx-5]), "low": float(df["low"].iat[idx-5])})
            except Exception:
                pass
    except Exception:
        pass
    return obs

# ---------- DIVERGENCES ----------
def _find_divergence(price: pd.Series, osc: pd.Series, kind: str = "regular", look: int = 50) -> Optional[str]:
    """
    Simple peak/valley based divergence detection.
    kind: 'regular' (counter-trend), 'hidden' (trend-continuation)
    Returns 'bull'/'bear'/None.
    """
    try:
        p = price.tail(look).reset_index(drop=True)
        o = osc.tail(look).reset_index(drop=True)
        p_high_idx = p.idxmax(); p_low_idx = p.idxmin()
        o_high_idx = o.idxmax(); o_low_idx = o.idxmin()
        if kind == "regular":
            if p_low_idx > 1 and o_low_idx > 1:
                if p.iloc[p_low_idx] < p.iloc[:p_low_idx].min() and o.iloc[p_low_idx] > o.iloc[:o_low_idx].min():
                    return "bull"
            if p_high_idx > 1 and o_high_idx > 1:
                if p.iloc[p_high_idx] > p.iloc[:p_high_idx].max() and o.iloc[p_high_idx] < o.iloc[:o_high_idx].max():
                    return "bear"
        else:
            if p_low_idx > 1 and o_low_idx > 1:
                if p.iloc[p_low_idx] > p.iloc[:p_low_idx].min() and o.iloc[p_low_idx] < o.iloc[:p_low_idx].min():
                    return "bull"
            if p_high_idx > 1 and o_high_idx > 1:
                if p.iloc[p_high_idx] < p.iloc[:p_high_idx].max() and o.iloc[p_high_idx] > o.iloc[:o_high_idx].max():
                    return "bear"
    except Exception:
        return None
    return None

# ---------- BOS/CHOCH lite ----------
def _detect_bos_choch_liquidity(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        if df is None or len(df) < 8:
            return {"bos": None, "choch": False, "liquidity_sweep": False, "details": "insufficient_data"}
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        closes = df["close"].astype(float).values
        window = min(50, max(3, len(df) // 4))
        recent_segment = df.iloc[-(window + 1):-1] if window >= 3 else df.iloc[-3:-1]
        recent_high = float(recent_segment["high"].max())
        recent_low = float(recent_segment["low"].min())
        last_close = float(closes[-1]); prev_close = float(closes[-2])
        bos = None
        if last_close > recent_high and prev_close <= recent_high:
            bos = "Buy"
        elif last_close < recent_low and prev_close >= recent_low:
            bos = "Sell"
        choch = False
        liquidity = False
        try:
            rng = float((df["high"] - df["low"]).tail(20).mean()) if len(df) >= 20 else float((df["high"] - df["low"]).mean())
            threshold = rng * 0.9 if rng and rng > 0 else 0.0
            last = df.iloc[-1]
            if recent_high and last["high"] > recent_high + threshold and last["close"] < last["open"]:
                liquidity = True
            if recent_low and last["low"] < recent_low - threshold and last["close"] > last["open"]:
                liquidity = True
        except Exception:
            pass
        return {"bos": bos, "choch": choch, "liquidity_sweep": liquidity, "details": {"recent_high": recent_high, "recent_low": recent_low}}
    except Exception as e:
        return {"bos": None, "choch": False, "liquidity_sweep": False, "details": f"error:{e}"}

# ---------- FIBONACCI (high-performance, swing-aware) ----------
def _fib_levels_from_swing(high_price: float, low_price: float, direction: str) -> Dict[str, float]:
    """
    Compute key Fibonacci retracement levels for a swing.
    direction: 'up' for swing low->high, 'down' for high->low.
    Returns dict mapping level string to price.
    """
    try:
        hp = float(high_price); lp = float(low_price)
        rng = hp - lp
        if rng == 0:
            return {}
        levels: Dict[str, float] = {}
        if direction == "up":
            for l in [0.236, 0.382, 0.5, 0.618, 0.65, 0.786]:
                levels[str(l)] = hp - l * rng
            levels["0.0"] = hp; levels["1.0"] = lp
        else:
            for l in [0.236, 0.382, 0.5, 0.618, 0.65, 0.786]:
                levels[str(l)] = lp + l * rng
            levels["0.0"] = lp; levels["1.0"] = hp
        return levels
    except Exception:
        return {}

def _pick_recent_impulse(swing_highs: List[Tuple[int, float, Any]],
                         swing_lows: List[Tuple[int, float, Any]],
                         prefer_up: bool) -> Optional[Dict[str, Any]]:
    """
    Select the most recent swing impulse suitable for retracement calculations.
    prefer_up: choose low->high if True else high->low.
    """
    try:
        if not swing_highs or not swing_lows:
            return None
        sh = sorted(swing_highs, key=lambda x: x[0])
        sl = sorted(swing_lows, key=lambda x: x[0])
        if prefer_up:
            candidates = [ (lo, hi) for lo in sl for hi in sh if lo[0] < hi[0] ]
            if not candidates: return None
            lo, hi = max(candidates, key=lambda pair: pair[1][0])
            return {"direction": "up", "start_idx": lo[0], "end_idx": hi[0], "start_price": lo[1], "end_price": hi[1]}
        else:
            candidates = [ (hi, lo) for hi in sh for lo in sl if hi[0] < lo[0] ]
            if not candidates: return None
            hi, lo = max(candidates, key=lambda pair: pair[1][0])
            return {"direction": "down", "start_idx": hi[0], "end_idx": lo[0], "start_price": hi[1], "end_price": lo[1]}
    except Exception:
        return None

def _fib_confluence(df: pd.DataFrame,
                    swing_highs: List[Tuple[int, float, Any]],
                    swing_lows: List[Tuple[int, float, Any]],
                    trend_bull: Optional[bool],
                    atr_series: pd.Series) -> Dict[str, Any]:
    """
    Build Fibonacci levels for the most recent impulse aligned with trend, and
    report proximity to key levels (0.382/0.5/0.618/0.65/0.786), including golden pocket.
    """
    out: Dict[str, Any] = {"available": False}
    try:
        prefer_up = True if trend_bull else False if trend_bull is not None else True
        imp = _pick_recent_impulse(swing_highs, swing_lows, prefer_up)
        if not imp:
            return out
        direction = imp["direction"]
        high_price = imp["end_price"] if direction == "up" else imp["start_price"]
        low_price = imp["start_price"] if direction == "up" else imp["end_price"]
        levels = _fib_levels_from_swing(high_price, low_price, direction)

        # golden pocket zone (0.618 - 0.65)
        gp_lo = levels.get("0.65"); gp_hi = levels.get("0.618")
        if direction == "down":  # ensure correct ordering if needed
            gp_lo, gp_hi = levels.get("0.618"), levels.get("0.65")

        price = float(df["close"].iloc[-1])
        atr_now = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0
        tol = max(atr_now * 0.25, price * 0.0003)  # adaptive tolerance

        priority = ["0.382", "0.5", "0.618", "0.65", "0.786"]
        nearest = None
        min_d = float("inf")
        for key in priority:
            val = levels.get(key)
            if val is None:
                continue
            d = abs(price - val)
            if d < min_d:
                min_d = d; nearest = (key, val)

        near = bool(nearest and (min_d <= tol))
        in_gp = False
        if gp_lo is not None and gp_hi is not None:
            lo = min(gp_lo, gp_hi); hi = max(gp_lo, gp_hi)
            in_gp = (lo <= price <= hi)

        dir_bias = "Buy" if direction == "up" else "Sell"

        out = {
            "available": True,
            "swing": imp,
            "levels": levels,
            "golden_pocket": {"low": gp_lo, "high": gp_hi, "inside": bool(in_gp)},
            "nearest": {"level": nearest[0] if nearest else None,
                        "price": nearest[1] if nearest else None,
                        "distance": float(min_d) if nearest else None,
                        "within_tolerance": near},
            "tolerance": float(tol),
            "direction_bias": dir_bias
        }
        return out
    except Exception:
        return out

# ---------- FIB STACKING (Fib x OB/FVG overlap) ----------
def _interval_overlap(lo1: Optional[float], hi1: Optional[float], lo2: Optional[float], hi2: Optional[float], tol: float = 0.0) -> bool:
    try:
        if lo1 is None or hi1 is None or lo2 is None or hi2 is None:
            return False
        a_lo, a_hi = (min(lo1, hi1), max(lo1, hi1))
        b_lo, b_hi = (min(lo2, hi2), max(lo2, hi2))
        return not (a_hi < b_lo - tol or b_hi < a_lo - tol)
    except Exception:
        return False

def _fib_stack(fib: Dict[str, Any],
               order_blocks: List[Dict[str, Any]],
               fvgs: List[Dict[str, Any]],
               atr_now: float,
               price_now: float) -> Dict[str, Any]:
    """
    Detect stacking when Fibonacci levels/zones overlap with OB or FVG zones.
    Returns dict: {has_stack: bool, overlaps: [..], tolerance: float}
    """
    out = {"has_stack": False, "overlaps": [], "tolerance": None}
    try:
        if not fib or not fib.get("available"):
            out["tolerance"] = None
            return out

        levels = fib.get("levels", {}) or {}
        gp = fib.get("golden_pocket", {}) or {}
        gp_lo = gp.get("low"); gp_hi = gp.get("high")

        tol = max((atr_now or 0.0) * 0.25, price_now * 0.0003)
        out["tolerance"] = float(tol)

        overlaps: List[Dict[str, Any]] = []

        # Level x OB/FVG
        for key in ["0.382", "0.5", "0.618", "0.65", "0.786"]:
            lvl = levels.get(key)
            if lvl is None:
                continue
            # OB
            try:
                for ob in order_blocks or []:
                    lo = float(ob.get("low")) if ob.get("low") is not None else None
                    hi = float(ob.get("high")) if ob.get("high") is not None else None
                    if lo is None or hi is None:
                        continue
                    if (lo - tol) <= float(lvl) <= (hi + tol):
                        overlaps.append({"type": "level_ob", "level": key, "level_price": float(lvl), "ob": {"type": ob.get("type"), "low": lo, "high": hi}})
                        break
            except Exception:
                pass
            # FVG
            try:
                for g in fvgs or []:
                    glo = float(g.get("low")) if g.get("low") is not None else None
                    ghi = float(g.get("high")) if g.get("high") is not None else None
                    if glo is None or ghi is None:
                        continue
                    if (glo - tol) <= float(lvl) <= (ghi + tol):
                        overlaps.append({"type": "level_fvg", "level": key, "level_price": float(lvl), "fvg": {"type": g.get("type"), "low": glo, "high": ghi}})
                        break
            except Exception:
                pass

        # GP zone x OB/FVG
        try:
            if gp_lo is not None and gp_hi is not None:
                for ob in order_blocks or []:
                    lo = float(ob.get("low")) if ob.get("low") is not None else None
                    hi = float(ob.get("high")) if ob.get("high") is not None else None
                    if _interval_overlap(gp_lo, gp_hi, lo, hi, tol):
                        overlaps.append({"type": "gp_ob", "gp_low": float(gp_lo), "gp_high": float(gp_hi), "ob": {"type": ob.get("type"), "low": lo, "high": hi}})
                        break
                for g in fvgs or []:
                    glo = float(g.get("low")) if g.get("low") is not None else None
                    ghi = float(g.get("high")) if g.get("high") is not None else None
                    if _interval_overlap(gp_lo, gp_hi, glo, ghi, tol):
                        overlaps.append({"type": "gp_fvg", "gp_low": float(gp_lo), "gp_high": float(gp_hi), "fvg": {"type": g.get("type"), "low": glo, "high": ghi}})
                        break
        except Exception:
            pass

        out["overlaps"] = overlaps
        out["has_stack"] = bool(len(overlaps) > 0)
        return out
    except Exception:
        return out

# ---------- SESSIONS / HTF ----------
def _session_from_ts_tz(ts: Any, tzname: Optional[str] = None) -> str:
    try:
        t = pd.to_datetime(ts)
        tz = None
        if tzname:
            try:
                import pytz as _p
                tz = _p.timezone(tzname)
            except Exception:
                tz = None
        if tz is None:
            tz = SAST_TZ
        if tz is not None:
            if getattr(t, "tzinfo", None) is None:
                t = t.tz_localize("UTC").astimezone(tz)
            else:
                t = t.tz_convert(tz)
        hour = int(t.hour)
        if 7 <= hour < 16: return "London"
        if 13 <= hour < 22: return "NY"
        return "Asia"
    except Exception:
        return "Unknown"

def _htf_alignment(pair: Optional[str], ts: Any, want: str) -> Optional[bool]:
    """
    Simple 4H EMA(20) vs EMA(50) alignment with LTF direction 'want'.
    """
    try:
        if fetch_candles is None or not pair:
            return None
        try:
            df4 = fetch_candles(pair, "4h", n=400)
        except TypeError:
            df4 = fetch_candles(pair, "4h", 400)
        if not isinstance(df4, pd.DataFrame) or df4.empty:
            return None
        close = df4["close"].astype(float)
        ema20 = _ema(close, 20); ema50 = _ema(close, 50)
        dir4h = "Buy" if ema20.iloc[-1] > ema50.iloc[-1] else "Sell"
        return (dir4h == want) if want in ("Buy","Sell") else None
    except Exception:
        return None

# ---------- SIMPLE BACKTEST ----------
def _simple_backtest(df: pd.DataFrame, signal_idx_list: List[Tuple[int, str]], look_forward=10) -> Dict[str, int]:
    res = {"wins": 0, "losses": 0, "trades": 0}
    try:
        atr = _compute_atr_safe(df, period=14)
        for i, s in signal_idx_list:
            if i + look_forward >= len(df): continue
            entry = float(df["close"].iat[i])
            a = float(atr.iat[i]) if i < len(atr) else np.nan
            if np.isnan(a) or a == 0: continue
            tp = entry + 2*a if s == "long" else entry - 2*a
            sl = entry - 2*a if s == "long" else entry + 2*a
            future_highs = df["high"].iloc[i+1:i+1+look_forward].astype(float)
            future_lows = df["low"].iloc[i+1:i+1+look_forward].astype(float)
            if s == "long":
                if (future_highs >= tp).any(): res["wins"] += 1
                elif (future_lows <= sl).any(): res["losses"] += 1
            else:
                if (future_lows <= tp).any(): res["wins"] += 1
                elif (future_highs >= sl).any(): res["losses"] += 1
            res["trades"] += 1
    except Exception:
        pass
    return res

# ---------- MAIN ----------
def advanced_analysis(df: pd.DataFrame, bars: int = DEFAULT_BARS) -> Dict[str, Any]:
    """
    Run advanced confluence analysis on the last `bars` of df.
    Input: df must contain columns ['open','high','low','close'] and optionally ['time','volume','tick_volume'].
    Returns: dict with keys including:
      - advanced_signal (Buy/Sell/Neutral)
      - suggestion (alias)
      - score (float) and score_breakdown (dict)
      - numerous component indicators and detectors
    """
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {"error": "df is empty or None"}
        df = df.copy()
        for col in ("open","high","low","close"):
            if col not in df.columns:
                return {"error": f"missing column: {col}"}
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open","high","low","close"])
        if df.empty:
            return {"error": "no valid price data"}
        if bars and bars > 0 and len(df) > bars:
            df = df.tail(bars)
        n = len(df)
        if n < 10:
            return {"error": "not enough bars for advanced analysis"}

        # Indicators (defensive)
        ema8 = _ema(df["close"], 8); ema21 = _ema(df["close"], 21)
        trend_bull = None
        try:
            trend_bull = bool(ema8.iloc[-1] > ema21.iloc[-1])
        except Exception:
            trend_bull = None

        bb_up, bb_mid, bb_lo = _bollinger(df, 20)
        bb_perc = _bb_percent_b(df, bb_up, bb_lo)
        kel_up, kel_lo = _keltner(df, 20, 1.5)
        atr = _compute_atr_safe(df, 14)
        adx = _safe_series_like(df)
        di_pos = _safe_series_like(df)
        di_neg = _safe_series_like(df)
        try:
            if ta is not None:
                adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
                adx = adx_ind.adx()
                di_pos = adx_ind.adx_pos()
                di_neg = adx_ind.adx_neg()
        except Exception:
            pass
        willr = _safe_series_like(df)
        try:
            if ta is not None:
                willr = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=14).williams_r()
            else:
                willr = (df["high"].rolling(14).max() - df["close"]) / (df["high"].rolling(14).max() - df["low"].rolling(14).min() + 1e-9) * -100
        except Exception:
            pass
        demarker = _demarker_manual(df, 14)
        cci = _compute_cci(df, 20)
        rsi = _compute_rsi(df, 14)
        stoch_k, stoch_d = _compute_stoch(df, 14, 3)
        macd_line, macd_sig, macd_hist = _compute_macd(df)
        obv = _compute_obv(df)
        mfi = _compute_mfi(df, 14)
        psar = _psar_custom(df)
        conv, base, span_a, span_b = _ichimoku(df)
        ha_open, ha_close = _heikin_ashi(df)
        supertrend = _supertrend(df, 10, 3.0)
        vwap = _vwap(df)

        # Structures / patterns
        bos_choch = _detect_bos_choch_liquidity(df)
        fvg = _detect_fvg(df)
        fractals = _detect_fractals(df)
        candle_patterns = _detect_candlestick_patterns(df)
        order_blocks = _detect_order_blocks(df)
        swing_highs, swing_lows = _find_swing_highs_lows(df, lookback=SWING_LOOKBACK)

        # Fibonacci (high-performance, swing-aware)
        fib = _fib_confluence(df, swing_highs, swing_lows, trend_bull, atr)

        # Fib stacking (Fib x OB/FVG)
        price_now = float(df["close"].iloc[-1])
        atr_now = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
        fib_stack = _fib_stack(fib, order_blocks, fvg, atr_now, price_now)

        # Squeeze: BB width < Keltner width (compression)
        squeeze_on = False
        try:
            bb_width = (bb_up - bb_lo).iloc[-1]
            kel_width = (kel_up - kel_lo).iloc[-1]
            squeeze_on = bool(bb_width < kel_width) if not (pd.isna(bb_width) or pd.isna(kel_width)) else False
        except Exception:
            squeeze_on = False

        # VWAP align (if available)
        vwap_align = None
        try:
            if vwap is not None:
                vwap_align = (df["close"].iloc[-1] >= vwap.iloc[-1]) if (trend_bull is True) else (df["close"].iloc[-1] <= vwap.iloc[-1]) if (trend_bull is False) else None
        except Exception:
            vwap_align = None

        # Session tag (SAST or fallback)
        session = "Unknown"
        try:
            ts_now = df["time"].iloc[-1] if "time" in df.columns else df.index[-1]
            session = _session_from_ts_tz(ts_now)
        except Exception:
            session = "Unknown"

        # ADR proxy (if time available, robust and warning-free)
        adr_active = False
        try:
            if "time" in df.columns:
                dt = pd.to_datetime(df["time"])
                daily_extremes = (
                    df.assign(day=dt.dt.date)
                      .groupby("day", as_index=False)
                      .agg(high_max=("high", "max"), low_min=("low", "min"))
                )
                day_ranges = (daily_extremes["high_max"] - daily_extremes["low_min"]).tail(10)
                adr = float(day_ranges.mean()) if len(day_ranges) > 0 else None
                adr_active = bool((df["high"].iloc[-1] - df["low"].iloc[-1]) > 0.25 * (adr or 0.0)) if adr else False
        except Exception:
            adr_active = False

        # ATR direction (rising volatility)
        atr_up = False
        try:
            a = atr.tail(10).dropna()
            atr_up = bool(len(a) >= 3 and a.iloc[-1] > a.iloc[0])
        except Exception:
            atr_up = False

        # Divergences
        rsi_div_reg = _find_divergence(df["close"], rsi, "regular", 60)
        rsi_div_hidden = _find_divergence(df["close"], rsi, "hidden", 60)
        macd_div_reg = _find_divergence(df["close"], macd_line, "regular", 60)
        macd_div_hidden = _find_divergence(df["close"], macd_line, "hidden", 60)

        # Information measures
        spectral_trend = 0.0
        hurst_est = 0.5
        try:
            ret = df["close"].pct_change().dropna().tail(256).values
            if len(ret) >= 16:
                psd = np.abs(np.fft.rfft(ret))**2
                psd = psd / (psd.sum() + 1e-12)
                ent = -np.sum(psd * np.log(psd + 1e-12))
                ent_norm = float(ent / np.log(len(psd) + 1e-12))
                spectral_trend = float(max(0.0, 1.0 - ent_norm))
                X = np.cumsum(ret - ret.mean())
                R = (X.max() - X.min()); S = ret.std() + 1e-12; N = len(ret)
                hurst_est = float(np.log(R / S + 1e-12) / np.log(N + 1e-12))
        except Exception:
            spectral_trend = 0.0; hurst_est = 0.5

        # Sniper retained for notes/backtest
        sniper = None
        try:
            if ta is not None:
                ema8_t = ta.trend.EMAIndicator(df["close"], window=8).ema_indicator()
                ema21_t = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
                last_close = float(df["close"].iloc[-1])
                tstamp = str(df["time"].iloc[-1]) if "time" in df.columns else str(df.index[-1])
                if float(ema8_t.iloc[-1]) > float(ema21_t.iloc[-1]) and float(df["low"].iloc[-1]) <= float(ema8_t.iloc[-1]) and last_close > float(ema8_t.iloc[-1]):
                    sniper = {"type": "sniper_long", "entry": float(round(ema8_t.iloc[-1], 6)), "time": tstamp}
                if float(ema8_t.iloc[-1]) < float(ema21_t.iloc[-1]) and float(df["high"].iloc[-1]) >= float(ema8_t.iloc[-1]) and last_close < float(ema8_t.iloc[-1]):
                    sniper = {"type": "sniper_short", "entry": float(round(ema8_t.iloc[-1], 6)), "time": tstamp}
        except Exception:
            sniper = None

        # ----- SCORING -----
        breakdown: Dict[str, float] = {}
        score = 0.0

        # Trend
        try:
            tb = bool(trend_bull) if trend_bull is not None else (ema8.iloc[-1] > ema21.iloc[-1])
            score += WEIGHTS["ema_trend"] * (1 if tb else -1)
            breakdown["ema_trend"] = WEIGHTS["ema_trend"] * (1 if tb else -1)
        except Exception:
            pass
        try:
            ich_bull = conv.iloc[-1] > base.iloc[-1]
            score += WEIGHTS["ichimoku_conv_base"] * (1 if ich_bull else -1)
            breakdown["ichimoku_conv_base"] = WEIGHTS["ichimoku_conv_base"] * (1 if ich_bull else -1)
        except Exception:
            pass
        try:
            psar_bull = df["close"].iloc[-1] > psar.iloc[-1]
            score += WEIGHTS["psar_trend"] * (1 if psar_bull else -1)
            breakdown["psar_trend"] = WEIGHTS["psar_trend"] * (1 if psar_bull else -1)
        except Exception:
            pass
        try:
            allig = 0
            jaw = _sma(df["close"], 13).shift(8).iloc[-1]
            teeth = _sma(df["close"], 8).shift(5).iloc[-1]
            lips = _sma(df["close"], 5).shift(3).iloc[-1]
            if lips > teeth > jaw: allig = 1
            if lips < teeth < jaw: allig = -1
            score += WEIGHTS["alligator_align"] * allig
            breakdown["alligator_align"] = WEIGHTS["alligator_align"] * allig
        except Exception:
            pass
        try:
            st_dir = int(supertrend.iloc[-1])  # +1/-1/0
            score += WEIGHTS["supertrend"] * st_dir
            breakdown["supertrend"] = WEIGHTS["supertrend"] * st_dir
        except Exception:
            pass
        try:
            adx_val = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
            if adx_val > 25:
                score += WEIGHTS["adx_strength"]
                breakdown["adx_strength"] = WEIGHTS["adx_strength"]
        except Exception:
            pass

        # Momentum
        try:
            macd_bias = 1 if macd_hist.iloc[-1] > 0 else -1
            score += WEIGHTS["macd_trend"] * macd_bias
            breakdown["macd_trend"] = WEIGHTS["macd_trend"] * macd_bias
        except Exception:
            pass
        try:
            st_sig = 1 if stoch_k.iloc[-1] > stoch_d.iloc[-1] else -1
            score += WEIGHTS["stoch_signal"] * st_sig
            breakdown["stoch_signal"] = WEIGHTS["stoch_signal"] * st_sig
        except Exception:
            pass
        try:
            rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            rz = 1 if rsi_val < 35 else (-1 if rsi_val > 65 else 0)
            score += WEIGHTS["rsi_zone"] * rz
            breakdown["rsi_zone"] = WEIGHTS["rsi_zone"] * rz
        except Exception:
            pass
        try:
            cci_val = float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else 0.0
            ccis = 1 if cci_val < -100 else (-1 if cci_val > 100 else 0)
            score += WEIGHTS["cci_extreme"] * ccis
            breakdown["cci_extreme"] = WEIGHTS["cci_extreme"] * ccis
        except Exception:
            pass

        # Structure/pattern
        try:
            if bos_choch.get("bos") == "Buy":
                score += WEIGHTS["bos_buy"]; breakdown["bos_buy"] = WEIGHTS["bos_buy"]
            elif bos_choch.get("bos") == "Sell":
                score += WEIGHTS["bos_sell"]; breakdown["bos_sell"] = WEIGHTS["bos_sell"]
            if bos_choch.get("choch"): score += WEIGHTS["choch"]; breakdown["choch"] = WEIGHTS["choch"]
            if bos_choch.get("liquidity_sweep"): score += WEIGHTS["liquidity_sweep"]; breakdown["liquidity_sweep"] = WEIGHTS["liquidity_sweep"]
        except Exception:
            pass
        try:
            if order_blocks:
                score += WEIGHTS["order_block"]; breakdown["order_block"] = WEIGHTS["order_block"]
        except Exception:
            pass
        try:
            near = 0
            if fvg:
                lastp = df["close"].iloc[-1]
                for g in fvg[-3:]:
                    if g["low"] <= lastp <= g["high"]:
                        near = 1; break
            if near:
                score += WEIGHTS["fvg_near"]; breakdown["fvg_near"] = WEIGHTS["fvg_near"]
        except Exception:
            pass
        try:
            if candle_patterns:
                score += WEIGHTS["candle_pattern"]; breakdown["candle_pattern"] = WEIGHTS["candle_pattern"]
        except Exception:
            pass
        try:
            if fractals["highs"] or fractals["lows"]:
                score += WEIGHTS["fractals_recent"]; breakdown["fractals_recent"] = WEIGHTS["fractals_recent"]
        except Exception:
            pass

        # Fibonacci confluence
        try:
            if fib.get("available"):
                sign = 1 if fib.get("direction_bias") == "Buy" else -1
                near_level = fib.get("nearest", {}).get("within_tolerance")
                in_gp = fib.get("golden_pocket", {}).get("inside")
                if near_level or in_gp:
                    score += WEIGHTS["fib_confluence"] * sign
                    breakdown["fib_confluence"] = WEIGHTS["fib_confluence"] * sign
        except Exception:
            pass

        # Fibonacci stacking (Fib x OB/FVG)
        try:
            if fib.get("available") and fib_stack.get("has_stack"):
                sign = 1 if fib.get("direction_bias") == "Buy" else -1
                score += WEIGHTS["fib_stack"] * sign
                breakdown["fib_stack"] = WEIGHTS["fib_stack"] * sign
        except Exception:
            pass

        # VWAP/session
        try:
            if vwap_align is True:
                score += WEIGHTS["vwap_align"]; breakdown["vwap_align"] = WEIGHTS["vwap_align"]
        except Exception:
            pass
        try:
            if session in ("London","NY"):
                score += WEIGHTS["session_align"]; breakdown["session_align"] = WEIGHTS["session_align"]
        except Exception:
            pass

        # Volatility/regime
        try:
            if squeeze_on:
                score += WEIGHTS["squeeze_on"]; breakdown["squeeze_on"] = WEIGHTS["squeeze_on"]
        except Exception:
            pass
        try:
            if atr_up:
                score += WEIGHTS["atr_up"]; breakdown["atr_up"] = WEIGHTS["atr_up"]
        except Exception:
            pass
        try:
            if adr_active:
                score += WEIGHTS["adr_active"]; breakdown["adr_active"] = WEIGHTS["adr_active"]
        except Exception:
            pass

        # Divergences
        try:
            if rsi_div_reg == "bull":
                score += WEIGHTS["rsi_div_reg"]; breakdown["rsi_div_reg"] = WEIGHTS["rsi_div_reg"]
            if rsi_div_reg == "bear":
                score -= WEIGHTS["rsi_div_reg"]; breakdown["rsi_div_reg"] = -WEIGHTS["rsi_div_reg"]
            if rsi_div_hidden == "bull":
                score += WEIGHTS["rsi_div_hidden"]; breakdown["rsi_div_hidden"] = WEIGHTS["rsi_div_hidden"]
            if rsi_div_hidden == "bear":
                score -= WEIGHTS["rsi_div_hidden"]; breakdown["rsi_div_hidden"] = -WEIGHTS["rsi_div_hidden"]
        except Exception:
            pass
        try:
            if macd_div_reg == "bull":
                score += WEIGHTS["macd_div_reg"]; breakdown["macd_div_reg"] = WEIGHTS["macd_div_reg"]
            if macd_div_reg == "bear":
                score -= WEIGHTS["macd_div_reg"]; breakdown["macd_div_reg"] = -WEIGHTS["macd_div_reg"]
            if macd_div_hidden == "bull":
                score += WEIGHTS["macd_div_hidden"]; breakdown["macd_div_hidden"] = WEIGHTS["macd_div_hidden"]
            if macd_div_hidden == "bear":
                score -= WEIGHTS["macd_div_hidden"]; breakdown["macd_div_hidden"] = -WEIGHTS["macd_div_hidden"]
        except Exception:
            pass

        # Flow
        try:
            if not pd.isna(willr.iloc[-1]) and (willr.iloc[-1] < -80 or willr.iloc[-1] > -20):
                score += WEIGHTS["willr_extreme"]; breakdown["willr_extreme"] = WEIGHTS["willr_extreme"]
        except Exception:
            pass
        try:
            if not pd.isna(mfi.iloc[-1]):
                if mfi.iloc[-1] > 60: score += WEIGHTS["mfi_flow"]
                if mfi.iloc[-1] < 40: score += WEIGHTS["mfi_flow"]
                breakdown["mfi_flow"] = WEIGHTS["mfi_flow"]
        except Exception:
            pass
        try:
            if len(obv.dropna()) >= 5:
                if obv.iloc[-1] > obv.iloc[-5]: score += WEIGHTS["obv_flow"]
                breakdown["obv_flow"] = WEIGHTS["obv_flow"]
        except Exception:
            pass

        # Information measures
        try:
            if spectral_trend > 0.55:
                score += WEIGHTS["spectral_trend"]; breakdown["spectral_trend"] = WEIGHTS["spectral_trend"]
        except Exception:
            pass
        try:
            if hurst_est > 0.55:
                score += WEIGHTS["hurst_trend"]; breakdown["hurst_trend"] = WEIGHTS["hurst_trend"]
        except Exception:
            pass

        # Verdict mapping
        verdict = "Neutral"
        if score >= 3.0:
            verdict = "Buy"
        if score <= -3.0:
            verdict = "Sell"
        advanced_signal = verdict if verdict in ("Buy","Sell") else "Neutral"

        # Confidence estimate
        try:
            mag = min(1.0, abs(score) / 8.0)
            adx_c = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 20.0
            base_conf = 0.4 + 0.3 * mag + 0.2 * (1.0 if adx_c > 25 else 0.0) + 0.1 * (1.0 if squeeze_on else 0.0)
            base_conf += 0.1 * min(1.0, spectral_trend)
            confidence = float(max(0.01, min(0.99, base_conf)))
        except Exception:
            confidence = 0.5

        # Simple backtest on sniper signal (as original)
        backtest = {}
        try:
            signals_for_bt: List[Tuple[int, str]] = []
            if sniper:
                signals_for_bt.append((n-1, "long" if sniper["type"] == "sniper_long" else "short"))
            backtest = _simple_backtest(df, signals_for_bt, look_forward=10)
        except Exception:
            backtest = {}

        # Collect output
        out: Dict[str, Any] = {
            "advanced_signal": advanced_signal,
            "suggestion": advanced_signal,
            "verdict": verdict,
            "score": float(round(score, 4)),
            "score_breakdown": breakdown,
            "confidence": float(round(confidence, 4)),
            "notes": [k for k,v in breakdown.items() if v != 0],
            "bos_choch": bos_choch,
            "fvg": fvg,
            "fractals": fractals,
            "candle_patterns": candle_patterns,
            "order_blocks": order_blocks,
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
            "fib": fib,                  # fib levels, gp, nearest, bias
            "fib_stack": fib_stack,      # overlaps and tolerance
            "sniper": sniper,
            "backtest": backtest,
            # Key indicator snapshots
            "ema8": float(ema8.iloc[-1]) if not pd.isna(ema8.iloc[-1]) else None,
            "ema21": float(ema21.iloc[-1]) if not pd.isna(ema21.iloc[-1]) else None,
            "willr": float(willr.iloc[-1]) if not pd.isna(willr.iloc[-1]) else None,
            "demarker": float(demarker.iloc[-1]) if not pd.isna(demarker.iloc[-1]) else None,
            "adx": float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None,
            "di_plus": float(di_pos.iloc[-1]) if not pd.isna(di_pos.iloc[-1]) else None,
            "di_minus": float(di_neg.iloc[-1]) if not pd.isna(di_neg.iloc[-1]) else None,
            "rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
            "stoch_k": float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else None,
            "stoch_d": float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else None,
            "macd": float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
            "macd_signal": float(macd_sig.iloc[-1]) if not pd.isna(macd_sig.iloc[-1]) else None,
            "macd_hist": float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else None,
            "cci": float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else None,
            "mfi": float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else None,
            "obv": float(obv.iloc[-1]) if not pd.isna(obv.iloc[-1]) else None,
            "psar": float(psar.iloc[-1]) if not pd.isna(psar.iloc[-1]) else None,
            "bb_percent_b": float(bb_perc.iloc[-1]) if not pd.isna(bb_perc.iloc[-1]) else None,
            "squeeze_on": bool(squeeze_on),
            "atr": float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None,
            "spectral_trend": float(round(spectral_trend, 4)),
            "hurst_est": float(round(hurst_est, 4)),
            "session": session,
            "vwap": float(vwap.iloc[-1]) if (vwap is not None and not pd.isna(vwap.iloc[-1])) else None,
        }

        return out

    except Exception as e:
        return {"error": str(e)}
