"""
Market Psychology Engine

This module provides tools to quantify the market's character or "personality".
Its primary function is to calculate the Hurst Exponent, a measure of a time
series' memory and predictability.
"""
import numpy as np
import pandas as pd
from typing import Optional

def calculate_hurst(series: pd.Series, max_lags: int = 100) -> Optional[float]:
    """
    Calculates the Hurst Exponent for a given time series.

    H > 0.5: The series is trending (persistent).
    H < 0.5: The series is mean-reverting (anti-persistent).
    H = 0.5: The series is a geometric random walk.

    Returns: A float between 0 and 1, or None if calculation fails.
    """
    if not isinstance(series, pd.Series) or series.empty or len(series) < 100:
        return None
    
    try:
        # We need to use log-prices for this calculation
        log_series = np.log(series.astype(float).replace(0, np.nan).dropna())
        if len(log_series) < 100:
            return None

        tau = []
        lagvec = []
        
        # Build lags vector
        lags = range(2, min(len(log_series)-1, max_lags))

        for lag in lags:
            # Calculate the variance of the lagged differences
            pp = np.subtract(log_series[lag:].values, log_series[:-lag].values)
            if len(pp) > 0:
                tau.append(np.sqrt(np.std(pp)))
                lagvec.append(lag)

        if not tau:
            return None

        # Use a polyfit on the log-log plot to find the slope (alpha)
        m = np.polyfit(np.log(lagvec), np.log(tau), 1)
        
        # Hurst Exponent is the slope
        hurst = m[0]
        
        return max(0.0, min(1.0, hurst)) # Clamp the value between 0 and 1
    
    except Exception:
        return None
