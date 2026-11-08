"""
Feature engineering for market anomaly & volatility.

This module is imported by BOTH the offline training script and the
live inference loop. This guarantees we do not suffer from
"train/serve skew" (where features are calculated differently in
training vs production).
"""

import pandas as pd
import numpy as np

# List of columns we will feed into the ML models
FEATURE_COLUMNS = [
    "return_1",
    "return_5",
    "volatility_5",
    "volatility_20",
    "volume_zscore",
    "price_vs_mean",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with 'price' and 'volume', calculate our ML features.
    
    CRITICAL RULE: We only use `.shift()` or rolling windows that look
    BACKWARDS. We never use future data to calculate current features.
    """
    out = df.copy()

    # 1. Log returns: ln(price_t / price_t-1)
    # Log returns are preferred in finance because they are time-additive
    # and better behaved statistically than simple percentage returns.
    out["return_1"] = np.log(out["price"] / out["price"].shift(1))
    
    # 2. 5-tick return (captures slightly longer momentum)
    out["return_5"] = np.log(out["price"] / out["price"].shift(5))

    # 3. Rolling Volatility (standard deviation of 1-tick returns)
    out["volatility_5"] = out["return_1"].rolling(window=5).std()
    out["volatility_20"] = out["return_1"].rolling(window=20).std()

    # 4. Volume Z-Score: How unusual is the current volume?
    # (volume - rolling_mean) / rolling_std
    vol_mean = out["volume"].rolling(window=20).mean()
    vol_std = out["volume"].rolling(window=20).std()
    out["volume_zscore"] = (out["volume"] - vol_mean) / (vol_std + 1e-8) # 1e-8 prevents div-by-zero

    # 5. Price vs. 20-tick Mean: How far has price strayed from its average?
    price_mean_20 = out["price"].rolling(window=20).mean()
    out["price_vs_mean"] = (out["price"] - price_mean_20) / (price_mean_20 + 1e-8)

    return out