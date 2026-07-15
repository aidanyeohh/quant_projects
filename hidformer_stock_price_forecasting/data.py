"""
Data pipeline for the Hidformer stock-forecasting experiments (TensorFlow).

Mirrors Section 5.1 / 4.1 of the paper:
  - 6 "Consumer Goods" sector tickers, Yahoo Finance daily OHLCV(+AdjClose).
  - Min-max normalization over the whole sample, per Eq. (1).
  - Sliding window: use T_x=128 past days to predict the next T_y=128
    close prices.
  - Train/validation split: first 95% for training, last 5% for validation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import tensorflow as tf

TICKERS = ["AAPL", "KO", "MO", "PEP", "PG", "TM"]  # Table 1 (Consumer Goods sector)
FEATURE_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def download_ticker(symbol: str, start: str = "1980-12-12", end: str = "2023-12-29") -> pd.DataFrame:
    """Downloads daily OHLCV data from Yahoo Finance for one ticker."""
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[FEATURE_COLS].dropna()
    return df


def minmax_normalize(arr: np.ndarray, stats: dict | None = None):
    """Eq. (1): x_hat = (x - x_min) / (x_max - x_min), computed jointly
    across all price dimensions (Open/High/Low/Close/AdjClose) and
    separately for Volume, as described in Section 4.1."""
    arr = arr.copy().astype(np.float64)
    price_cols = slice(0, 5)   # Open, High, Low, Close, Adj Close
    vol_col = 5                # Volume

    if stats is None:
        price_min, price_max = arr[:, price_cols].min(), arr[:, price_cols].max()
        vol_min, vol_max = arr[:, vol_col].min(), arr[:, vol_col].max()
        stats = {"price_min": price_min, "price_max": price_max,
                 "vol_min": vol_min, "vol_max": vol_max}

    arr[:, price_cols] = (arr[:, price_cols] - stats["price_min"]) / (
        stats["price_max"] - stats["price_min"] + 1e-12)
    arr[:, vol_col] = (arr[:, vol_col] - stats["vol_min"]) / (
        stats["vol_max"] - stats["vol_min"] + 1e-12)
    return arr, stats


def make_windows(series: np.ndarray, seq_len: int = 128, pred_len: int = 128,
                  close_col: int = 3, stride: int = 1):
    """Builds (X, y) sliding-window arrays from a [T, C] series.
    X: [N, seq_len, C]   y: [N, pred_len]"""
    series = series.astype(np.float32)
    max_start = len(series) - seq_len - pred_len
    starts = list(range(0, max(max_start, 0) + 1, stride))
    X = np.stack([series[s: s + seq_len] for s in starts]) if starts else \
        np.empty((0, seq_len, series.shape[1]), dtype=np.float32)
    y = np.stack([series[s + seq_len: s + seq_len + pred_len, close_col] for s in starts]) if starts else \
        np.empty((0, pred_len), dtype=np.float32)
    return X, y


def make_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = False,
                     drop_remainder: bool = False) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(X), 1), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(symbol: str, seq_len: int = 128, pred_len: int = 128,
                    train_frac: float = 0.95, start: str = "1980-12-12",
                    end: str = "2023-12-29"):
    """Downloads one ticker and returns (X_train, y_train, X_val, y_val,
    norm_stats, raw_df) following the paper's 95%/5% chronological split."""
    df = download_ticker(symbol, start=start, end=end)
    arr = df[FEATURE_COLS].values
    split_idx = int(len(arr) * train_frac)

    train_raw = arr[:split_idx]
    # normalize using statistics from the training portion only (avoids
    # look-ahead bias), then apply the same stats to the validation portion
    train_norm, stats = minmax_normalize(train_raw)
    val_norm, _ = minmax_normalize(arr[split_idx - seq_len:], stats=stats)

    X_train, y_train = make_windows(train_norm, seq_len, pred_len)
    X_val, y_val = make_windows(val_norm, seq_len, pred_len)
    return X_train, y_train, X_val, y_val, stats, df
