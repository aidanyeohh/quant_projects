"""Bulk daily price-volume download for the recovered ASX ticker pool.

Prices are stored as one wide parquet file per field (dates on the index,
tickers on the columns) because every downstream calculation in this project
is a cross-sectional operation over that shape.

Both raw and dividend-adjusted closes are kept. Adjusted closes drive returns;
raw closes drive the price floor and the turnover figure, because a dividend
adjustment rescales history and would otherwise understate the traded value of
older bars.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR

FIELDS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def _panel_path(field: str, directory: Path = RAW_DIR) -> Path:
    return directory / f"{field}.parquet"


def download_daily(
    tickers: list[str],
    start: str = "2005-01-01",
    end: str | None = None,
    chunk_size: int = 100,
    directory: Path = RAW_DIR,
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for ``tickers`` and cache it as wide parquet panels.

    Chunks are merged into the cache after each request, so an interrupted run
    resumes from what is already on disk rather than re-downloading it.
    """
    import yfinance as yf

    cached: dict[str, pd.DataFrame] = {}
    if resume:
        for field in FIELDS.values():
            path = _panel_path(field, directory)
            if path.exists():
                cached[field] = pd.read_parquet(path)

    have = set(cached.get("close", pd.DataFrame()).columns)
    remaining = [t for t in tickers if t not in have]
    if verbose:
        print(f"downloading {len(remaining)} of {len(tickers)} tickers")

    for start_idx in range(0, len(remaining), chunk_size):
        batch = remaining[start_idx : start_idx + chunk_size]
        raw = yf.download(
            batch,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=True,
            group_by="column",
        )
        if raw is None or raw.empty:
            continue
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_product([raw.columns, [batch[0]]])

        for yahoo_field, name in FIELDS.items():
            if yahoo_field not in raw.columns.get_level_values(0):
                continue
            block = raw[yahoo_field].dropna(how="all", axis=1)
            if block.empty:
                continue
            cached[name] = (
                block
                if name not in cached
                else cached[name].join(
                    block[[c for c in block.columns if c not in cached[name].columns]],
                    how="outer",
                )
            )

        for name, frame in cached.items():
            frame.sort_index().to_parquet(_panel_path(name, directory))

        if verbose:
            width = cached.get("close", pd.DataFrame()).shape[1]
            print(f"  {start_idx + len(batch):>6}/{len(remaining)} requested, "
                  f"{width} tickers stored")

    return {name: frame.sort_index() for name, frame in cached.items()}


def load_panels(directory: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Read cached price panels from disk."""
    panels: dict[str, pd.DataFrame] = {}
    for name in FIELDS.values():
        path = _panel_path(name, directory)
        if path.exists():
            panels[name] = pd.read_parquet(path).sort_index()
    if "close" not in panels:
        raise FileNotFoundError(
            f"no price panels in {directory}; run scripts/02_download_prices.py first"
        )
    return panels


def download_benchmark(
    symbols: tuple[str, ...],
    start: str = "2005-01-01",
    end: str | None = None,
    directory: Path = PROCESSED_DIR,
) -> pd.Series:
    """Daily total returns of the first benchmark symbol that has data.

    ``^AXKO`` (S&P/ASX 300) is preferred; the fallbacks exist because Yahoo's
    coverage of the smaller Australian index tickers is patchy over long
    histories.
    """
    import yfinance as yf

    for symbol in symbols:
        frame = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if frame is None or frame.empty:
            continue
        close = frame["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        returns = close.pct_change().rename(symbol)
        returns.to_frame().to_parquet(directory / "benchmark.parquet")
        return returns
    raise RuntimeError(f"no benchmark data for any of {symbols}")


def load_benchmark(directory: Path = PROCESSED_DIR) -> pd.Series:
    frame = pd.read_parquet(directory / "benchmark.parquet")
    return frame.iloc[:, 0]
