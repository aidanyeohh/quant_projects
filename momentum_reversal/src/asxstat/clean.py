"""Turning raw Yahoo panels into a research-grade daily dataset.

Three things are being defended against here.

Bad ticks.  Yahoo's ASX history contains occasional unadjusted split bars and
stray zero prices. Left alone these appear to a reversal signal as a huge
one-day move followed by a huge bounce, which is exactly the pattern the
strategy is looking for, so they manufacture fake alpha.

Non-trading days.  A halted or simply untraded microcap prints the previous
close with zero volume. Treating that as a real observation understates
volatility and makes a stale price look like a fresh one.

Delisting.  A dead company's column simply stops. The last trade date is
recorded per ticker so the backtester can force an exit and so that
`survivorship` can quantify what the missing final return is worth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# A single-day move larger than this is treated as a data error rather than a
# real return. Genuine ASX single-day moves beyond +/- 80% are rare enough, and
# concentrated enough in illiquid names, that excluding them costs little.
MAX_ABS_DAILY_RETURN = 0.8


@dataclass
class Dataset:
    """Everything the signals and the backtester need, aligned on one calendar."""

    close: pd.DataFrame          # raw close, AUD
    adj_close: pd.DataFrame      # dividend and split adjusted
    volume: pd.DataFrame         # shares
    returns: pd.DataFrame        # daily simple total return
    dollar_volume: pd.DataFrame  # AUD traded per day
    traded: pd.DataFrame         # bool: a real trade printed that day
    last_trade: pd.Series        # last date each ticker traded

    @property
    def tickers(self) -> pd.Index:
        return self.close.columns

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index


def _mask_untraded(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """True where a genuine trade printed.

    Zero or missing volume means no trade, whatever the price column says.
    """
    return close.notna() & (volume.fillna(0) > 0)


def build_dataset(
    panels: dict[str, pd.DataFrame],
    start: str | None = None,
    end: str | None = None,
) -> Dataset:
    """Assemble a cleaned :class:`Dataset` from raw download panels."""
    close = panels["close"].copy()
    adj_close = panels.get("adj_close", close).copy()
    volume = panels["volume"].reindex_like(close)

    # yfinance omits 'Adj Close' entirely for some tickers; fall back to close
    # so those names keep a price-return series rather than dropping out.
    adj_close = adj_close.reindex_like(close)
    adj_close = adj_close.where(adj_close.notna(), close)

    if start is not None:
        close = close.loc[start:]
        adj_close = adj_close.loc[start:]
        volume = volume.loc[start:]
    if end is not None:
        close = close.loc[:end]
        adj_close = adj_close.loc[:end]
        volume = volume.loc[:end]

    close = close.replace(0.0, np.nan)
    adj_close = adj_close.replace(0.0, np.nan)

    traded = _mask_untraded(close, volume)

    # Returns are computed on the adjusted series but only recognised where a
    # trade actually printed, so a stale carried-forward price cannot create a
    # zero return that dilutes volatility estimates.
    returns = adj_close.pct_change(fill_method=None)
    returns = returns.where(traded & traded.shift(1))
    returns = returns.where(returns.abs() <= MAX_ABS_DAILY_RETURN)

    dollar_volume = (close * volume).where(traded)

    last_trade = traded.apply(
        lambda col: col[col].index.max() if col.any() else pd.NaT
    )

    return Dataset(
        close=close,
        adj_close=adj_close,
        volume=volume,
        returns=returns,
        dollar_volume=dollar_volume,
        traded=traded,
        last_trade=last_trade,
    )


def delisting_report(data: Dataset, tolerance_days: int = 10) -> pd.DataFrame:
    """Classify each ticker as still trading or delisted, with its lifespan.

    A name whose last trade is more than ``tolerance_days`` before the end of
    the sample is treated as delisted. That covers takeovers, mergers, schemes
    of arrangement and administrations alike; Yahoo does not distinguish them,
    which is the reason for the stress test in :mod:`asxstat.survivorship`.
    """
    sample_end = data.dates.max()
    cutoff = sample_end - pd.Timedelta(int(tolerance_days), "D")

    first_trade = data.traded.apply(
        lambda col: col[col].index.min() if col.any() else pd.NaT
    )

    report = pd.DataFrame(
        {
            "first_trade": first_trade,
            "last_trade": data.last_trade,
            "n_trading_days": data.traded.sum(),
        }
    )
    report["delisted"] = report["last_trade"] < cutoff
    report["years_listed"] = (
        (report["last_trade"] - report["first_trade"]).dt.days / 365.25
    )
    return report.sort_values("last_trade")
