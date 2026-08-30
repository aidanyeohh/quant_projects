"""Building an ASX ticker pool that includes companies which no longer exist.

The survivorship problem
------------------------
Any list of "the ASX 300" that can be downloaded today contains only companies
that survived to today. Backtesting momentum on that list is close to
guaranteed to flatter the result: the losers the strategy shorted or avoided
are the ones that later went to zero and dropped out of the list, so they are
silently excluded from the sample.

Free point-in-time index membership for the ASX does not exist. The workaround
used here is to stop relying on index membership altogether:

1.  Enumerate the ASX ticker code space directly. Almost every ASX ordinary
    share has a three-letter code, so AAA through ZZZ is 17,576 candidates, a
    set that is independent of survival. Yahoo continues to serve price history
    for a large share of delisted Australian tickers, so probing that space
    recovers dead companies that a constituent list would never mention.

2.  Rebuild membership each month from trailing liquidity computed only from
    data available at that date (see :func:`point_in_time_universe`).

The recovered pool is still not complete: Yahoo drops some very old delistings,
and the ASX recycles codes between companies. The `survivorship` module
measures what is left rather than assuming it away.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ASX_SUFFIX, RAW_DIR, UniverseConfig

PROBE_PATH = RAW_DIR / "ticker_probe.parquet"


def candidate_codes(lengths: tuple[int, ...] = (3,)) -> list[str]:
    """Every alphabetic ASX code of the given lengths, as Yahoo symbols.

    Three letters covers the overwhelming majority of ASX ordinary shares.
    Four-letter codes exist but are dominated by ETFs, warrants and preference
    lines that are not the target of an equity stat-arb book.
    """
    codes: list[str] = []
    for length in lengths:
        codes.extend(
            "".join(letters) + ASX_SUFFIX
            for letters in product(string.ascii_uppercase, repeat=length)
        )
    return codes


@dataclass
class ProbeResult:
    """Summary of what price history exists for one candidate ticker."""

    ticker: str
    n_months: int
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None
    median_close: float


def _summarise_probe(frame: pd.DataFrame, tickers: list[str]) -> list[ProbeResult]:
    """Reduce a wide monthly-close frame to one row per ticker."""
    results: list[ProbeResult] = []
    for ticker in tickers:
        if ticker not in frame.columns:
            results.append(ProbeResult(ticker, 0, None, None, np.nan))
            continue
        series = frame[ticker].dropna()
        if series.empty:
            results.append(ProbeResult(ticker, 0, None, None, np.nan))
            continue
        results.append(
            ProbeResult(
                ticker=ticker,
                n_months=int(series.shape[0]),
                first_date=series.index[0],
                last_date=series.index[-1],
                median_close=float(series.median()),
            )
        )
    return results


# A ticker that is certain to have data. Its presence in a chunk's response is
# the test of whether Yahoo answered the request or throttled it. Without this,
# a throttled chunk is indistinguishable from a chunk of codes that never
# existed, and the scan silently records hundreds of false delistings.
CANARY = "ANZ.AX"


def _download_chunk(batch: list[str], canary: str = CANARY) -> pd.DataFrame | None:
    """One monthly-bar request, returning the wide close frame or None."""
    import yfinance as yf

    request = batch if canary in batch else [*batch, canary]
    raw = yf.download(
        request,
        period="max",
        interval="1mo",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw is None or raw.empty:
        return None
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(request[0])
    return closes


def _chunk_with_retry(
    batch: list[str],
    attempts: int = 4,
    backoff: float = 20.0,
    canary: str = CANARY,
    verbose: bool = True,
) -> tuple[pd.DataFrame | None, bool]:
    """Request a chunk until the canary comes back, or attempts run out.

    Returns ``(closes, trusted)``. ``trusted`` is False when every attempt was
    throttled, in which case the caller must not record the batch as empty.
    """
    import time

    for attempt in range(attempts):
        closes = _download_chunk(batch, canary=canary)
        canary_ok = (
            closes is not None
            and canary in closes.columns
            and closes[canary].notna().any()
        )
        if canary_ok:
            return closes, True
        wait = backoff * (2**attempt)
        if verbose:
            print(f"    throttled (canary empty), waiting {wait:.0f}s", flush=True)
        time.sleep(wait)
    return None, False


def probe_ticker_space(
    codes: list[str] | None = None,
    chunk_size: int = 40,
    min_months: int = 24,
    cache_path: Path = PROBE_PATH,
    resume: bool = True,
    verbose: bool = True,
    pause: float = 0.5,
) -> pd.DataFrame:
    """Discover which candidate codes ever had price history on Yahoo.

    Uses monthly bars, which is roughly twenty times lighter than daily and is
    enough to decide whether a ticker deserves a full daily pull.

    Chunks are small and every one carries a canary ticker, because Yahoo
    silently returns empty frames for valid symbols when a request is too
    large or too frequent. A throttled chunk that was recorded as "no data"
    would become a fabricated delisting in the dataset, which is precisely the
    error this whole module exists to avoid, so a chunk whose canary comes back
    empty is retried with exponential backoff and never written to the cache.

    Results are cached to parquet after every chunk so an interrupted scan
    resumes instead of restarting.
    """
    import time

    codes = codes or candidate_codes()

    done: pd.DataFrame | None = None
    if resume and cache_path.exists():
        done = pd.read_parquet(cache_path)
        seen = set(done.index)
        remaining = [c for c in codes if c not in seen]
    else:
        remaining = list(codes)

    if verbose:
        print(f"probing {len(remaining)} of {len(codes)} candidate codes", flush=True)

    rows: list[ProbeResult] = []
    skipped: list[str] = []

    for start in range(0, len(remaining), chunk_size):
        batch = remaining[start : start + chunk_size]
        closes, trusted = _chunk_with_retry(batch, verbose=verbose)

        if not trusted:
            skipped.extend(batch)
            if verbose:
                print(f"    giving up on {len(batch)} codes this pass", flush=True)
            continue

        rows.extend(_summarise_probe(closes, batch))

        partial = pd.DataFrame([r.__dict__ for r in rows]).set_index("ticker")
        combined = partial if done is None else pd.concat([done, partial])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(cache_path)

        if verbose and (start // chunk_size) % 10 == 0:
            hits = int((combined["n_months"] >= min_months).sum())
            print(
                f"  {start + len(batch):>6}/{len(remaining)} probed, "
                f"{hits} tickers with >= {min_months} months",
                flush=True,
            )
        time.sleep(pause)

    if skipped and verbose:
        print(
            f"\n{len(skipped)} codes were never answered and stay unprobed; "
            "re-run to retry them",
            flush=True,
        )

    final = pd.read_parquet(cache_path)
    final["is_listed"] = final["n_months"] >= min_months
    return final.sort_index()


def live_pool(probe: pd.DataFrame, min_months: int = 24) -> list[str]:
    """Tickers worth pulling daily history for, survivors and delistings alike."""
    keep = probe.loc[probe["n_months"] >= min_months]
    return sorted(keep.index.tolist())


def point_in_time_universe(
    close: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    cfg: UniverseConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild index-style membership each month without look-ahead.

    At every rebalance date a name qualifies only if, using data strictly up to
    that date, it traded recently, traded often enough over the trailing
    window, clears the price and turnover floors, and then ranks inside the top
    ``cfg.size`` by trailing median turnover.

    Returns ``(members, liquidity_decile)``, both aligned to ``close``.
    ``members`` is True where the name is in the universe that day; membership
    is held constant between rebalances, which is what a real book does.
    ``liquidity_decile`` is 0 for the most liquid names and drives the
    liquidity-dependent slippage in the cost model.
    """
    rebalance_dates = list(close.resample(cfg.rebalance_freq).last().index)

    members = pd.DataFrame(False, index=close.index, columns=close.columns)
    deciles = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    window = cfg.liquidity_window
    min_obs = int(window * cfg.min_data_fraction)

    for position, date in enumerate(rebalance_dates):
        history = close.loc[:date]
        if history.shape[0] < window:
            continue
        window_close = history.iloc[-window:]
        window_dv = dollar_volume.reindex(window_close.index)

        traded = window_close.notna().sum()
        recent = window_close.iloc[-cfg.max_staleness_days :].notna().any()
        median_dv = window_dv.median()
        last_price = window_close.ffill().iloc[-1]

        eligible = (
            (traded >= min_obs)
            & recent
            & (median_dv >= cfg.min_dollar_volume)
            & (last_price >= cfg.min_price)
        )
        if not eligible.any():
            continue

        ranked = median_dv[eligible].sort_values(ascending=False)
        chosen = ranked.iloc[: cfg.size].index

        # Hold this membership from the day after the rebalance to the next one.
        next_date = (
            rebalance_dates[position + 1]
            if position + 1 < len(rebalance_dates)
            else None
        )
        span = close.index[close.index > date]
        if next_date is not None:
            span = span[span <= next_date]
        if len(span) == 0:
            continue

        members.loc[span, chosen] = True

        decile = pd.Series(
            np.floor(np.arange(len(chosen)) / max(len(chosen) / 10.0, 1.0)).clip(0, 9),
            index=chosen,
        )
        deciles.loc[span, chosen] = decile.to_numpy()

    return members, deciles
