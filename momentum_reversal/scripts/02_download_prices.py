"""Step 2: pull daily OHLCV for the recovered ticker pool, plus the benchmark.

Reads the pool written by step 1 and caches wide parquet panels under data/raw.
Resumable in the same way: already-downloaded tickers are skipped.

    python scripts/02_download_prices.py --start 2005-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from asxstat import download
from asxstat.config import BENCHMARK, BENCHMARK_FALLBACKS, RAW_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, default="2005-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-passes", type=int, default=5,
                        help="repeat the download to fill throttled gaps")
    parser.add_argument("--limit", type=int, default=None,
                        help="only download the first N tickers, for a smoke test")
    args = parser.parse_args()

    pool_path = RAW_DIR / "ticker_pool.csv"
    if not pool_path.exists():
        raise SystemExit("run scripts/01_probe_universe.py first")

    tickers = pd.read_csv(pool_path)["ticker"].tolist()
    if args.limit:
        tickers = tickers[: args.limit]

    # Yahoo silently returns empty frames for valid symbols when it throttles,
    # so a single pass leaves gaps. Because the downloader resumes from what is
    # already cached, simply repeating it fills those gaps; stop when a pass
    # stops adding tickers.
    previous = -1
    for attempt in range(args.max_passes):
        panels = download.download_daily(
            tickers,
            start=args.start,
            end=args.end,
            chunk_size=args.chunk_size,
        )
        have = panels["close"].shape[1]
        print(f"pass {attempt + 1}: {have:,} tickers cached")
        if have == previous or have >= len(tickers):
            break
        previous = have

    returns_bm = download.download_benchmark(
        (BENCHMARK, *BENCHMARK_FALLBACKS), start=args.start, end=args.end
    )

    close = panels["close"]
    print()
    print(f"tickers with daily data : {close.shape[1]:,}")
    print(f"date range              : {close.index.min().date()} to {close.index.max().date()}")
    print(f"benchmark               : {returns_bm.name}, {len(returns_bm):,} days")


if __name__ == "__main__":
    main()
