"""Step 1: discover every ASX ticker that ever had price history.

Scans the three-letter ASX code space (AAA..ZZZ, 17,576 candidates) with cheap
monthly bars and records which ones Yahoo has data for. Dead companies are the
point of the exercise: they are what a downloadable ASX 300 constituent list
leaves out.

Resumable. Interrupt it and run it again; it picks up from the cache.

    python scripts/01_probe_universe.py --chunk-size 200 --min-months 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from asxstat import universe
from asxstat.config import RAW_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--min-months", type=int, default=24)
    parser.add_argument(
        "--lengths",
        type=str,
        default="3",
        help="comma-separated code lengths to scan, e.g. '3' or '3,4'",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    lengths = tuple(int(x) for x in args.lengths.split(","))
    codes = universe.candidate_codes(lengths)

    probe = universe.probe_ticker_space(
        codes=codes,
        chunk_size=args.chunk_size,
        min_months=args.min_months,
        resume=not args.no_resume,
    )

    pool = universe.live_pool(probe, min_months=args.min_months)
    pd.Series(pool, name="ticker").to_csv(RAW_DIR / "ticker_pool.csv", index=False)

    listed = probe.loc[probe["is_listed"]]
    still_trading = listed["last_date"] >= listed["last_date"].max() - pd.Timedelta(days=30)

    print()
    print(f"candidates probed        : {len(probe):,}")
    print(f"with >= {args.min_months} months history : {len(pool):,}")
    print(f"still trading recently   : {int(still_trading.sum()):,}")
    print(f"stopped trading (dead)   : {int((~still_trading).sum()):,}")
    print(f"pool written to          : {RAW_DIR / 'ticker_pool.csv'}")


if __name__ == "__main__":
    main()
