"""Step 4: is the momentum result robust, or does it live in the illiquid tail?

The baseline run leaves one uncomfortable observation. Momentum's Sharpe is
clearly positive across the ASX 300 and close to zero across the ASX 100, which
is the direction that should worry a researcher rather than please one: an edge
concentrated in the smallest, widest-spread, hardest-to-borrow names is the
edge most likely to be an artefact of costs, delisting or plain illiquidity.

This script runs the best momentum configuration across a range of universe
sizes and both execution styles, so the size dependence can be read directly
instead of inferred from two points.

    python scripts/04_momentum_robustness.py --start 2010-01-01
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from asxstat import backtest, clean, download, metrics, signals
from asxstat.config import DEFAULT, RESULTS_DIR, UniverseConfig
from asxstat.universe import point_in_time_universe

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--sizes", type=str, default="500,300,200,100,50")
    args = parser.parse_args()

    panels = download.load_panels()
    data = clean.build_dataset(panels, start=args.start, end=args.end)
    try:
        benchmark = download.load_benchmark()
    except FileNotFoundError:
        benchmark = None

    mom = signals.risk_adjusted_momentum(data.returns, lookback=252, skip=21)

    # The configuration the sweep selected: monthly rebalance, five-day
    # smoothing, which is where net return peaked.
    base = replace(DEFAULT.backtest, rebalance_freq="ME", signal_smoothing=5)

    rows: list[pd.Series] = []
    for size in [int(s) for s in args.sizes.split(",")]:
        ucfg = UniverseConfig(size=size)
        members, deciles = point_in_time_universe(
            data.close, data.dollar_volume, ucfg
        )
        median_names = int(members.sum(axis=1).median())

        for style in ("market", "limit"):
            cfg = replace(base, costs=replace(base.costs, execution_style=style))
            label = f"top{size}_{style}"
            result = backtest.run_backtest(
                mom, data, members, deciles, cfg, name=label
            )
            summary = metrics.performance_summary(
                net=result.net_returns,
                gross=result.gross_returns,
                turnover=result.turnover,
                benchmark=benchmark,
                name=label,
            )
            summary["median_names"] = median_names
            rows.append(summary)

        # Information coefficient does not depend on the cost model, so it
        # separates "the signal is weaker here" from "the costs ate it".
        ic = backtest.information_coefficient(
            mom, data.returns, members, horizons=(21, 63)
        )
        stats = backtest.ic_summary(ic)
        print(f"\ntop{size}: median {median_names} names")
        print(stats[["mean_ic", "t_stat"]])

    table = pd.concat(rows, axis=1).T
    table.to_csv(RESULTS_DIR / "momentum_robustness.csv")
    print("\n=== momentum by universe size, monthly rebalance, 5d smoothing ===")
    print(table[["median_names", "ann_return", "ann_return_gross", "sharpe",
                 "ann_turnover", "break_even_bps", "max_drawdown"]])


if __name__ == "__main__":
    main()
