"""Step 3: the baseline research run.

Builds the cleaned dataset, reconstructs the point-in-time universe, then puts
four signals through the same pipeline:

    mom_12_1        risk-adjusted 12-month momentum, skipping the last month
    rev_5d          five-day cross-sectional reversal
    rev_5d_quiet    the same reversal, tilted towards low-turnover names
    rev_resid_5d    reversal on returns with statistical factors removed

For each one it reports the information coefficient by horizon, the quantile
spread, and full performance net of the liquidity-scaled cost model. It then
blends them risk-parity and, finally, re-runs the blend under three
survivorship scenarios so the headline Sharpe comes with an error bar.

    python scripts/03_run_baseline.py --start 2010-01-01 --universe-size 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from asxstat import backtest, clean, download, metrics, signals, survivorship
from asxstat.config import DEFAULT, RESULTS_DIR, UniverseConfig
from asxstat.universe import point_in_time_universe

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def build(args) -> tuple:
    panels = download.load_panels()
    data = clean.build_dataset(panels, start=args.start, end=args.end)
    print(f"dataset: {data.close.shape[1]:,} tickers, "
          f"{data.close.shape[0]:,} days, "
          f"{data.dates.min().date()} to {data.dates.max().date()}")

    delisting = clean.delisting_report(data)
    n_dead = int(delisting["delisted"].sum())
    print(f"delisted during or before sample: {n_dead:,} of {len(delisting):,} "
          f"({n_dead / max(len(delisting), 1):.1%})")

    ucfg = UniverseConfig(size=args.universe_size)
    members, deciles = point_in_time_universe(data.close, data.dollar_volume, ucfg)
    print(f"universe: median {int(members.sum(axis=1).median()):,} names per day")

    return data, delisting, members, deciles


def build_signals(data, members) -> dict[str, pd.DataFrame]:
    turnover = signals.abnormal_turnover(data.dollar_volume)

    mom = signals.risk_adjusted_momentum(data.returns, lookback=252, skip=21)
    rev = signals.reversal(data.returns, lookback=5)
    rev_quiet = signals.condition_signal(
        rev, turnover, mask=members, direction=-1, strength=0.5
    )

    print("fitting rolling factor model for residual reversal ...")
    residuals, _, _ = signals.residual_returns(
        data.returns, n_factors=3, window=252, refit_every=21
    )
    rev_resid = signals.reversal(residuals, lookback=5)

    return {
        "mom_12_1": mom,
        "rev_5d": rev,
        "rev_5d_quiet": rev_quiet,
        "rev_resid_5d": rev_resid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--universe-size", type=int, default=300)
    parser.add_argument("--compare-sizes", type=str, default="300,100",
                        help="universe sizes to run side by side")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="simulated delisting histories")
    parser.add_argument("--annual-hazard", type=float, default=0.05,
                        help="annual probability a name delists, before the "
                             "trailing-return tilt")
    parser.add_argument("--terminal-return", type=float, default=None,
                        help="assumed return on a simulated delisting; the "
                             "default varies it by liquidity decile, from a "
                             "takeover at zero to a collapse at -85%%")
    args = parser.parse_args()

    data, delisting, members, deciles = build(args)

    coverage = survivorship.coverage_report(data, delisting)
    coverage.to_csv(RESULTS_DIR / "coverage.csv")
    print("\nsurvivorship coverage by year (share of trading names that later delisted)")
    print(coverage.tail(10))

    try:
        benchmark = download.load_benchmark()
    except FileNotFoundError:
        benchmark = None
        print("\nno benchmark cached; alpha/beta will be skipped")

    signal_frames = build_signals(data, members)
    cfg = DEFAULT.backtest

    results: dict[str, backtest.BacktestResult] = {}
    summaries: list[pd.Series] = []

    for name, scores in signal_frames.items():
        print(f"\n=== {name} ===")

        ic = backtest.information_coefficient(scores, data.returns, members)
        print(backtest.ic_summary(ic))

        quantiles = backtest.quantile_returns(scores, data, members)
        spread = quantiles["long_short"].mean() * 252
        print(f"top-minus-bottom quintile, annualised: {spread:.2%}")

        result = backtest.run_backtest(scores, data, members, deciles, cfg, name=name)
        results[name] = result

        summary = metrics.performance_summary(
            net=result.net_returns,
            gross=result.gross_returns,
            turnover=result.turnover,
            benchmark=benchmark,
            name=name,
        )
        summaries.append(summary)
        print(summary)

    table = metrics.compare(summaries)
    table.to_csv(RESULTS_DIR / "signal_performance.csv")
    print("\n=== all signals, net of costs ===")
    print(table)

    blended = backtest.combine(results)
    blended.to_csv(RESULTS_DIR / "blended_returns.csv")
    combined_summary = metrics.performance_summary(
        blended["combined"], benchmark=benchmark, name="risk_parity_blend"
    )
    print("\n=== risk-parity blend ===")
    print(combined_summary)
    print("\ncorrelation between sleeves")
    print(blended.drop(columns="combined").corr())

    run_rebalance_sweep(data, signal_frames, members, deciles, cfg, benchmark)

    sizes = [int(s) for s in args.compare_sizes.split(",")]
    run_universe_size_comparison(data, signal_frames, cfg, benchmark, sizes)

    run_survivorship_scenarios(
        data, delisting, members, deciles, signal_frames, cfg, benchmark, args
    )


def run_survivorship_scenarios(
    data, delisting, members, deciles, signal_frames, cfg, benchmark, args
) -> None:
    """Put an error bar on every headline number.

    The free ASX data reaching this script is survivor-only: Yahoo has purged
    almost every delisted Australian ticker, so the sample cannot be repaired,
    only bounded. Two independent bounds are produced.

    The simulation injects deaths at a realistic rate, concentrated in names
    whose price has already fallen, and reports how far performance moves. The
    universe-size comparison exploits the fact that large-cap delistings are
    mostly takeovers, which resolve near zero, while small-cap delistings are
    mostly collapses, which resolve near -100%: an edge that survives in the
    ASX 100 is far less likely to be an artefact than one that only appears
    across the ASX 300.
    """
    print("\n=== survivorship: simulated delistings ===")
    print("free ASX data is survivor-only, so these are bounds, not corrections")

    for signal_name in ("mom_12_1", "rev_5d"):
        scores = signal_frames[signal_name]

        baseline = backtest.run_backtest(scores, data, members, deciles, cfg)
        baseline_sharpe = metrics.sharpe_ratio(baseline.net_returns)

        table = survivorship.bias_simulation(
            scores,
            data,
            members,
            deciles,
            cfg,
            universe_cfg=UniverseConfig(size=args.universe_size),
            n_trials=args.n_trials,
            annual_hazard=args.annual_hazard,
            terminal_return=args.terminal_return,
        )
        table.to_csv(RESULTS_DIR / f"survivorship_sim_{signal_name}.csv")

        low, median, high = table["sharpe"].quantile([0.05, 0.5, 0.95])
        print(f"\n{signal_name}")
        print(f"  survivor-only sharpe   : {baseline_sharpe:6.2f}")
        print(f"  simulated median       : {median:6.2f}")
        print(f"  simulated 5-95 pct     : {low:6.2f} to {high:6.2f}")
        print(f"  estimated bias         : {baseline_sharpe - median:6.2f} sharpe")
        print(f"  deaths per trial       : {table['n_delisted'].mean():.0f}")
        print(f"  median names in book   : {table['median_names'].mean():.0f}")
        if baseline_sharpe > high:
            print("  WARNING: survivor-only result sits outside the simulated range")


def run_rebalance_sweep(
    data, signal_frames, members, deciles, cfg, benchmark
) -> None:
    """Trade each signal at several rebalance frequencies and holding smooths.

    This is the decisive experiment for a cost-sensitive strategy. Both signals
    have clearly positive information coefficients, so the question is not
    whether they predict returns but whether the prediction survives the ASX
    spread. Slowing the book down trades a little signal decay for a large
    reduction in turnover, and the sweep locates the point where net return
    peaks rather than assuming daily trading.
    """
    from dataclasses import replace

    print("\n=== rebalance frequency sweep ===")
    rows: list[pd.Series] = []

    from asxstat.config import CostModel

    for signal_name, scores in signal_frames.items():
        for freq in ("D", "W", "ME"):
            for smoothing in (1, 5):
                for style in ("market", "limit"):
                    variant = replace(
                        cfg,
                        rebalance_freq=freq,
                        signal_smoothing=smoothing,
                        costs=replace(cfg.costs, execution_style=style),
                    )
                    label = f"{signal_name}_{freq}_s{smoothing}_{style}"
                    result = backtest.run_backtest(
                        scores, data, members, deciles, variant, name=label
                    )
                    rows.append(
                        metrics.performance_summary(
                            net=result.net_returns,
                            gross=result.gross_returns,
                            turnover=result.turnover,
                            benchmark=benchmark,
                            name=label,
                        )
                    )

    table = pd.concat(rows, axis=1).T
    table.to_csv(RESULTS_DIR / "rebalance_sweep.csv")
    print(table[["ann_return", "ann_return_gross", "sharpe", "ann_turnover",
                 "break_even_bps"]])

    best = table["sharpe"].idxmax()
    print(f"\nbest net sharpe: {best} at {table.loc[best, 'sharpe']:.2f}")


def run_universe_size_comparison(data, signal_frames, cfg, benchmark, sizes) -> None:
    """Run every signal at each universe size.

    Delisting risk, borrow availability and the bid-ask spread all worsen
    together as the universe widens, so a result that only appears at 300 names
    is the one to distrust.
    """
    print("\n=== universe size comparison ===")
    rows: list[pd.Series] = []

    for size in sizes:
        ucfg = UniverseConfig(size=size)
        members, deciles = point_in_time_universe(
            data.close, data.dollar_volume, ucfg
        )
        for name, scores in signal_frames.items():
            result = backtest.run_backtest(
                scores, data, members, deciles, cfg, name=f"{name}_top{size}"
            )
            rows.append(
                metrics.performance_summary(
                    net=result.net_returns,
                    gross=result.gross_returns,
                    turnover=result.turnover,
                    benchmark=benchmark,
                    name=f"{name}_top{size}",
                )
            )

    table = pd.concat(rows, axis=1).T
    table.to_csv(RESULTS_DIR / "universe_size_comparison.csv")
    print(table[["ann_return", "sharpe", "max_drawdown", "ann_turnover",
                 "break_even_bps"]])


if __name__ == "__main__":
    main()
