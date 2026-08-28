"""
Parameter Sensitivity Heatmaps -- Channel Breakout & Bollinger Bands
======================================================================

Sweeps the channel-breakout momentum system (regime_switch_backtest.py)
over its (X = exit_lookback, Y = entry_lookback) grid, and the Bollinger-
band reversal system over its (window, num_std) grid, running each
parameter combination as its own standalone backtest (net of costs) and
plotting the resulting annualised Sharpe ratio as a heatmap. This is a
robustness/sensitivity check on the default lookback choices in
RegimeConfig -- run it to see how much a given system's Sharpe depends
on its exact parameters, or to pick a less curve-fit default.

Each strategy is run on its own here (no VIX regime-switching) so the
heatmap reflects that system's parameter sensitivity in isolation.

Uses the same live ASX200 + ^AXVI OHLC pull as regime_switch_backtest.py.
-----------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vix_decile_backtest import data_output, performance_metrics
from regime_switch_backtest import (
    RegimeConfig,
    load_asx200_tickers,
    load_ohlc_panel,
    sanitize_ohlc_panel,
    channel_breakout_positions,
    bollinger_positions,
    simulate_positions_bps,
)


CHANNEL_EXIT_GRID = list(range(1, 60))        # X: exit lookback (days), 1..59
CHANNEL_ENTRY_GRID = list(range(1, 60))       # Y: entry lookback (days), 1..59, X < Y required

BOLLINGER_WINDOW_GRID = list(range(5, 201, 5))               # MA window (days), 5..200
BOLLINGER_STD_GRID = [round(0.5 + 0.25 * i, 2) for i in range(13)]  # 0.5..3.5 std devs


def sweep_channel_breakout(prices: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                            cfg: RegimeConfig) -> pd.DataFrame:
    """
    Annualised Sharpe (net of costs) of the momentum-only channel-
    breakout system for every (X, Y) combo in the module-level grids.
    Cells where X >= Y are left NaN -- the exit lookback must be shorter
    than the entry lookback by construction (see the rule in
    regime_switch_backtest.channel_breakout_positions).
    """
    sharpe = pd.DataFrame(index=CHANNEL_EXIT_GRID, columns=CHANNEL_ENTRY_GRID, dtype=float)
    sharpe.index.name = "exit_lookback_X"
    sharpe.columns.name = "entry_lookback_Y"

    combos = [(x, y) for x in CHANNEL_EXIT_GRID for y in CHANNEL_ENTRY_GRID if x < y]
    for n, (x, y) in enumerate(combos, 1):
        if n == 1 or n % 100 == 0 or n == len(combos):
            print(f"  channel breakout: combo {n}/{len(combos)} (X={x}, Y={y})...")
        pos = channel_breakout_positions(high, low, y, x)
        result = simulate_positions_bps(prices, pos, cfg)
        metrics = performance_metrics(result["daily_returns"], cfg, result["turnover"])
        sharpe.loc[x, y] = metrics["annualised_sharpe"]

    return sharpe


def sweep_bollinger(prices: pd.DataFrame, cfg: RegimeConfig) -> pd.DataFrame:
    """Annualised Sharpe (net of costs) of the reversal-only Bollinger-
    band system for every (window, num_std) combo in the module-level
    grids."""
    sharpe = pd.DataFrame(index=BOLLINGER_WINDOW_GRID, columns=BOLLINGER_STD_GRID, dtype=float)
    sharpe.index.name = "window"
    sharpe.columns.name = "num_std"

    combos = [(w, s) for w in BOLLINGER_WINDOW_GRID for s in BOLLINGER_STD_GRID]
    for n, (w, s) in enumerate(combos, 1):
        if n == 1 or n % 25 == 0 or n == len(combos):
            print(f"  bollinger: combo {n}/{len(combos)} (window={w}, num_std={s})...")
        pos = bollinger_positions(prices, w, s)
        result = simulate_positions_bps(prices, pos, cfg)
        metrics = performance_metrics(result["daily_returns"], cfg, result["turnover"])
        sharpe.loc[w, s] = metrics["annualised_sharpe"]

    return sharpe


def plot_sharpe_heatmap(sharpe: pd.DataFrame, title: str, xlabel: str, ylabel: str, save_path: str,
                         max_labeled_cells: int = 100, max_ticks: int = 20):
    """
    Sequential heatmap (dark = low Sharpe, light = high Sharpe) on a
    perceptually-uniform colormap -- at this grid density the surface
    reads as a continuous field rather than a small lookup table, so a
    single-hue magnitude ramp (as opposed to a red/white/blue diverging
    scale) is what makes the ridge of good parameter combinations pop
    visually, matching how this kind of dense Sharpe-surface sweep is
    usually plotted. Invalid (X >= Y) combos are rendered as a flat
    neutral grey, distinct from the colormap's own dark end, so they
    can't be misread as a real (bad) Sharpe. Cell values are only
    direct-labeled below `max_labeled_cells` -- past that the grid is
    dense enough that per-cell text just clutters the surface (the exact
    numbers are in the CSV saved alongside).
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))

    data = sharpe.values.astype(float)
    masked = np.ma.masked_invalid(data)

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#9e9e9e")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", origin="upper")

    n_rows, n_cols = data.shape
    row_step = max(1, -(-n_rows // max_ticks))   # ceil division
    col_step = max(1, -(-n_cols // max_ticks))
    ax.set_xticks(range(0, n_cols, col_step))
    ax.set_xticklabels([sharpe.columns[i] for i in range(0, n_cols, col_step)], rotation=45, ha="right")
    ax.set_yticks(range(0, n_rows, row_step))
    ax.set_yticklabels([sharpe.index[i] for i in range(0, n_rows, row_step)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if n_rows * n_cols <= max_labeled_cells:
        finite = data[np.isfinite(data)]
        mid = (finite.max() + finite.min()) / 2 if finite.size else 0.0
        for i in range(n_rows):
            for j in range(n_cols):
                val = data[i, j]
                if np.isfinite(val):
                    color = "white" if val < mid else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, label="Annualised Sharpe (net of costs)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved heatmap to {save_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("PARAMETER SENSITIVITY HEATMAPS -- CHANNEL BREAKOUT & BOLLINGER BANDS")
    print("=" * 70)

    cfg = RegimeConfig()
    start_date = "2015-01-01"

    try:
        tickers = load_asx200_tickers()
        ohlc = sanitize_ohlc_panel(load_ohlc_panel(tickers, start=start_date))
    except Exception as e:
        raise SystemExit(
            "\nFailed to download data via yfinance/Wikipedia.\n"
            "This requires outbound internet access (Yahoo Finance and\n"
            "en.wikipedia.org). If you're running this inside a\n"
            "network-restricted environment, run it on your own machine\n"
            "or a notebook with normal internet access instead.\n"
            f"\nUnderlying error: {e}"
        )

    prices, high, low = ohlc["close"], ohlc["high"], ohlc["low"]

    print("\nSweeping channel breakout (X, Y)...")
    channel_sharpe = sweep_channel_breakout(prices, high, low, cfg)
    print("\nSweeping Bollinger bands (window, num_std)...")
    bollinger_sharpe = sweep_bollinger(prices, cfg)

    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("\n--- Channel breakout Sharpe grid (rows=X exit, cols=Y entry) ---")
    print(channel_sharpe)
    print("\n--- Bollinger Sharpe grid (rows=window, cols=num_std) ---")
    print(bollinger_sharpe)

    channel_sharpe.to_csv(f"{data_output}/channel_breakout_sharpe_grid.csv")
    bollinger_sharpe.to_csv(f"{data_output}/bollinger_sharpe_grid.csv")

    plot_sharpe_heatmap(
        channel_sharpe,
        title="Channel Breakout: Annualised Sharpe by (X, Y)",
        xlabel="Entry lookback Y (days)",
        ylabel="Exit lookback X (days)",
        save_path=f"{data_output}/channel_breakout_sharpe_heatmap.png",
    )
    plot_sharpe_heatmap(
        bollinger_sharpe,
        title="Bollinger Bands: Annualised Sharpe by (window, num_std)",
        xlabel="Std dev multiplier",
        ylabel="MA window (days)",
        save_path=f"{data_output}/bollinger_sharpe_heatmap.png",
    )
