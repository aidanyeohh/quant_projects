"""
Trade-Level Performance -- Top/Bottom 20 Names per Strategy
=============================================================

For each of the three tradeable strategies in regime_switch_backtest.py
(channel-breakout momentum, Bollinger-band reversal, and the VIX
regime-switch composite), walks every name's discrete position series
into individual entry->exit trade episodes, ranks names by their TOTAL
COMPOUNDED trade return (i.e. "if you'd taken every trade this strategy's
rules generated in this name, entry to exit, back to back"), and saves a
grid plot of the top 20 and bottom 20 names' share price with each
holding period shaded.

This ranks a strategy's RULES on a per-name basis -- it does not use the
portfolio-level position sizing from simulate_positions_bps, so it isn't
a claim about realised portfolio P&L (see regime_switch_backtest.py for
that). It answers a different, complementary question: which names does
this trading rule work well/badly on, and what did those entries and
exits actually look like against the price?

Buy-and-hold is excluded from the per-name trade grids -- it has no
entries/exits to plot -- but IS included, alongside the three tradeable
strategies, in the overall (portfolio-level) return comparison plot,
which runs the actual simulate_positions_bps / buy_and_hold engines from
regime_switch_backtest.py to answer the complementary aggregate
question: how did the whole book do, not just its best/worst names.

Uses the same live ASX200 + ^AXVI OHLC pull as regime_switch_backtest.py.
-----------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vix_decile_backtest import data_output, vix_expanding_deciles, performance_metrics
from regime_switch_backtest import (
    RegimeConfig,
    load_asx200_tickers,
    load_ohlc_panel,
    sanitize_ohlc_panel,
    channel_breakout_positions,
    bollinger_positions,
    regime_switch_positions,
    simulate_positions_bps,
    buy_and_hold,
    load_vix_series,
)


TOP_N = 20


def extract_trades(positions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Walks each name's discrete position series (long-only {0,1}, or
    long/short {-1,0,1}) and extracts individual entry->exit trade
    episodes. Entry/exit prices are the close on the day AFTER the
    signal changes -- matching simulate_positions_bps's convention that
    a position set at day t's close only starts earning returns from day
    t+1 onward, so plotted trades reflect what was actually tradeable,
    not a same-day look-ahead fill. A trade still open at the end of the
    sample is dropped (not closed out), since it has no realised return
    to rank on.
    """
    pos = positions.reindex(columns=prices.columns).fillna(0.0)
    pos_arr = pos.to_numpy()
    px = prices.to_numpy()
    dates = prices.index
    n_days, n_names = pos_arr.shape

    records = []
    for c in range(n_names):
        ticker = prices.columns[c]
        state = 0.0
        entry_i = None
        for i in range(n_days):
            cur = pos_arr[i, c]
            if cur == state:
                continue
            if state == 0.0 and cur != 0.0:
                entry_i = i + 1
                side = cur
            elif state != 0.0 and cur == 0.0:
                exit_i = i + 1
                if entry_i is not None and entry_i < n_days and exit_i < n_days:
                    entry_price, exit_price = px[entry_i, c], px[exit_i, c]
                    if np.isfinite(entry_price) and np.isfinite(exit_price) and entry_price > 0:
                        trade_ret = side * (exit_price / entry_price - 1.0)
                        records.append({
                            "ticker": ticker,
                            "side": "long" if side > 0 else "short",
                            "entry_date": dates[entry_i], "exit_date": dates[exit_i],
                            "entry_price": entry_price, "exit_price": exit_price,
                            "trade_return": trade_ret,
                        })
                entry_i = None
            state = cur

    return pd.DataFrame.from_records(records)


def rank_names_by_strategy_return(trades: pd.DataFrame) -> pd.Series:
    """
    Total compounded return per name across all its closed trades under
    a given strategy -- used purely to RANK names by how well the
    strategy's rules performed on them, independent of any portfolio
    position sizing.
    """
    if trades.empty:
        return pd.Series(dtype=float)
    compounded = trades.groupby("ticker")["trade_return"].apply(lambda r: np.prod(1 + r) - 1)
    return compounded.sort_values(ascending=False)


def plot_trade_grid(tickers: list, prices: pd.DataFrame, trades: pd.DataFrame,
                     title: str, save_path: str, n_cols: int = 4):
    """
    Grid of small per-name price charts, one per ticker, with each
    holding period shaded -- so the "what did the strategy actually do"
    question reads at a glance per name, not just as an aggregate return
    number.
    """
    n = len(tickers)
    n_cols = min(n_cols, n) if n > 0 else 1
    n_rows = int(np.ceil(n / n_cols)) if n > 0 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows), squeeze=False)

    for idx, ticker in enumerate(tickers):
        ax = axes[idx // n_cols][idx % n_cols]
        px = prices[ticker].dropna()
        ax.plot(px.index, px.values, color="#4c72b0", linewidth=1, zorder=1)

        t_trades = trades[trades["ticker"] == ticker].sort_values("entry_date")
        for _, tr in t_trades.iterrows():
            ax.axvspan(tr["entry_date"], tr["exit_date"], color="#2ca02c", alpha=0.12, zorder=0)
        if len(t_trades):
            total_ret = np.prod(1 + t_trades["trade_return"]) - 1
            ax.set_title(f"{ticker}  ({total_ret:+.1%}, {len(t_trades)} trades)", fontsize=9)
        else:
            ax.set_title(ticker, fontsize=9)
        ax.tick_params(labelsize=7, rotation=0)
        ax.margins(x=0.02)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(title, y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


def plot_overall_returns(strategy_returns: dict, cfg: RegimeConfig, save_path: str):
    """
    Portfolio-level view, alongside the per-name trade grids above: a
    cumulative growth-of-$1 curve (net of costs) for every strategy plus
    a Sharpe comparison bar, using the SAME simulate_positions_bps /
    buy_and_hold engines as regime_switch_backtest.py -- i.e. this is
    the actual realised book return, not the per-name trade-return
    ranking used to pick the top/bottom names.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sharpes = {}
    for name, daily_returns in strategy_returns.items():
        curve = (1 + daily_returns.fillna(0)).cumprod()
        axes[0].plot(curve.index, curve.values, label=name)
        sharpes[name] = performance_metrics(daily_returns, cfg)["annualised_sharpe"]

    axes[0].set_title("Cumulative Growth of $1 (net of costs)")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(fontsize=8)

    names = list(sharpes.keys())
    axes[1].bar(names, [sharpes[n] for n in names], color="#4c72b0")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Annualised Sharpe (net of costs)")
    axes[1].set_title("Sharpe Comparison")
    axes[1].tick_params(axis="x", rotation=25, labelsize=8)
    for label in axes[1].get_xticklabels():
        label.set_ha("right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


def analyze_and_plot_trade_performance(ohlc: dict, vix: pd.Series, cfg: RegimeConfig = None,
                                        top_n: int = TOP_N) -> dict:
    cfg = cfg or RegimeConfig()

    ohlc = sanitize_ohlc_panel(ohlc)
    prices, high, low = ohlc["close"], ohlc["high"], ohlc["low"]
    common_idx = prices.index.intersection(vix.index)
    prices, high, low = prices.loc[common_idx], high.loc[common_idx], low.loc[common_idx]
    vix = vix.loc[common_idx]

    mom_pos = channel_breakout_positions(high, low, cfg.channel_entry_lookback, cfg.channel_exit_lookback)
    rev_pos = bollinger_positions(prices, cfg.bollinger_window, cfg.bollinger_num_std)
    vix_dec = vix_expanding_deciles(vix, cfg.vix_n_deciles)
    combo_pos = regime_switch_positions(mom_pos, rev_pos, vix_dec, cfg.low_vol_max_decile)

    strategies = {
        "Momentum (Channel Breakout)": mom_pos,
        "Reversal (Bollinger Bands)": rev_pos,
        "Regime-Switch Composite": combo_pos,
    }

    results = {}
    for name, pos in strategies.items():
        print(f"\nExtracting trades for {name}...")
        trades = extract_trades(pos, prices)
        ranked = rank_names_by_strategy_return(trades)
        results[name] = {"trades": trades, "ranked_returns": ranked}

        if ranked.empty:
            print(f"  No closed trades for {name} -- skipping plots.")
            continue

        top = ranked.head(top_n).index.tolist()
        bottom = ranked.tail(top_n).index.tolist()[::-1]  # worst first
        print(f"  {len(ranked)} names traded, {len(trades)} total closed trades. "
              f"Best: {ranked.index[0]} ({ranked.iloc[0]:+.1%}), "
              f"Worst: {ranked.index[-1]} ({ranked.iloc[-1]:+.1%})")

        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_trade_grid(top, prices, trades, f"{name}: Top {len(top)} Performers",
                         f"{data_output}/{slug}_top_performers.png")
        plot_trade_grid(bottom, prices, trades, f"{name}: Bottom {len(bottom)} Underperformers",
                         f"{data_output}/{slug}_underperformers.png")

    print("\nSimulating portfolio-level returns for the overall comparison plot...")
    strategy_returns = {
        name: simulate_positions_bps(prices, pos, cfg)["daily_returns"]
        for name, pos in strategies.items()
    }
    strategy_returns["Buy-and-Hold"] = buy_and_hold(prices, cfg)["daily_returns"]
    plot_overall_returns(strategy_returns, cfg, f"{data_output}/overall_strategy_returns.png")
    for name in results:
        results[name]["daily_returns"] = strategy_returns[name]
    results["Buy-and-Hold"] = {"daily_returns": strategy_returns["Buy-and-Hold"]}

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TRADE-LEVEL PERFORMANCE -- TOP/BOTTOM 20 NAMES PER STRATEGY")
    print("=" * 70)

    cfg = RegimeConfig()
    start_date = "2015-01-01"

    try:
        tickers = load_asx200_tickers()
        ohlc = load_ohlc_panel(tickers, start=start_date)
        vix = load_vix_series(start=start_date)
    except Exception as e:
        raise SystemExit(
            "\nFailed to download data via yfinance/Wikipedia.\n"
            "This requires outbound internet access (Yahoo Finance and\n"
            "en.wikipedia.org). If you're running this inside a\n"
            "network-restricted environment, run it on your own machine\n"
            "or a notebook with normal internet access instead.\n"
            f"\nUnderlying error: {e}"
        )

    analyze_and_plot_trade_performance(ohlc, vix, cfg, top_n=TOP_N)
