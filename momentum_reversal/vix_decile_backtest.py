"""
VIX-Decile Conditioned Momentum vs. Reversal Backtest -- ASX Universe
========================================================================

Research question: Across VIX deciles (global volatility regimes), is a
cross-sectional momentum strategy or a short-term reversal strategy
more profitable, on a risk-adjusted, cost-adjusted basis, when applied
to ASX-listed equities?

Universe: ASX-listed stocks, prices pulled live via `yfinance`.
VIX: CBOE VIX index level (^VIX), also pulled live via `yfinance`.

-----------------------------------------------------------------------
DATA SOURCING NOTES
-----------------------------------------------------------------------
1. Price data comes from Yahoo Finance via `yfinance`. ASX tickers on
   Yahoo carry a `.AX` suffix (e.g. `BHP.AX`, `CBA.AX`). The default
   ticker list below (`DEFAULT_ASX_TICKERS`) is a hand-picked sample of
   large, liquid ASX names so the script runs out of the box -- for an
   actual thesis you should replace this with a proper point-in-time
   index constituent list (e.g. S&P/ASX 200 historical constituents),
   since Yahoo's free data has no delisted/removed names and therefore
   carries survivorship bias. A free static snapshot of current index
   membership is not the same as a point-in-time history.

2. The VIX (^VIX) is a US (S&P 500 options-implied) volatility index.
   Using it to condition an ASX-stock strategy is a common "global risk
   appetite" proxy in the literature, but note it is not an ASX-local
   vol measure. If your thesis wants a local proxy instead, the ASX's
   own index is the S&P/ASX 200 VIX, ticker `^AXVI` on some data
   vendors (availability on Yahoo Finance is inconsistent -- verify
   before relying on it, or source it from the ASX itself). Swap the
   ticker in `load_vix_series()` if you want to use it instead of, or
   alongside, ^VIX.

3. Calendar note: ^VIX is a US session index while ASX trades on a
   different local session. This script aligns on the calendar-date
   index intersection of both series (i.e. it uses the VIX print from
   the US trading day that closes hours after the ASX day already
   ended). If you want the VIX level as of the *previous* US close
   relative to each ASX trading day instead, shift the VIX series by
   one day before calling `run_full_analysis` -- document whichever
   choice you make in the thesis methodology.

-----------------------------------------------------------------------
Run this file directly to pull live ASX + VIX data via yfinance and
run the full backtest end to end.
-----------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


# =========================================================================
# 1. CONFIGURATION
# =========================================================================

@dataclass
class BacktestConfig:
    momentum_lookback: int = 252     # ~12 months of trading days
    momentum_skip: int = 21          # skip most recent month (12-1 momentum)
    reversal_lookback: int = 21      # ~1 month for short-term reversal
    rebalance_freq: str = "ME"       # 'ME' month-end, 'W' weekly (pandas >= 2.2 offset aliases)
    n_quantiles: int = 5             # quintile long/short portfolios
    vix_n_deciles: int = 10
    starting_capital: float = 10_000_000.0   # USD notional for share-count sim
    cost_per_share: float = 0.005    # USD, both sides (buy and sell)
    risk_free_rate_annual: float = 0.0
    min_names_per_leg: int = 5       # skip a rebalance if too few names ranked
    trading_days_per_year: int = 252


data_output = "C:/Users/atsye/Desktop/Python_data/quant_projects/momentum_reversal/data_output"

# =========================================================================
# 2. DATA LOADING -- live via yfinance, ASX universe
# =========================================================================

# Sample of large, liquid ASX-listed names (S&P/ASX 50-ish, hand-picked).
# REPLACE with a proper point-in-time index constituent list for the
# actual thesis -- see module docstring re: survivorship bias.
DEFAULT_ASX_TICKERS = [
    "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX", "ANZ.AX", "MQG.AX",
    "WES.AX", "WOW.AX", "TLS.AX", "RIO.AX", "GMG.AX", "TCL.AX", "FMG.AX",
    "STO.AX", "WDS.AX", "REA.AX", "ALL.AX", "COL.AX", "QBE.AX", "SUN.AX",
    "IAG.AX", "AMC.AX", "ORG.AX", "S32.AX", "APA.AX", "SCG.AX", "MPL.AX",
    "ASX.AX", "JHX.AX", "SHL.AX", "COH.AX", "XRO.AX", "RMD.AX", "NCM.AX",
    "SGP.AX", "VCX.AX", "MGR.AX", "BXB.AX", "AGL.AX", "QAN.AX", "TWE.AX",
    "SEK.AX", "CPU.AX", "ALD.AX", "IGO.AX", "MIN.AX", "PLS.AX", "NST.AX",
    "EVN.AX", "ORI.AX",
]


def load_price_panel(
    tickers=None,
    start: str = "2015-01-01",
    end: str = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Pulls adjusted close prices for ASX tickers from Yahoo Finance via
    yfinance. Requires an internet connection with access to Yahoo
    Finance's servers (query1/query2.finance.yahoo.com) -- if you're
    running this in a network-restricted sandbox, run it locally or in
    an environment where those hosts are reachable instead.

        pip install yfinance
    """
    import yfinance as yf

    tickers = tickers or DEFAULT_ASX_TICKERS
    print(f"Downloading {len(tickers)} ASX tickers from yfinance "
          f"({start} to {end or 'today'})...")

    raw = yf.download(
        tickers, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False, group_by="ticker", threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)})
    else:
        # single-ticker download falls back to flat columns
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(axis=1, how="all")
    dropped = set(tickers) - set(prices.columns)
    if dropped:
        print(f"Warning: no data returned for {sorted(dropped)}")

    return prices


def load_vix_series(start: str = "2015-01-01", end: str = None) -> pd.Series:
    """Pulls the ASX VIX (^AXVI) daily close via yfinance."""
    import yfinance as yf

    print("Downloading ^AXVI from yfinance...")
    raw = yf.download("^AXVI", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        vix = raw[("Close", "^AXVI")]
    else:
        vix = raw["Close"]
    vix.name = "VIX"
    return vix.dropna()


# =========================================================================
# 3. SIGNAL CONSTRUCTION
# =========================================================================

def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()


def momentum_signal(prices: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Classic 12-1 momentum: return over [t-lookback, t-skip], i.e.
    cumulative return ending `skip` days ago so the most recent
    reversal-prone month is excluded."""
    far = prices.shift(skip)
    near = prices.shift(lookback)
    return far / near - 1.0


def reversal_signal(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Short-term reversal: negative of trailing `lookback`-day return.
    High recent losers score high (expected to bounce back)."""
    trailing_ret = prices.pct_change(lookback)
    return -trailing_ret


# =========================================================================
# 4. VIX DECILES (walk-forward, no look-ahead)
# =========================================================================

def vix_expanding_deciles(vix: pd.Series, n_deciles: int = 10, min_history: int = 252) -> pd.Series:
    """
    Assigns each date a decile (1..n_deciles) based on where that day's
    VIX level sits in the EXPANDING historical distribution up to and
    including that day. This avoids look-ahead bias that a full-sample
    quantile cut would introduce (a full-sample cut implicitly uses
    future VIX prints to define "high" vs "low" volatility at date t).

    Dates before `min_history` observations are available are left as NaN.
    """
    vix = vix.sort_index()
    out = pd.Series(index=vix.index, dtype=float)
    values = vix.values
    for i in range(len(vix)):
        if i < min_history:
            continue
        hist = values[: i + 1]
        rank_pct = (hist <= hist[-1]).mean()  # percentile of today's value in history so far
        decile = int(np.ceil(rank_pct * n_deciles))
        decile = min(max(decile, 1), n_deciles)
        out.iloc[i] = decile
    return out


# =========================================================================
# 5. PORTFOLIO SIMULATION (with per-share transaction costs)
# =========================================================================

def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    s = pd.Series(index=index, data=1)
    grouped = s.resample(freq).apply(lambda x: x.index[-1] if len(x) else None).dropna()
    return pd.DatetimeIndex(grouped.values)


def build_quantile_weights(signal_row: pd.Series, n_quantiles: int, min_names: int) -> pd.Series:
    """Equal-weighted long top quantile, short bottom quantile.
    Returns a weights series (sums to 0, gross = 2.0 -> 1.0 long / 1.0 short)."""
    s = signal_row.dropna()
    if len(s) < min_names * n_quantiles:
        return pd.Series(dtype=float)

    ranks = s.rank(pct=True)
    top = s[ranks >= 1 - 1.0 / n_quantiles]
    bottom = s[ranks <= 1.0 / n_quantiles]
    if len(top) < min_names or len(bottom) < min_names:
        return pd.Series(dtype=float)

    w = pd.Series(0.0, index=s.index)
    w[top.index] = 1.0 / len(top)
    w[bottom.index] = -1.0 / len(bottom)
    return w


def simulate_long_short(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    cfg: BacktestConfig,
) -> dict:
    """
    Simulates a rebalanced long/short quantile portfolio with explicit
    share-count tracking so per-share transaction costs can be applied.

    Returns dict with:
        'daily_returns'   : net-of-cost daily portfolio return series
        'gross_returns'   : pre-cost daily portfolio return series
        'turnover'        : per-rebalance turnover (fraction of gross book traded)
        'costs'           : per-rebalance dollar transaction costs
        'weights_history' : dict[date -> weights series] for diagnostics
    """
    rdates = rebalance_dates(prices.index, cfg.rebalance_freq)
    rdates = rdates[rdates.isin(signal.dropna(how="all").index)]

    daily_rets = compute_daily_returns(prices)

    portfolio_value = cfg.starting_capital
    shares = pd.Series(0.0, index=prices.columns)  # signed share counts (long/short)

    daily_port_ret = pd.Series(0.0, index=prices.index)
    gross_port_ret = pd.Series(0.0, index=prices.index)
    turnover_log = {}
    cost_log = {}
    weights_history = {}

    rdates_set = set(rdates)
    all_dates = prices.index

    for i in range(1, len(all_dates)):
        today = all_dates[i]
        yesterday = all_dates[i - 1]

        # 1. mark-to-market P&L from yesterday's positions over today's return
        px_today = prices.loc[today]
        px_yday = prices.loc[yesterday]
        valid = shares.ne(0) & px_today.notna() & px_yday.notna()
        pnl = (shares[valid] * (px_today[valid] - px_yday[valid])).sum()

        pre_rebalance_value = portfolio_value + pnl
        gross_ret_today = pnl / portfolio_value if portfolio_value != 0 else 0.0

        cost_today = 0.0
        if today in rdates_set:
            sig_row = signal.loc[today] if today in signal.index else pd.Series(dtype=float)
            target_weights = build_quantile_weights(sig_row, cfg.n_quantiles, cfg.min_names_per_leg)

            if len(target_weights) > 0:
                px = px_today.reindex(target_weights.index)
                valid_px = px.notna() & (px > 0)
                target_weights = target_weights[valid_px]
                px = px[valid_px]

                target_dollar = target_weights * pre_rebalance_value
                target_shares = (target_dollar / px).round(0)  # whole shares

                all_names = shares.index.union(target_shares.index)
                old_shares = shares.reindex(all_names).fillna(0.0)
                new_shares = pd.Series(0.0, index=all_names)
                new_shares.update(target_shares)

                traded_shares = (new_shares - old_shares).abs()
                cost_today = traded_shares.sum() * cfg.cost_per_share

                traded_dollar = (traded_shares * prices.loc[today].reindex(all_names)).sum()
                gross_book = pre_rebalance_value * 2.0  # 1.0 long + 1.0 short notional
                turnover_log[today] = traded_dollar / gross_book if gross_book > 0 else np.nan
                cost_log[today] = cost_today
                weights_history[today] = target_weights

                shares = new_shares.reindex(prices.columns).fillna(0.0)

        portfolio_value = pre_rebalance_value - cost_today
        daily_port_ret.loc[today] = gross_ret_today - (cost_today / (pre_rebalance_value if pre_rebalance_value != 0 else 1))
        gross_port_ret.loc[today] = gross_ret_today

    return {
        "daily_returns": daily_port_ret,
        "gross_returns": gross_port_ret,
        "turnover": pd.Series(turnover_log),
        "costs": pd.Series(cost_log),
        "weights_history": weights_history,
    }


# =========================================================================
# 6. PERFORMANCE METRICS
# =========================================================================

def max_drawdown_and_duration(cum_curve: pd.Series):
    """Returns (max_drawdown [negative fraction], max_duration_in_days)."""
    running_max = cum_curve.cummax()
    drawdown = cum_curve / running_max - 1.0
    max_dd = drawdown.min()

    # duration: longest stretch from a peak to the point equity recovers
    in_dd = drawdown < 0
    max_duration = 0
    current_start = None
    for date, flag in in_dd.items():
        if flag and current_start is None:
            current_start = date
        elif not flag and current_start is not None:
            duration = (date - current_start).days
            max_duration = max(max_duration, duration)
            current_start = None
    if current_start is not None:  # still in drawdown at end of sample
        duration = (cum_curve.index[-1] - current_start).days
        max_duration = max(max_duration, duration)

    return max_dd, max_duration


def performance_metrics(daily_returns: pd.Series, cfg: BacktestConfig, turnover: pd.Series = None) -> dict:
    r = daily_returns.dropna()
    if len(r) < 2:
        return {k: np.nan for k in [
            "annualised_return", "annualised_volatility", "annualised_sharpe",
            "max_drawdown", "max_drawdown_duration_days", "avg_turnover_per_rebalance", "n_obs"
        ]}

    ppy = cfg.trading_days_per_year
    ann_return = (1 + r).prod() ** (ppy / len(r)) - 1
    ann_vol = r.std() * np.sqrt(ppy)
    rf = cfg.risk_free_rate_annual
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan

    cum_curve = (1 + r).cumprod()
    max_dd, max_dd_dur = max_drawdown_and_duration(cum_curve)

    avg_turnover = turnover.mean() if turnover is not None and len(turnover) > 0 else np.nan

    return {
        "annualised_return": ann_return,
        "annualised_volatility": ann_vol,
        "annualised_sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_duration_days": max_dd_dur,
        "avg_turnover_per_rebalance": avg_turnover,
        "n_obs": len(r),
    }


# =========================================================================
# 7. CONDITIONING ON VIX DECILE
# =========================================================================

def metrics_by_vix_decile(
    daily_returns: pd.Series,
    turnover: pd.Series,
    vix_decile: pd.Series,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """
    For each VIX decile, computes performance metrics using ONLY the
    daily returns realised on dates whose contemporaneous (walk-forward,
    no-lookahead) VIX decile equals that bucket. This answers: "on days
    when the market was in vol-decile d, how did this strategy do?"

    Note: because deciles are computed walk-forward and can shift day to
    day, a single rebalance period may straddle multiple deciles; the
    daily-return slicing here is the more granular and defensible way to
    condition on regime (vs. bucketing whole rebalance periods).
    """
    aligned_decile = vix_decile.reindex(daily_returns.index)
    rows = {}
    for d in range(1, cfg.vix_n_deciles + 1):
        mask = aligned_decile == d
        sub_returns = daily_returns[mask]
        # turnover: attribute a rebalance's turnover to the decile on that rebalance date
        sub_turnover = turnover[turnover.index.map(lambda dt: aligned_decile.get(dt) == d)] if turnover is not None else None
        rows[d] = performance_metrics(sub_returns, cfg, sub_turnover)
    return pd.DataFrame(rows).T.rename_axis("vix_decile")


# =========================================================================
# 8. MAIN PIPELINE
# =========================================================================

def run_full_analysis(prices: pd.DataFrame, vix: pd.Series, cfg: BacktestConfig = None):
    cfg = cfg or BacktestConfig()

    common_idx = prices.index.intersection(vix.index)
    prices = prices.loc[common_idx]
    vix = vix.loc[common_idx]

    print(f"Universe: {prices.shape[1]} names, {len(common_idx)} trading days "
          f"({common_idx[0].date()} to {common_idx[-1].date()})")

    mom_sig = momentum_signal(prices, cfg.momentum_lookback, cfg.momentum_skip)
    rev_sig = reversal_signal(prices, cfg.reversal_lookback)
    vix_dec = vix_expanding_deciles(vix, cfg.vix_n_deciles)

    print("Simulating momentum strategy...")
    mom_result = simulate_long_short(prices, mom_sig, cfg)
    print("Simulating reversal strategy...")
    rev_result = simulate_long_short(prices, rev_sig, cfg)

    mom_by_decile = metrics_by_vix_decile(mom_result["daily_returns"], mom_result["turnover"], vix_dec, cfg)
    rev_by_decile = metrics_by_vix_decile(rev_result["daily_returns"], rev_result["turnover"], vix_dec, cfg)

    mom_overall = performance_metrics(mom_result["daily_returns"], cfg, mom_result["turnover"])
    rev_overall = performance_metrics(rev_result["daily_returns"], cfg, rev_result["turnover"])

    comparison = pd.DataFrame({
        "momentum_sharpe": mom_by_decile["annualised_sharpe"],
        "reversal_sharpe": rev_by_decile["annualised_sharpe"],
    })
    comparison["better_strategy"] = np.where(
        comparison["momentum_sharpe"] > comparison["reversal_sharpe"], "momentum",
        np.where(comparison["reversal_sharpe"] > comparison["momentum_sharpe"], "reversal", "tie/NaN")
    )

    return {
        "momentum_by_decile": mom_by_decile,
        "reversal_by_decile": rev_by_decile,
        "momentum_overall": mom_overall,
        "reversal_overall": rev_overall,
        "comparison": comparison,
        "vix_decile_series": vix_dec,
        "momentum_result": mom_result,
        "reversal_result": rev_result,
    }


def plot_results(results: dict, save_path: str = "vix_decile_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    comp = results["comparison"]
    x = comp.index
    axes[0].bar(x - 0.15, comp["momentum_sharpe"], width=0.3, label="Momentum")
    axes[0].bar(x + 0.15, comp["reversal_sharpe"], width=0.3, label="Reversal")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("VIX decile (1 = calmest, 10 = most volatile)")
    axes[0].set_ylabel("Annualised Sharpe ratio")
    axes[0].set_title("Sharpe Ratio by VIX Decile")
    axes[0].set_xticks(range(1, 11))
    axes[0].legend()

    mom_curve = (1 + results["momentum_result"]["daily_returns"].fillna(0)).cumprod()
    rev_curve = (1 + results["reversal_result"]["daily_returns"].fillna(0)).cumprod()
    axes[1].plot(mom_curve.index, mom_curve.values, label="Momentum")
    axes[1].plot(rev_curve.index, rev_curve.values, label="Reversal")
    axes[1].set_title("Cumulative Growth of $1 (net of costs)")
    axes[1].set_ylabel("Growth of $1")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


# =========================================================================
# 9. DEMO ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ASX MOMENTUM vs REVERSAL BY VIX DECILE -- live data via yfinance")
    print("=" * 70)

    cfg = BacktestConfig()

    try:
        prices = load_price_panel(start="2015-01-01")
        vix = load_vix_series(start="2015-01-01")
    except Exception as e:
        raise SystemExit(
            "\nFailed to download data via yfinance.\n"
            "This requires outbound internet access to Yahoo Finance's\n"
            "servers (query1/query2.finance.yahoo.com). If you're running\n"
            "this inside a network-restricted environment, run it on your\n"
            "own machine or a notebook with normal internet access instead.\n"
            f"\nUnderlying error: {e}"
        )

    if prices.shape[1] < cfg.min_names_per_leg * cfg.n_quantiles:
        raise SystemExit(
            f"Only got price data for {prices.shape[1]} tickers, which is too "
            f"few to form {cfg.n_quantiles} quantile portfolios with at least "
            f"{cfg.min_names_per_leg} names per leg. Expand DEFAULT_ASX_TICKERS."
        )

    results = run_full_analysis(prices, vix, cfg)

    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    print("\n--- Momentum: metrics by VIX decile ---")
    print(results["momentum_by_decile"])

    print("\n--- Reversal: metrics by VIX decile ---")
    print(results["reversal_by_decile"])

    print("\n--- Overall (all deciles pooled) ---")
    print("Momentum:", results["momentum_overall"])
    print("Reversal:", results["reversal_overall"])

    print("\n--- Sharpe comparison & implied better strategy per decile ---")
    print(results["comparison"])

    results["momentum_by_decile"].to_csv(f"{data_output}/momentum_by_vix_decile.csv")
    results["reversal_by_decile"].to_csv(f"{data_output}/reversal_by_vix_decile.csv")
    results["comparison"].to_csv(f"{data_output}/strategy_comparison_by_vix_decile.csv")

    plot_results(results)