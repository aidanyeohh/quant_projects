"""
Regime-Switching Momentum/Reversal Backtest -- ASX200 Universe
================================================================

Thesis: a composite strategy that trades per-name channel-breakout
MOMENTUM on days when the ASX VIX is in a LOW-volatility regime, and
switches to per-name Bollinger-Band REVERSAL on days when the ASX VIX
is in a HIGH-volatility regime, produces stronger risk-adjusted
performance than either pure momentum, pure reversal, or a passive
buy-and-hold of the same universe, on a cost-adjusted basis.

This builds on `vix_decile_backtest.py` (VIX decile classification,
walk-forward/no-lookahead) but replaces its cross-sectional,
continuous-score momentum/reversal signals with two per-name, rule-based
trading systems -- these produce discrete positions (not ranks), so they
need their own signal construction and simulation engine rather than the
quantile long/short machinery:

  1. MOMENTUM = Donchian-style channel breakout (long/flat, per name):
       - ENTER long when today's HIGH >= the highest HIGH of the prior
         `channel_entry_lookback` (Y) days.
       - EXIT to flat when today's LOW <= the lowest LOW of the prior
         `channel_exit_lookback` (X) days, X < Y.
     This is long-only by construction -- the rule as specified never
     shorts.
  2. REVERSAL = Bollinger Bands (long/flat, per name), using a
     `bollinger_window`-day SMA +/- `bollinger_num_std` standard
     deviations:
       - ENTER long (oversold) when close <= lower band.
       - EXIT to flat when close crosses back through the middle band.
     Both systems in this module are run LONG-ONLY -- Bollinger's short
     leg (short the overbought break, symmetric to the above) is still
     available via `bollinger_positions(..., long_only=False)` but isn't
     used by the default pipeline here.
  3. A composite REGIME-SWITCH signal: the channel-breakout position on
     low-VIX-decile days, the Bollinger position on high-VIX-decile
     days, decile threshold configurable via
     `RegimeConfig.low_vol_max_decile`.
  4. A live S&P/ASX 200 constituent list scraped from Wikipedia (instead
     of the ~50-name hand-picked sample), for closer coverage of the
     actual index. This is still a CURRENT snapshot with no
     delisted/removed names, so it carries survivorship bias -- see the
     note in vix_decile_backtest.py for how to fix this properly.
  5. An event-driven, FIXED-SLOT, weight-drift backtest engine: because
     breakout/band triggers can fire on any day (not just a fixed
     calendar schedule), a name is only actually TRADED on the day its
     own signal changes -- entries are sized at a fixed fraction of the
     book (1/`position_slots_per_leg`) and then just drift with price,
     like buy-and-hold, until that name's own exit. Other names already
     held are never re-touched just because some other name entered or
     exited elsewhere in the universe -- an earlier equal-weight-the-
     whole-book version re-normalised every held position on every
     single change anywhere in a ~100-200 name universe, which happens
     on the vast majority of trading days, and the resulting turnover
     was enough on its own to erase a genuinely positive gross Sharpe.
     Transaction costs are charged in bps of traded notional: 20bps
     all-in for market orders (~7bps commission + ~13bps assumed
     slippage), 7bps for limit orders (commission only).
  6. A passive, equal-weighted, long-only buy-and-hold benchmark on the
     same universe.
  7. The full performance-evaluation panel required by the assignment:
     annualised return, volatility, Sharpe, max drawdown, and CAPM
     alpha/beta vs the S&P/ASX 200 index (^AXJO).

-----------------------------------------------------------------------
Run this file directly to pull live ASX200 + ^AXVI + ^AXJO data via
yfinance/Wikipedia and run the full comparison end to end.
-----------------------------------------------------------------------
"""

import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from dataclasses import dataclass

from vix_decile_backtest import (
    BacktestConfig,
    DEFAULT_ASX_TICKERS,
    data_output,
    load_vix_series,
    compute_daily_returns,
    vix_expanding_deciles,
    max_drawdown_and_duration,
    performance_metrics,
    metrics_by_vix_decile,
)


# =========================================================================
# 1. CONFIGURATION
# =========================================================================

@dataclass
class RegimeConfig(BacktestConfig):
    cost_bps_market: float = 20.0    # commissions (~7bps) + assumed slippage (~13bps)
    cost_bps_limit: float = 7.0      # commissions only
    low_vol_max_decile: int = 5      # VIX deciles 1..low_vol_max_decile => "low vol" => momentum
                                      # deciles low_vol_max_decile+1..10  => "high vol" => reversal

    # channel-breakout momentum (Donchian-style, long/flat). X < Y.
    channel_entry_lookback: int = 55   # Y: N-day highest-high breakout to enter long
    channel_exit_lookback: int = 20    # X: N-day lowest-low breakdown to exit to flat

    # Bollinger-band reversal (long-only by default; see bollinger_positions)
    bollinger_window: int = 20         # SMA/std lookback for the bands
    bollinger_num_std: float = 2.0     # band half-width, in standard deviations

    # Fixed-slot position sizing (see simulate_positions_bps): each entry
    # is sized at 1/position_slots_per_leg of the book and holds that
    # weight (aside from price drift) until its own exit -- NOT an
    # enforced cap, just the sizing denominator. If more names are
    # simultaneously active than there are slots, gross leg exposure
    # floats above 1.0x. Tune this to roughly the average number of
    # names you expect to see flagged at once for a given universe size.
    position_slots_per_leg: int = 20

    # NOTE: momentum_lookback / momentum_skip / reversal_lookback are
    # inherited from BacktestConfig but unused here -- they belonged to
    # vix_decile_backtest.py's continuous-score signals, superseded above.


# =========================================================================
# 2. DATA LOADING
# =========================================================================

def load_asx200_tickers() -> list:
    """
    Scrapes the current S&P/ASX 200 constituent list from Wikipedia and
    maps each code to its Yahoo Finance ticker (`.AX` suffix). Wikipedia
    403s requests without a descriptive User-Agent header.

    This is a CURRENT snapshot only -- it has no delisted/removed names,
    so results carry survivorship bias, same caveat as
    vix_decile_backtest.DEFAULT_ASX_TICKERS. For a real thesis, source a
    point-in-time index history instead.
    """
    url = "https://en.wikipedia.org/wiki/S%26P/ASX_200"
    headers = {"User-Agent": "Mozilla/5.0 (research script; contact: yoyoman021@gmail.com)"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        constituents = next(t for t in tables if "Code" in t.columns)
        codes = constituents["Code"].astype(str).str.strip()
        codes = codes[codes.ne("")]
        tickers = [f"{c}.AX" for c in codes.tolist()]
        print(f"Loaded {len(tickers)} ASX200 constituents from Wikipedia.")
        return tickers
    except Exception as e:
        print(f"Warning: failed to scrape ASX200 list from Wikipedia ({e}); "
              f"falling back to the {len(DEFAULT_ASX_TICKERS)}-name sample "
              f"in vix_decile_backtest.DEFAULT_ASX_TICKERS.")
        return DEFAULT_ASX_TICKERS


def load_ohlc_panel(
    tickers=None,
    start: str = "2015-01-01",
    end: str = None,
    interval: str = "1d",
) -> dict:
    """
    Pulls adjusted High, Low and Close for `tickers` from Yahoo Finance.
    High/Low are needed for the channel-breakout signal --
    vix_decile_backtest.load_price_panel only pulls Close, which was
    sufficient for that module's continuous-return signals but not for a
    breakout defined on intraday highs/lows. Uses the same
    auto_adjust=True back-adjustment as the rest of this codebase so
    High/Low/Close stay mutually consistent through split/dividend
    history.

        pip install yfinance
    """
    import yfinance as yf

    tickers = tickers or DEFAULT_ASX_TICKERS
    print(f"Downloading {len(tickers)} ASX tickers (OHLC) from yfinance "
          f"({start} to {end or 'today'})...")

    raw = yf.download(
        tickers, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False, group_by="ticker", threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        present = [t for t in tickers if t in raw.columns.get_level_values(0)]
        close = pd.DataFrame({t: raw[t]["Close"] for t in present})
        high = pd.DataFrame({t: raw[t]["High"] for t in present})
        low = pd.DataFrame({t: raw[t]["Low"] for t in present})
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        high = raw[["High"]].rename(columns={"High": tickers[0]})
        low = raw[["Low"]].rename(columns={"Low": tickers[0]})

    close = close.dropna(axis=1, how="all")
    high = high.reindex(columns=close.columns)
    low = low.reindex(columns=close.columns)

    dropped = set(tickers) - set(close.columns)
    if dropped:
        print(f"Warning: no data returned for {sorted(dropped)}")

    return {"high": high, "low": low, "close": close}


def sanitize_ohlc_panel(ohlc: dict) -> dict:
    """
    Masks non-positive Close prices as missing (yfinance's
    auto_adjust=True back-adjustment occasionally produces negative or
    zero "adjusted close" for illiquid penny stocks, e.g. MEZ.AX, which
    would otherwise show up as a >100% single-day loss). High/Low are
    masked wherever the corresponding Close is missing, or wherever
    they're themselves non-positive, so a bad print can't trigger a
    breakout/band signal off an untrustworthy bar.
    """
    close = ohlc["close"]
    bad = (close <= 0).sum().sum()
    if bad:
        print(f"Warning: masking {bad} non-positive close observations "
              f"(likely yfinance adjustment artifacts) as missing.")
    close = close.where(close > 0)

    high = ohlc["high"].reindex_like(close).where(close.notna())
    low = ohlc["low"].reindex_like(close).where(close.notna())
    high = high.where(high > 0)
    low = low.where(low > 0)

    return {"high": high, "low": low, "close": close}


def load_benchmark_series(start: str = "2015-01-01", end: str = None) -> pd.Series:
    """
    Pulls the S&P/ASX 200 index level (^AXJO) via yfinance, used as both
    the buy-and-hold market reference point and the CAPM benchmark for
    alpha/beta. Note ^AXJO is a price index (excludes dividends),
    consistent with the price-only equity data used elsewhere here.
    """
    import yfinance as yf

    print("Downloading ^AXJO (S&P/ASX 200 index) from yfinance...")
    raw = yf.download("^AXJO", start=start, end=end, auto_adjust=True, progress=False)
    s = raw[("Close", "^AXJO")] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
    s.name = "AXJO"
    return s.dropna()


# =========================================================================
# 3. SIGNAL CONSTRUCTION -- discrete per-name positions
# =========================================================================

def channel_breakout_positions(high: pd.DataFrame, low: pd.DataFrame,
                                entry_lookback: int, exit_lookback: int) -> pd.DataFrame:
    """
    Donchian-style channel breakout, long/flat, per name:
        ENTER long when today's high >= the highest high of the prior
            `entry_lookback` (Y) days.
        EXIT to flat when today's low <= the lowest low of the prior
            `exit_lookback` (X) days (X < entry_lookback).
    The rolling thresholds are shifted by one day so "prior Y/X days"
    never includes today -- only today's actual high/low is compared
    against a threshold built purely from the past, so there's no
    look-ahead in the threshold itself. A missing high/low on a given
    day simply can't trigger entry/exit that day (comparison is False),
    holding whatever state the name was already in.
    Returns a DataFrame of {0.0 = flat, 1.0 = long}.
    """
    entry_threshold = high.rolling(entry_lookback).max().shift(1)
    exit_threshold = low.rolling(exit_lookback).min().shift(1)
    entry_signal = (high >= entry_threshold).to_numpy()
    exit_signal = (low <= exit_threshold).to_numpy()

    # Vectorized long/flat hysteresis: state(t) = long if the most recent
    # entry signal (of either type) is more recent than the most recent
    # exit signal. This is exactly equivalent to the sequential
    # "if flat: check entry; elif long: check exit" state machine --
    # entry signals while already long, and exit signals while already
    # flat, don't change the state either way, so only the relative
    # ordering of the two signal types' last occurrence matters. Avoids
    # a per-name Python loop, which matters here since parameter sweeps
    # call this once per (X, Y) grid point.
    n_rows = high.shape[0]
    day_idx = np.arange(n_rows)[:, None]
    last_entry_idx = np.maximum.accumulate(np.where(entry_signal, day_idx, -1), axis=0)
    last_exit_idx = np.maximum.accumulate(np.where(exit_signal, day_idx, -1), axis=0)
    pos = (last_entry_idx > last_exit_idx).astype(float)

    return pd.DataFrame(pos, index=high.index, columns=high.columns)


def bollinger_positions(close: pd.DataFrame, window: int, num_std: float,
                         long_only: bool = False) -> pd.DataFrame:
    """
    Bollinger Band reversal, per name:
        ENTER long (oversold) when close <= lower band.
        EXIT to flat when close crosses back through the middle band.
    With `long_only=False`, also symmetrically:
        ENTER short (overbought) when close >= upper band.
        EXIT to flat when close crosses back through the middle band.
    Bands use the standard (unshifted) rolling SMA/std ending today --
    the resulting position, like the breakout signal above, is only
    applied to the portfolio from the following trading day onward (see
    simulate_positions_bps), so using today's close to size today's
    signal doesn't leak into today's realised return.

    Both strategies in this module are run long-only by default (see
    channel_breakout_positions, which is long/flat by construction), so
    `long_only` defaults to True here too, giving a {0.0 = flat, 1.0 =
    long} DataFrame. The long-only case is exactly the same two-event
    long/flat hysteresis as channel_breakout_positions (entry = cross
    below the lower band, exit = cross back above the mid band), so it
    reuses the same vectorized "most recent event wins" trick rather
    than a per-name loop. `long_only=False` returns {-1.0, 0.0, 1.0} but
    needs the explicit per-name loop below -- a 3-state hysteresis isn't
    reducible to the same two-event trick.
    """
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std

    long_entry = (close <= lower).to_numpy()
    exit_signal = (close >= mid).to_numpy()

    if long_only:
        n_rows = close.shape[0]
        day_idx = np.arange(n_rows)[:, None]
        last_entry_idx = np.maximum.accumulate(np.where(long_entry, day_idx, -1), axis=0)
        last_exit_idx = np.maximum.accumulate(np.where(exit_signal, day_idx, -1), axis=0)
        pos = (last_entry_idx > last_exit_idx).astype(float)
        return pd.DataFrame(pos, index=close.index, columns=close.columns)

    short_entry = (close >= upper).to_numpy()
    close_np = close.to_numpy()
    mid_np = mid.to_numpy()

    n_rows, n_cols = close.shape
    pos = np.zeros((n_rows, n_cols))
    for c in range(n_cols):
        state = 0.0
        le = long_entry[:, c]
        se = short_entry[:, c]
        px = close_np[:, c]
        mline = mid_np[:, c]
        col = pos[:, c]
        for i in range(n_rows):
            if np.isnan(mline[i]):
                col[i] = state
                continue
            if state == 0.0:
                if le[i]:
                    state = 1.0
                elif se[i]:
                    state = -1.0
            elif state == 1.0:
                if px[i] >= mline[i]:
                    state = 0.0
            elif state == -1.0:
                if px[i] <= mline[i]:
                    state = 0.0
            col[i] = state

    return pd.DataFrame(pos, index=close.index, columns=close.columns)


def regime_switch_positions(
    mom_pos: pd.DataFrame,
    rev_pos: pd.DataFrame,
    vix_decile: pd.Series,
    low_vol_max_decile: int,
) -> pd.DataFrame:
    """
    Composite per-name position: on days whose walk-forward VIX decile
    is <= low_vol_max_decile ("calm"), use the channel-breakout
    (momentum) position; on higher-decile ("stressed") days, use the
    Bollinger (reversal) position. Days before VIX has enough history to
    assign a decile (NaN) default to momentum. Both inputs are already
    point-in-time / no-lookahead by construction (see
    channel_breakout_positions, bollinger_positions,
    vix_expanding_deciles in vix_decile_backtest.py).
    """
    aligned_decile = vix_decile.reindex(mom_pos.index)
    is_low_vol = aligned_decile.le(low_vol_max_decile) | aligned_decile.isna()
    low_vol_mask = pd.DataFrame(
        np.repeat(is_low_vol.values[:, None], mom_pos.shape[1], axis=1),
        index=mom_pos.index, columns=mom_pos.columns,
    )
    return mom_pos.where(low_vol_mask, rev_pos)


# =========================================================================
# 4. BACKTEST ENGINE -- event-driven, weight-drift, bps costs
# =========================================================================

def simulate_positions_bps(prices: pd.DataFrame, positions: pd.DataFrame, cfg: RegimeConfig,
                            cost_bps: float = None, position_slots_per_leg: int = None) -> dict:
    """
    Event-driven, FIXED-SLOT simulation for discrete per-name position
    signals. Every long entry is sized at +1/`position_slots_per_leg` of
    the book, every short entry at -1/`position_slots_per_leg`, fixed at
    the moment of entry -- a name's weight then just drifts with price
    (like buy_and_hold) until ITS OWN exit trades it back to zero. No
    other name's weight is touched in between.

    This is deliberately NOT "equal-weight across whatever's currently
    held": an earlier version re-derived each name's weight as
    1/n_currently_long (or n_short) and re-applied that to the WHOLE book
    every time *any* name entered or exited anywhere in the universe.
    With ~100-200 names and a signal that fires often (Bollinger reversal
    especially), that meant re-touching essentially the entire book on
    ~97% of trading days even though only a handful of names actually
    changed state -- turnover/costs from that alone ran ~20%+/year and
    were enough on their own to turn a genuinely positive gross Sharpe
    into a near-zero or negative net Sharpe. Fixed-slot sizing removes
    that "phantom rebalancing": a held position's weight only ever moves
    because of its own price drift, or because it's the one entering or
    exiting.

    `position_slots_per_leg` isn't an enforced cap -- if more names are
    simultaneously flagged than there are slots, gross exposure on that
    leg floats above 1.0x (a fixed-fractional-risk-per-trade model, same
    idea real systematic breakout/reversion books use, rather than
    arbitrarily picking which signals to skip). Defaults to
    `cfg.position_slots_per_leg`.

    Works on raw numpy arrays rather than per-day pandas Series lookups
    -- this loop is the hot path for parameter sweeps (called once per
    grid point), and pandas' per-call index-alignment overhead dominates
    runtime at that scale.
    """
    cost_bps = cfg.cost_bps_market if cost_bps is None else cost_bps
    slots = cfg.position_slots_per_leg if position_slots_per_leg is None else position_slots_per_leg
    slot_weight = 1.0 / slots

    names = prices.columns
    positions = positions.reindex(columns=names)

    daily_asset_rets = compute_daily_returns(prices).reindex(columns=names).fillna(0.0).to_numpy()
    pos_arr = positions.fillna(0.0).to_numpy()
    target_arr = pos_arr * slot_weight

    # a name is only ever traded on the day ITS OWN position changes
    own_changed = np.zeros_like(pos_arr, dtype=bool)
    own_changed[1:] = pos_arr[1:] != pos_arr[:-1]

    n_days, n_names = daily_asset_rets.shape
    weights = np.zeros(n_names)
    daily_port_ret = np.zeros(n_days)
    gross_port_ret = np.zeros(n_days)
    cost_frac = cost_bps / 10_000.0
    turnover_log = {}
    cost_log = {}
    dates = prices.index

    for i in range(1, n_days):
        ret_today = daily_asset_rets[i]
        gross_ret_today = float(weights @ ret_today)
        # weights drift with today's price moves; only names that flip
        # entry/exit today actually trade
        weights = weights * (1.0 + ret_today)

        cost_today = 0.0
        chg = own_changed[i]
        if chg.any():
            traded_weight = np.abs(target_arr[i, chg] - weights[chg]).sum()
            turnover_log[dates[i]] = traded_weight / 2.0  # fraction of gross book (2.0) traded
            cost_today = traded_weight * cost_frac
            cost_log[dates[i]] = cost_today
            weights[chg] = target_arr[i, chg]

        gross_port_ret[i] = gross_ret_today
        daily_port_ret[i] = gross_ret_today - cost_today

    return {
        "daily_returns": pd.Series(daily_port_ret, index=dates),
        "gross_returns": pd.Series(gross_port_ret, index=dates),
        "turnover": pd.Series(turnover_log),
        "costs": pd.Series(cost_log),
    }


def buy_and_hold(prices: pd.DataFrame, cfg: RegimeConfig, cost_bps: float = None) -> dict:
    """
    Fully-invested, equal-weighted, long-only buy-and-hold of the
    universe: allocate equal dollar value across every name with a
    valid price on day 1, pay a one-time entry cost (limit-order
    assumption, 7bps), then hold with no further rebalancing. This is
    the passive benchmark the thesis needs to beat.

    Tracked in DOLLAR VALUE per name (not weight): a name's value can
    drift arbitrarily far above its starting share of the book -- a big
    multi-year winner dominating an unrebalanced book is a real,
    expected feature of buy-and-hold -- but can never go negative.
    A naive weight-drift formulation instead lets a winner's *weight*
    balloon past 100% of the original book and then, on giving some of
    it back, implies a portfolio return below -100% in a single day --
    a mathematically invalid outcome for an unlevered long-only book,
    which this dollar-value formulation rules out by construction.
    """
    cost_bps = cfg.cost_bps_limit if cost_bps is None else cost_bps

    valid0 = prices.iloc[0].notna()
    n = valid0.sum()
    entry_value = (1.0 / n) * (1 - cost_bps / 10_000.0)
    asset_value = pd.Series(np.where(valid0, entry_value, 0.0), index=prices.columns)

    daily_asset_rets = compute_daily_returns(prices)
    port_value = pd.Series(index=prices.index, dtype=float)
    port_value.iloc[0] = asset_value.sum()

    for i in range(1, len(prices.index)):
        today = prices.index[i]
        asset_ret_today = daily_asset_rets.loc[today].reindex(asset_value.index).fillna(0.0)
        asset_value = asset_value * (1 + asset_ret_today)
        port_value.loc[today] = asset_value.sum()

    # the one-time entry cost is a uniform scale factor applied to every
    # name's starting dollar value, so it only shows up in the day-0
    # return -- every subsequent pct_change is identical whether or not
    # the cost was ever charged, which is exactly what "gross" needs.
    daily_port_ret = port_value.pct_change()
    gross_port_ret = daily_port_ret.copy()
    gross_port_ret.iloc[0] = 0.0
    daily_port_ret.iloc[0] = port_value.iloc[0] - 1.0  # entry cost, relative to $1 invested

    return {
        "daily_returns": daily_port_ret,
        "gross_returns": gross_port_ret,
        "turnover": pd.Series(dtype=float),
        "costs": pd.Series(dtype=float),
    }


# =========================================================================
# 5. PERFORMANCE METRICS -- extends vix_decile_backtest with alpha/beta
# =========================================================================

def performance_metrics_with_alpha_beta(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series,
    cfg: RegimeConfig,
    turnover: pd.Series = None,
) -> dict:
    """CAPM alpha/beta of `daily_returns` vs `benchmark_returns` via OLS,
    layered on top of vix_decile_backtest.performance_metrics."""
    metrics = performance_metrics(daily_returns, cfg, turnover)

    df = pd.concat([daily_returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1).dropna()
    if len(df) > 20:
        beta, alpha_daily = np.polyfit(df["benchmark"].values, df["strategy"].values, 1)
        alpha_annualised = alpha_daily * cfg.trading_days_per_year
    else:
        beta, alpha_annualised = np.nan, np.nan

    metrics["alpha_annualised"] = alpha_annualised
    metrics["beta"] = beta
    return metrics


# =========================================================================
# 6. MAIN PIPELINE
# =========================================================================

def run_regime_switch_analysis(ohlc: dict, vix: pd.Series, benchmark: pd.Series,
                                cfg: RegimeConfig = None) -> dict:
    cfg = cfg or RegimeConfig()

    ohlc = sanitize_ohlc_panel(ohlc)
    prices, high, low = ohlc["close"], ohlc["high"], ohlc["low"]

    common_idx = prices.index.intersection(vix.index)
    prices = prices.loc[common_idx]
    high = high.loc[common_idx]
    low = low.loc[common_idx]
    vix = vix.loc[common_idx]
    bench_ret = benchmark.pct_change().reindex(common_idx)

    print(f"Universe: {prices.shape[1]} names, {len(common_idx)} trading days "
          f"({common_idx[0].date()} to {common_idx[-1].date()})")

    mom_pos = channel_breakout_positions(high, low, cfg.channel_entry_lookback, cfg.channel_exit_lookback)
    rev_pos = bollinger_positions(prices, cfg.bollinger_window, cfg.bollinger_num_std)
    vix_dec = vix_expanding_deciles(vix, cfg.vix_n_deciles)
    combo_pos = regime_switch_positions(mom_pos, rev_pos, vix_dec, cfg.low_vol_max_decile)

    print("Simulating momentum-only (channel breakout)...")
    mom_result = simulate_positions_bps(prices, mom_pos, cfg)
    print("Simulating reversal-only (Bollinger bands)...")
    rev_result = simulate_positions_bps(prices, rev_pos, cfg)
    print("Simulating regime-switch composite...")
    combo_result = simulate_positions_bps(prices, combo_pos, cfg)
    print("Simulating buy-and-hold benchmark...")
    bh_result = buy_and_hold(prices, cfg)

    results = {
        "Momentum-only": mom_result,
        "Reversal-only": rev_result,
        "Regime-Switch": combo_result,
        "Buy-and-Hold": bh_result,
    }

    summary = pd.DataFrame({
        name: performance_metrics_with_alpha_beta(res["daily_returns"], bench_ret, cfg, res["turnover"])
        for name, res in results.items()
    }).T

    summary_gross = pd.DataFrame({
        name: performance_metrics_with_alpha_beta(res["gross_returns"], bench_ret, cfg, res["turnover"])
        for name, res in results.items()
    }).T

    regime_breakdown_sharpe = pd.DataFrame({
        name: metrics_by_vix_decile(res["daily_returns"], res["turnover"], vix_dec, cfg)["annualised_sharpe"]
        for name, res in results.items() if name != "Buy-and-Hold"
    })

    return {
        "summary": summary,
        "summary_gross": summary_gross,
        "regime_breakdown_sharpe": regime_breakdown_sharpe,
        "vix_decile_series": vix_dec,
        "results": results,
    }


def plot_results(analysis: dict, cfg: RegimeConfig, save_path: str = "regime_switch_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, res in analysis["results"].items():
        curve = (1 + res["daily_returns"].fillna(0)).cumprod()
        axes[0].plot(curve.index, curve.values, label=name)
    axes[0].set_title("Cumulative Growth of $1 (net of costs)")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()

    breakdown = analysis["regime_breakdown_sharpe"]
    x = np.arange(1, cfg.vix_n_deciles + 1)
    width = 0.25
    for i, col in enumerate(breakdown.columns):
        axes[1].bar(x + (i - 1) * width, breakdown[col], width=width, label=col)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axvline(cfg.low_vol_max_decile + 0.5, color="grey", linestyle="--", linewidth=1,
                     label="low/high vol split")
    axes[1].set_xlabel("VIX decile (1 = calmest, 10 = most volatile)")
    axes[1].set_ylabel("Annualised Sharpe ratio")
    axes[1].set_title("Sharpe Ratio by VIX Decile")
    axes[1].set_xticks(x)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


# =========================================================================
# 7. DEMO ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ASX200 REGIME-SWITCH (CHANNEL BREAKOUT / BOLLINGER) vs BUY-AND-HOLD")
    print("=" * 70)

    cfg = RegimeConfig()
    start_date = "2015-01-01"

    try:
        tickers = load_asx200_tickers()
        ohlc = load_ohlc_panel(tickers, start=start_date)
        vix = load_vix_series(start=start_date)
        benchmark = load_benchmark_series(start=start_date)
    except Exception as e:
        raise SystemExit(
            "\nFailed to download data via yfinance/Wikipedia.\n"
            "This requires outbound internet access (Yahoo Finance and\n"
            "en.wikipedia.org). If you're running this inside a\n"
            "network-restricted environment, run it on your own machine\n"
            "or a notebook with normal internet access instead.\n"
            f"\nUnderlying error: {e}"
        )

    if ohlc["close"].shape[1] < cfg.min_names_per_leg:
        raise SystemExit(
            f"Only got price data for {ohlc['close'].shape[1]} tickers, which is "
            f"too few to form a diversified equal-weight book (need at least "
            f"{cfg.min_names_per_leg})."
        )

    analysis = run_regime_switch_analysis(ohlc, vix, benchmark, cfg)

    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    pd.set_option("display.width", 160)

    print("\n--- Performance summary, NET of transaction costs (Momentum-only / Reversal-only / Regime-Switch / Buy-and-Hold) ---")
    print(analysis["summary"])

    print("\n--- Performance summary, GROSS of transaction costs (same order) ---")
    print(analysis["summary_gross"])

    print("\n--- Sharpe ratio by VIX decile ---")
    print(analysis["regime_breakdown_sharpe"])

    thesis_holds = (
        analysis["summary"].loc["Regime-Switch", "annualised_sharpe"] > analysis["summary"].loc["Momentum-only", "annualised_sharpe"]
        and analysis["summary"].loc["Regime-Switch", "annualised_sharpe"] > analysis["summary"].loc["Reversal-only", "annualised_sharpe"]
        and analysis["summary"].loc["Regime-Switch", "annualised_sharpe"] > analysis["summary"].loc["Buy-and-Hold", "annualised_sharpe"]
    )
    print(f"\nThesis check -- does Regime-Switch beat Momentum-only, Reversal-only "
          f"AND Buy-and-Hold on Sharpe? {thesis_holds}")

    analysis["summary"].to_csv(f"{data_output}/regime_switch_summary.csv")
    analysis["summary_gross"].to_csv(f"{data_output}/regime_switch_summary_gross.csv")
    analysis["regime_breakdown_sharpe"].to_csv(f"{data_output}/regime_switch_sharpe_by_decile.csv")

    plot_results(analysis, cfg, save_path=f"{data_output}/regime_switch_comparison.png")
