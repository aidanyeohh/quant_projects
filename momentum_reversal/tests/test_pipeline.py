"""End-to-end smoke test on synthetic data.

Runs the whole chain - clean, universe, signals, backtest, metrics,
survivorship - on a generated panel that deliberately contains the awkward
cases: names that delist mid-sample, names that list late, untraded days and a
planted reversal effect. Fast enough to run before every download.

    python -m pytest tests/test_pipeline.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from asxstat import backtest, clean, metrics, signals, survivorship
from asxstat.config import BacktestConfig, UniverseConfig
from asxstat.universe import point_in_time_universe

N_DAYS = 900
N_NAMES = 60
SEED = 7


@pytest.fixture(scope="module")
def panels() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2015-01-01", periods=N_DAYS)
    tickers = [f"T{i:03d}.AX" for i in range(N_NAMES)]

    market = rng.normal(0, 0.010, N_DAYS)
    idio = rng.normal(0, 0.020, (N_DAYS, N_NAMES))
    beta = rng.uniform(0.5, 1.5, N_NAMES)
    returns = market[:, None] * beta[None, :] + idio

    # Plant a one-day reversal effect so the reversal signal has something real
    # to find; without it a passing test proves only that nothing crashed.
    returns[1:] -= 0.25 * idio[:-1]

    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers
    )
    volume = pd.DataFrame(
        rng.lognormal(12, 1.0, (N_DAYS, N_NAMES)), index=dates, columns=tickers
    )

    # Ten names delist at staggered points; five list late.
    for offset, ticker in enumerate(tickers[:10]):
        cut = 400 + offset * 30
        close.iloc[cut:, close.columns.get_loc(ticker)] = np.nan
        volume.iloc[cut:, volume.columns.get_loc(ticker)] = np.nan
    for offset, ticker in enumerate(tickers[10:15]):
        start = 200 + offset * 20
        close.iloc[:start, close.columns.get_loc(ticker)] = np.nan
        volume.iloc[:start, volume.columns.get_loc(ticker)] = np.nan

    # Sprinkle untraded days: price prints, volume is zero.
    halt = rng.random((N_DAYS, N_NAMES)) < 0.01
    volume = volume.mask(halt, 0.0)

    return {"close": close, "adj_close": close, "volume": volume}


@pytest.fixture(scope="module")
def data(panels) -> clean.Dataset:
    return clean.build_dataset(panels)


def test_zero_volume_days_are_not_returns(data):
    """A day with no trade must not contribute a return observation."""
    untraded = ~data.traded
    assert data.returns.where(untraded).notna().sum().sum() == 0


def test_delisting_report_finds_dead_names(data):
    report = clean.delisting_report(data)
    assert report["delisted"].sum() == 10
    assert report.loc[report["delisted"], "last_trade"].max() < data.dates.max()


def test_universe_has_no_membership_lookahead(data):
    """A delisted name must not be a member after it stops trading."""
    cfg = UniverseConfig(size=30, liquidity_window=60, min_dollar_volume=0.0)
    members, deciles = point_in_time_universe(data.close, data.dollar_volume, cfg)

    assert members.sum(axis=1).max() <= 30
    assert members.sum(axis=1).median() > 0

    for ticker, last in data.last_trade.items():
        if pd.isna(last):
            continue
        after = members.loc[members.index > last + pd.Timedelta(45, "D"), ticker]
        assert not after.any(), f"{ticker} still a member long after delisting"

    assert deciles.max().max() <= 9


def test_reversal_signal_recovers_planted_effect(data):
    cfg = UniverseConfig(size=40, liquidity_window=60, min_dollar_volume=0.0)
    members, _ = point_in_time_universe(data.close, data.dollar_volume, cfg)

    rev = signals.reversal(data.returns, lookback=1)
    ic = backtest.information_coefficient(rev, data.returns, members, horizons=(1,))
    summary = backtest.ic_summary(ic)
    assert summary.loc["ic_1d", "mean_ic"] > 0
    assert summary.loc["ic_1d", "t_stat"] > 2


def test_backtest_book_is_dollar_neutral_and_costed(data):
    ucfg = UniverseConfig(size=40, liquidity_window=60, min_dollar_volume=0.0)
    members, deciles = point_in_time_universe(data.close, data.dollar_volume, ucfg)

    rev = signals.reversal(data.returns, lookback=1)
    # Daily rebalancing so every row is a freshly built book; on a weekly
    # schedule the held weights drift between rebalances by design.
    cfg = BacktestConfig(rebalance_freq="D")
    result = backtest.run_backtest(rev, data, members, deciles, cfg)

    active = result.weights[result.weights.abs().sum(axis=1) > 0]
    # Targets are exactly neutral, but a halted name cannot be traded out, so
    # the held book carries a small residual exposure. It must stay small and
    # must not be systematically one-sided.
    net = active.sum(axis=1)
    assert net.abs().median() < 1e-8, "book is not neutral on unhalted days"
    assert net.abs().max() <= 3 * cfg.max_weight
    assert abs(net.mean()) < 0.01, "residual exposure is systematically biased"
    # Stranded positions in halted names also push gross slightly above
    # nominal; the typical day must still sit exactly at the target.
    gross = active.abs().sum(axis=1)
    assert gross.median() == pytest.approx(cfg.gross_leverage, rel=1e-3)
    assert gross.max() <= cfg.gross_leverage * 1.25
    # A stranded position keeps drifting, so it can breach the cap between
    # the days it can be traded. It must not run away from it.
    assert result.weights.abs().max().max() <= cfg.max_weight * 1.5

    assert (result.costs >= 0).all()
    assert result.net_returns.sum() < result.gross_returns.sum()


def test_no_weight_held_in_delisted_names(data):
    ucfg = UniverseConfig(size=40, liquidity_window=60, min_dollar_volume=0.0)
    members, deciles = point_in_time_universe(data.close, data.dollar_volume, ucfg)
    rev = signals.reversal(data.returns, lookback=1)
    result = backtest.run_backtest(rev, data, members, deciles, BacktestConfig())

    for ticker, last in data.last_trade.items():
        if pd.isna(last):
            continue
        held_after = result.weights.loc[result.weights.index > last, ticker]
        assert (held_after == 0).all(), f"position survived {ticker} delisting"


def test_delisting_overlay_books_a_terminal_return(data):
    report = clean.delisting_report(data)
    stressed = survivorship.apply_delisting_returns(
        data, report, terminal_return=-1.0, exclude_takeovers_above=None
    )

    dead = report.index[report["delisted"]]
    booked = 0
    for ticker in dead:
        last = report.loc[ticker, "last_trade"]
        following = stressed.returns.loc[stressed.dates > last, ticker].dropna()
        if not following.empty:
            assert following.iloc[0] == pytest.approx(-1.0)
            assert stressed.traded.loc[following.index[0], ticker]
            booked += 1
    assert booked >= 8


def test_names_that_ran_up_into_delisting_are_treated_as_takeovers(data):
    """A name bid up before it disappears should not be booked at -100%."""
    report = clean.delisting_report(data)
    dead = report.index[report["delisted"]]

    doctored = data.returns.copy()
    target = dead[0]
    last = report.loc[target, "last_trade"]
    doctored.loc[doctored.index <= last, target] = doctored.loc[
        doctored.index <= last, target
    ].fillna(0.0)
    doctored.loc[doctored.index[-40:], target] = doctored.loc[
        doctored.index[-40:], target
    ]
    window = doctored.loc[:last, target].tail(21).index
    doctored.loc[window, target] = 0.03  # a 21-day, +80% run into the exit

    from dataclasses import replace as _replace

    bid = _replace(data, returns=doctored)
    stressed = survivorship.apply_delisting_returns(
        bid, report, terminal_return=-1.0, exclude_takeovers_above=0.20
    )

    following = stressed.returns.loc[stressed.dates > last, target].dropna()
    assert following.iloc[0] == pytest.approx(0.0)


def test_survivor_only_universe_drops_dead_names(data):
    ucfg = UniverseConfig(size=40, liquidity_window=60, min_dollar_volume=0.0)
    members, _ = point_in_time_universe(data.close, data.dollar_volume, ucfg)
    report = clean.delisting_report(data)
    biased = survivorship.survivor_only_members(members, report)

    dead = report.index[report["delisted"]]
    assert not biased[dead].any().any()
    assert biased.sum().sum() < members.sum().sum()


def test_residual_returns_strip_the_common_factor(data):
    residuals, loadings, factors = signals.residual_returns(
        data.returns, n_factors=2, window=252, refit_every=63, min_names=30
    )
    overlap = residuals.dropna(how="all")
    assert not overlap.empty

    market = data.returns.mean(axis=1)
    raw_corr = data.returns.corrwith(market).mean()
    resid_corr = residuals.corrwith(market).mean()
    assert abs(resid_corr) < abs(raw_corr)
    assert factors.dropna(how="all").shape[1] == 2


def test_metrics_are_finite_and_signed_correctly():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2015-01-01", periods=1000)
    good = pd.Series(rng.normal(0.0006, 0.006, 1000), index=dates)

    summary = metrics.performance_summary(good, gross=good, turnover=good.abs())
    assert summary["sharpe"] > 0
    assert summary["max_drawdown"] <= 0
    assert summary["ann_volatility"] > 0
    assert np.isfinite(summary["break_even_bps"])

    bench = pd.Series(rng.normal(0.0003, 0.010, 1000), index=dates)
    ab = metrics.alpha_beta(good, bench)
    assert np.isfinite(ab["alpha"]) and np.isfinite(ab["beta"])


def test_simulated_delistings_target_past_losers(data):
    """The simulated hazard must land on falling names, not at random."""
    rng = np.random.default_rng(3)
    stressed, delisted = survivorship.simulate_delistings(
        data, rng, annual_hazard=0.30, terminal_return=-0.60
    )
    assert len(delisted) > 5

    trailing = np.log1p(data.returns).rolling(252, min_periods=126).sum()
    ranks = trailing.rank(axis=1, pct=True)
    hit_ranks = [
        ranks.loc[date, ticker]
        for ticker, date in delisted.items()
        if not pd.isna(ranks.loc[date, ticker])
    ]
    assert np.nanmean(hit_ranks) < 0.5, "deaths are not concentrated in losers"

    for ticker, date in delisted.items():
        assert stressed.returns.loc[date, ticker] == pytest.approx(-0.60)
        assert not stressed.traded.loc[stressed.dates > date, ticker].any()


def test_bias_simulation_returns_a_distribution(data):
    ucfg = UniverseConfig(size=40, liquidity_window=60, min_dollar_volume=0.0)
    members, deciles = point_in_time_universe(data.close, data.dollar_volume, ucfg)
    rev = signals.reversal(data.returns, lookback=1)

    table = survivorship.bias_simulation(
        rev, data, members, deciles, BacktestConfig(), n_trials=3, annual_hazard=0.10
    )
    assert table.shape[0] == 3
    assert table["n_delisted"].min() > 0
    assert np.isfinite(table["sharpe"]).all()


def test_slower_rebalancing_cuts_turnover(data):
    """Holding the book between rebalances must reduce trading, not returns."""
    ucfg = UniverseConfig(size=40, liquidity_window=60, min_dollar_volume=0.0)
    members, deciles = point_in_time_universe(data.close, data.dollar_volume, ucfg)
    mom = signals.momentum(data.returns, lookback=126, skip=21)

    daily = backtest.run_backtest(
        mom, data, members, deciles, BacktestConfig(rebalance_freq="D")
    )
    monthly = backtest.run_backtest(
        mom, data, members, deciles, BacktestConfig(rebalance_freq="ME")
    )

    assert monthly.turnover.sum() < daily.turnover.sum() / 3
    assert monthly.costs.sum() < daily.costs.sum()
    # The book is still invested, not switched off.
    assert monthly.weights.abs().sum(axis=1).max() > 0.5


def test_scheduled_book_still_exits_delisted_names(data):
    """Forced exits must happen on any day, not only on rebalance days."""
    ucfg = UniverseConfig(size=40, liquidity_window=60, min_dollar_volume=0.0)
    members, deciles = point_in_time_universe(data.close, data.dollar_volume, ucfg)
    mom = signals.momentum(data.returns, lookback=126, skip=21)
    result = backtest.run_backtest(
        mom, data, members, deciles, BacktestConfig(rebalance_freq="ME")
    )

    for ticker, last in data.last_trade.items():
        if pd.isna(last):
            continue
        held_after = result.weights.loc[result.weights.index > last, ticker]
        assert (held_after == 0).all(), f"position survived {ticker} delisting"


def test_no_turnover_on_days_the_book_is_not_traded(data):
    """Between rebalances the book drifts, so turnover must be exactly zero.

    Guards a subtle accounting error: measuring drift with today's return
    rather than yesterday's charges trading on days nothing was traded, which
    inflates costs enough to make a slow signal look untradable.
    """
    ucfg = UniverseConfig(size=40, liquidity_window=60, min_dollar_volume=0.0)
    members, deciles = point_in_time_universe(data.close, data.dollar_volume, ucfg)
    mom = signals.momentum(data.returns, lookback=126, skip=21)

    result = backtest.run_backtest(
        mom, data, members, deciles, BacktestConfig(rebalance_freq="ME")
    )

    month_ends = set(
        result.weights.index.to_series().resample("ME").last().dropna().values
    )
    off_schedule = result.turnover[
        ~result.turnover.index.isin(month_ends)
    ]
    # Allow the first day and forced delisting exits, which are legitimate.
    assert (off_schedule.round(10) > 0).mean() < 0.05

    monthly_turnover = result.turnover.sum() / (len(result.turnover) / 21)
    assert monthly_turnover < 2.5, "book cannot trade more than its own gross"
