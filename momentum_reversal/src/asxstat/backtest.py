"""Unconstrained dollar-neutral backtest, in the style set out in the brief.

"Unconstrained" here means the book is built straight from the signal: weights
are proportional to the cross-sectional score, long and short legs are equal in
dollars, and no risk model, sector cap or optimiser sits in between. The only
constraints applied are the ones without which the result would be fiction -
a per-name weight cap, a one-day lag between observing a signal and trading it,
and an exit forced on any name that stops trading.

Costs are charged on turnover, per side, and scale with the liquidity decile
of each name, because an ASX 300 book that ignores the spread on its smallest
constituents will report a Sharpe it cannot capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .clean import Dataset
from .config import BacktestConfig
from .signals import cross_sectional_zscore


@dataclass
class BacktestResult:
    """Daily output of one backtest run."""

    weights: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    n_long: pd.Series
    n_short: pd.Series
    # Net dollar exposure of the book. Targets are built dollar neutral, but a
    # position in a name that is halted cannot be traded out, so a small
    # residual exposure is unavoidable and worth watching rather than hiding.
    net_exposure: pd.Series
    name: str = "strategy"

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gross": self.gross_returns,
                "net": self.net_returns,
                "cost": self.costs,
                "turnover": self.turnover,
                "n_long": self.n_long,
                "n_short": self.n_short,
                "net_exposure": self.net_exposure,
            }
        )


def _scale_leg(
    leg: pd.DataFrame, target: float, max_weight: float, iterations: int = 8
) -> pd.DataFrame:
    """Scale one non-negative leg to ``target`` dollars without breaching the cap.

    Rescaling and clipping fight each other - clipping the winners pushes the
    leg total below target, rescaling pushes the next names over the cap - so
    the pair is iterated, which is the standard water-filling solution. If the
    cap makes the target unreachable (too few names) the leg simply sits at the
    cap and the book runs below its nominal gross, which is the honest outcome.
    """
    scaled = leg
    for _ in range(iterations):
        total = scaled.sum(axis=1).replace(0.0, np.nan)
        scaled = scaled.div(total, axis=0) * target
        scaled = scaled.clip(upper=max_weight)
    return scaled.fillna(0.0)


def _build_neutral_book(
    z: pd.DataFrame, gross_leverage: float, max_weight: float
) -> pd.DataFrame:
    """Turn scores into an exactly dollar-neutral, capped book.

    The two legs are scaled independently rather than the book being scaled as
    a whole, because independent scaling is what makes neutrality exact: any
    common factor applied to a leg leaves it non-negative and leaves the other
    leg untouched. A final trim to the smaller leg guarantees the two match
    even where the weight cap bound on one side only.
    """
    demeaned = z.sub(z.mean(axis=1), axis=0)

    target = gross_leverage / 2.0
    longs = _scale_leg(demeaned.clip(lower=0.0), target, max_weight)
    shorts = _scale_leg((-demeaned).clip(lower=0.0), target, max_weight)

    smaller = pd.concat(
        [longs.sum(axis=1), shorts.sum(axis=1)], axis=1
    ).min(axis=1)

    longs = longs.div(longs.sum(axis=1).replace(0.0, np.nan), axis=0).mul(smaller, axis=0)
    shorts = shorts.div(shorts.sum(axis=1).replace(0.0, np.nan), axis=0).mul(smaller, axis=0)

    return (longs - shorts).fillna(0.0)


def _apply_trading_schedule(
    target: pd.DataFrame,
    data: Dataset,
    rebalance_freq: str,
) -> pd.DataFrame:
    """Hold the book between rebalances instead of retrading it every day.

    Trading to exact target weights daily is the default in a naive backtest
    and it is wrong for any signal slower than a day: the score moves a little
    each session, the book chases it, and a twelve-month momentum signal ends
    up turning over thirty times a year for no informational gain. Here the
    book is set to target only on scheduled days and left to drift with the
    market in between, which is how a real book behaves and what the cost
    numbers should be measured against.

    Two kinds of non-trading are distinguished, and conflating them is an easy
    way to get large fictitious costs. A name that is halted or simply did not
    trade today cannot be *traded*, but the position in it is still *held*; it
    keeps drifting and is retraded at the next opportunity. Only a name past
    its final trade is permanently gone, and that position is forced flat on
    whatever day the delisting happens, scheduled or not.
    """
    dates = target.index
    if rebalance_freq.upper() == "D":
        is_scheduled = np.ones(len(dates), dtype=bool)
    else:
        marks = dates.to_series().resample(rebalance_freq).last()
        is_scheduled = dates.isin(marks.values)

    target_values = np.nan_to_num(target.to_numpy())
    return_values = np.nan_to_num(data.returns.reindex_like(target).to_numpy())
    tradable = data.traded.reindex_like(target).fillna(False).to_numpy()

    last_trade = data.last_trade.reindex(target.columns)
    alive = (
        dates.to_numpy()[:, None] <= last_trade.to_numpy()[None, :]
    ) & last_trade.notna().to_numpy()[None, :]

    held = np.zeros(target.shape)
    current = np.zeros(target.shape[1])

    for row in range(target.shape[0]):
        if is_scheduled[row]:
            # Only names that can actually trade today are moved to target;
            # the rest keep the position they already have.
            current = np.where(tradable[row], target_values[row], current)
        current = np.where(alive[row], current, 0.0)
        held[row] = current
        current = current * (1.0 + return_values[row])

    return pd.DataFrame(held, index=dates, columns=target.columns)


def build_weights(
    scores: pd.DataFrame,
    data: Dataset,
    members: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Convert raw signal scores into the weights actually held each day.

    The lag and the tradability mask are applied to the scores, before the book
    is built, not to the finished weights. Dropping a name from a finished book
    would leave it neither neutral nor correctly sized; dropping it from the
    scores means the book is constructed from what was actually tradable.
    """
    smoothed = scores
    if cfg.signal_smoothing > 1:
        smoothed = scores.rolling(
            cfg.signal_smoothing, min_periods=1
        ).mean()

    z = cross_sectional_zscore(smoothed, mask=members, winsorise=cfg.winsorise_z)

    # A signal seen at the close of day t can only be traded from day t+1.
    z = z.shift(cfg.execution_lag_days)

    # A name that did not trade has no fresh price and so no usable score;
    # what happens to any existing position in it is decided by the trading
    # schedule below, not here.
    z = z.where(data.traded.reindex_like(z))

    target = _build_neutral_book(z, cfg.gross_leverage, cfg.max_weight)

    return _apply_trading_schedule(target, data, cfg.rebalance_freq)


def _slippage_bps(deciles: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Per-name, per-day slippage in bps, from the liquidity decile."""
    if cfg.costs.execution_style == "limit":
        return pd.DataFrame(0.0, index=deciles.index, columns=deciles.columns)
    filled = deciles.fillna(9.0).clip(0.0, 9.0)
    weight = filled / 9.0
    return (
        cfg.costs.slippage_bps_liquid * (1.0 - weight)
        + cfg.costs.slippage_bps_illiquid * weight
    )


def run_backtest(
    scores: pd.DataFrame,
    data: Dataset,
    members: pd.DataFrame,
    deciles: pd.DataFrame,
    cfg: BacktestConfig,
    name: str = "strategy",
) -> BacktestResult:
    """Run one signal through the book and return daily gross and net returns."""
    weights = build_weights(scores, data, members, cfg)
    returns = data.returns.reindex_like(weights).fillna(0.0)

    gross = (weights * returns).sum(axis=1)

    # Yesterday's book drifts with yesterday's returns before it is retraded,
    # so turnover is measured against the drifted weights, not the raw ones.
    # The drift must use yesterday's return, not today's: w[t-1] is held
    # through period t-1 and earns r[t-1], and what arrives at the start of
    # period t is w[t-1] * (1 + r[t-1]). Using r[t] here charges turnover on
    # days the book was never traded, which silently inflates costs and makes
    # every slow signal look untradable.
    drifted = (weights * (1.0 + returns)).shift(1).fillna(0.0)
    traded_delta = (weights - drifted).abs()

    bps = _slippage_bps(deciles.reindex_like(weights), cfg) + cfg.costs.commission_bps
    costs = (traded_delta * bps / 10_000.0).sum(axis=1)

    turnover = traded_delta.sum(axis=1)
    net = gross - costs

    return BacktestResult(
        weights=weights,
        gross_returns=gross,
        net_returns=net,
        costs=costs,
        turnover=turnover,
        n_long=(weights > 0).sum(axis=1),
        n_short=(weights < 0).sum(axis=1),
        net_exposure=weights.sum(axis=1),
        name=name,
    )


def information_coefficient(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    members: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 21, 63),
) -> pd.DataFrame:
    """Rank correlation of the signal with subsequent returns, by horizon.

    This is the cheapest honest test of whether a signal has any predictive
    content at all, and it decides the holding period: the horizon where the
    information coefficient peaks is the horizon the book should trade.
    """
    masked_scores = scores.where(members)
    out: dict[int, pd.Series] = {}

    for horizon in horizons:
        forward = (
            np.log1p(returns).rolling(horizon).sum().shift(-horizon).where(members)
        )
        ic = masked_scores.corrwith(forward, axis=1, method="spearman")
        out[horizon] = ic

    frame = pd.DataFrame(out)
    frame.columns = [f"ic_{h}d" for h in horizons]
    return frame


def ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    """Mean information coefficient, its t-statistic, and the hit rate."""
    stats = pd.DataFrame(
        {
            "mean_ic": ic.mean(),
            "std_ic": ic.std(),
            "n_obs": ic.notna().sum(),
        }
    )
    stats["t_stat"] = stats["mean_ic"] / (stats["std_ic"] / np.sqrt(stats["n_obs"]))
    stats["hit_rate"] = (ic > 0).sum() / ic.notna().sum()
    return stats


def quantile_returns(
    scores: pd.DataFrame,
    data: Dataset,
    members: pd.DataFrame,
    n_quantiles: int = 5,
    horizon: int = 1,
) -> pd.DataFrame:
    """Equal-weight forward returns by signal quantile.

    A monotone spread across quantiles is the evidence that a signal is real
    rather than an artefact of a handful of extreme names.
    """
    masked = scores.where(members)
    forward = np.log1p(data.returns).rolling(horizon).sum().shift(-horizon)

    ranks = masked.rank(axis=1, pct=True)
    out = {}
    for q in range(n_quantiles):
        low, high = q / n_quantiles, (q + 1) / n_quantiles
        bucket = (ranks > low) & (ranks <= high)
        out[f"q{q + 1}"] = forward.where(bucket).mean(axis=1)

    frame = pd.DataFrame(out)
    frame["long_short"] = frame[f"q{n_quantiles}"] - frame["q1"]
    return frame


def combine(
    results: dict[str, BacktestResult],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Blend several strategies' net returns into one book.

    With no weights supplied the blend is risk-parity on realised volatility,
    which is the sensible default when combining a momentum sleeve and a
    reversal sleeve whose volatilities differ by a factor of two or more.
    """
    net = pd.DataFrame({k: r.net_returns for k, r in results.items()})

    if weights is None:
        inv_vol = 1.0 / net.std()
        w = inv_vol / inv_vol.sum()
    else:
        w = pd.Series(weights).reindex(net.columns).fillna(0.0)
        w = w / w.abs().sum()

    net["combined"] = net[w.index].mul(w, axis=1).sum(axis=1)
    return net
