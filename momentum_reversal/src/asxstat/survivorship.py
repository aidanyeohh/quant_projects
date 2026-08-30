"""Measuring and stressing survivorship bias rather than assuming it away.

Three distinct problems hide under the one label, and they need separate
treatment.

1.  Universe selection bias. Using today's S&P/ASX 300 membership means every
    stock in the sample was good enough to still be in the index today. This is
    handled upstream, in :mod:`asxstat.universe`, by rebuilding membership from
    trailing liquidity each month. :func:`survivor_only_members` builds the
    biased version deliberately so the size of the effect can be measured
    instead of argued about.

2.  Missing dead names. Yahoo serves many but not all delisted ASX tickers.
    :func:`coverage_report` reports how much of the sample is delisted names,
    which is the direct evidence on whether the recovery worked. A universe
    with almost no delistings has not avoided the bias, it has hidden it.

3.  Missing terminal returns. Even when a dead company is in the sample, its
    price series usually just stops rather than printing the final collapse or
    the takeover premium. :func:`apply_delisting_returns` books an explicit
    terminal return so the assumption is visible and can be stressed.

Why this matters most for the short leg: a momentum book is short the names
that have been falling, and those are precisely the names that delist. If their
final move is missing, the short leg's best trades are censored out of the
sample and momentum looks worse than it was. A reversal book is long them, so
the bias runs the other way. Reporting the strategy under several terminal
return assumptions is the only honest way to present either.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .clean import Dataset


def coverage_report(
    data: Dataset, delisting: pd.DataFrame, freq: str = "YE"
) -> pd.DataFrame:
    """How much of the sample, over time, is made of names that later died.

    A healthy recovered pool shows a delisted share that is high early in the
    sample and falls towards zero at the end, because recent years have not had
    time to produce delistings yet. A flat near-zero line means the pool is
    still survivor-only.
    """
    delisted_names = delisting.index[delisting["delisted"]]
    is_delisted = data.traded.columns.isin(delisted_names)

    alive = data.traded.resample(freq).sum() > 0
    total = alive.sum(axis=1)
    dead_later = alive.loc[:, is_delisted].sum(axis=1)

    return pd.DataFrame(
        {
            "n_trading": total,
            "n_eventually_delisted": dead_later,
            "share_eventually_delisted": dead_later / total.replace(0, np.nan),
        }
    )


def survivor_only_members(
    members: pd.DataFrame, delisting: pd.DataFrame
) -> pd.DataFrame:
    """The deliberately biased universe: only names still trading at the end.

    Running the same signal through this and through the point-in-time universe
    isolates the selection component of survivorship bias as a difference in
    Sharpe, which is the number worth quoting in the write-up.
    """
    survivors = delisting.index[~delisting["delisted"]]
    keep = members.columns.isin(survivors)
    biased = members.copy()
    biased.loc[:, ~keep] = False
    return biased


def apply_delisting_returns(
    data: Dataset,
    delisting: pd.DataFrame,
    terminal_return: float = -0.30,
    exclude_takeovers_above: float | None = 0.20,
    takeover_lookback: int = 21,
) -> Dataset:
    """Book an explicit final return on the day after each delisting.

    ``terminal_return`` is the assumed move between the last observed close and
    the actual end of the position. The default of -30% is a middle case: US
    delisting studies put the figure near -30% for exchange-forced delistings,
    while ASX delistings are a mix of those and takeovers, which resolve
    positive.

    Because a takeover and an administration should not get the same
    assumption, a name whose last ``takeover_lookback`` days show a cumulative
    move above ``exclude_takeovers_above`` is treated as acquired and given a
    zero terminal return instead: the bid has already been paid into the price.

    Run the backtest at -1.0, -0.3 and 0.0 and report all three. If the
    conclusion flips between them, the strategy is a statement about the
    delisting assumption, not about momentum or reversal.
    """
    returns = data.returns.copy()
    traded = data.traded.copy()

    dead = delisting.index[delisting["delisted"]]
    dates = data.dates

    for ticker in dead:
        if ticker not in returns.columns:
            continue
        last = delisting.loc[ticker, "last_trade"]
        if pd.isna(last):
            continue
        position = dates.searchsorted(last) + 1
        if position >= len(dates):
            continue
        exit_date = dates[position]

        assumed = terminal_return
        if exclude_takeovers_above is not None:
            recent = returns.loc[:last, ticker].tail(takeover_lookback)
            run_up = float(np.expm1(np.log1p(recent.dropna()).sum()))
            if run_up > exclude_takeovers_above:
                assumed = 0.0

        returns.loc[exit_date, ticker] = assumed
        traded.loc[exit_date, ticker] = True

    return replace(data, returns=returns, traded=traded)


# Delisting is not one event type. A liquid ASX name almost always leaves the
# market through a takeover, where the bid is already in the price and the
# terminal return is close to zero. An illiquid one leaves through
# administration, where it is close to a total loss. Applying a single average
# to both misstates the bias for every strategy that sorts on size or
# liquidity, which is all of them.
TERMINAL_RETURN_BY_DECILE = (
    0.0, 0.0, -0.05, -0.15, -0.30, -0.45, -0.60, -0.70, -0.80, -0.85,
)

# Hazard multiplier by the same liquidity decile, 0 being most liquid. Large
# caps do leave the index, but they rarely disappear.
HAZARD_BY_DECILE = (0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 1.9, 2.2)


def simulate_delistings(
    data: Dataset,
    rng: np.random.Generator,
    deciles: pd.DataFrame | None = None,
    annual_hazard: float = 0.05,
    terminal_return: float | None = None,
    hazard_by_trailing_return: tuple[float, ...] = (3.0, 1.6, 1.0, 0.7, 0.4),
    trailing_window: int = 252,
    burn_in: int = 252,
) -> tuple[Dataset, pd.Series]:
    """Inject synthetic delistings into a survivor-only sample.

    The free ASX data available here contains almost no companies that died,
    so the sample cannot be repaired. What it can do is answer a different and
    still useful question: if deaths had been present at realistic rates, how
    much would the result move? Running the strategy over many simulated
    histories turns "there is survivorship bias" into a number with an error
    bar.

    Two things must not be got wrong.

    *When* deaths happen. Delisting is not random - companies die after their
    price falls, which is exactly where a momentum book is short and a reversal
    book is long. ``hazard_by_trailing_return`` multiplies the base hazard by
    quintile of trailing return, worst quintile first, so simulated deaths land
    where real ones do.

    *What they cost*. Passing a single ``terminal_return`` applies the same
    number to a takeover and an administration. Leaving it as None instead uses
    :data:`TERMINAL_RETURN_BY_DECILE` and :data:`HAZARD_BY_DECILE`, so liquid
    names rarely die and settle near zero when they do, while the illiquid tail
    dies often and near-totally. That distinction is the whole reason the
    universe-size comparison means anything.

    The price panels are truncated alongside the returns, so a universe rebuilt
    from this dataset sees the death too and stops selecting the name. Without
    that the simulation quietly shrinks the tradable cross-section instead of
    replacing the dead name with the next-most-liquid survivor, and the Sharpe
    it reports measures concentration rather than survivorship.

    Returns the modified dataset and the delisting date of each affected name.
    """
    dates = data.dates
    trailing = np.log1p(data.returns).rolling(
        trailing_window, min_periods=trailing_window // 2
    ).sum()

    # Convert an annual hazard into a daily one, then let the trailing-return
    # quintile scale it.
    daily_hazard = 1.0 - (1.0 - annual_hazard) ** (1.0 / 252.0)
    quintiles = trailing.rank(axis=1, pct=True)
    multiplier = pd.DataFrame(1.0, index=dates, columns=data.tickers)
    for q, factor in enumerate(hazard_by_trailing_return):
        low, high = q / 5.0, (q + 1) / 5.0
        multiplier = multiplier.mask((quintiles > low) & (quintiles <= high), factor)

    # Liquidity scales both how often a name dies and what it is worth when it
    # does. Names outside the ranked universe fall back to the illiquid tail.
    if deciles is not None:
        decile = deciles.reindex_like(multiplier).ffill().fillna(9.0).clip(0, 9)
        liquidity_factor = decile.map(lambda d: HAZARD_BY_DECILE[int(d)])
        multiplier = multiplier * liquidity_factor
    else:
        decile = None

    hazard = (multiplier * daily_hazard).where(data.traded)
    draws = pd.DataFrame(
        rng.random(hazard.shape), index=hazard.index, columns=hazard.columns
    )
    dies = (draws < hazard).iloc[burn_in:]

    returns = data.returns.copy()
    traded = data.traded.copy()
    close = data.close.copy()
    adj_close = data.adj_close.copy()
    volume = data.volume.copy()
    dollar_volume = data.dollar_volume.copy()
    last_trade = data.last_trade.copy()

    delist_dates: dict[str, pd.Timestamp] = {}
    for ticker in data.tickers:
        hits = dies.index[dies[ticker].to_numpy()]
        if len(hits) == 0:
            continue
        date = hits[0]
        delist_dates[ticker] = date

        if terminal_return is not None:
            assumed = terminal_return
        elif decile is not None:
            assumed = TERMINAL_RETURN_BY_DECILE[int(decile.loc[date, ticker])]
        else:
            assumed = -0.60

        returns.loc[date, ticker] = assumed
        after = returns.index > date
        returns.loc[after, ticker] = np.nan
        traded.loc[after, ticker] = False
        close.loc[after, ticker] = np.nan
        adj_close.loc[after, ticker] = np.nan
        volume.loc[after, ticker] = np.nan
        dollar_volume.loc[after, ticker] = np.nan
        last_trade[ticker] = date

    return (
        replace(
            data,
            returns=returns,
            traded=traded,
            close=close,
            adj_close=adj_close,
            volume=volume,
            dollar_volume=dollar_volume,
            last_trade=last_trade,
        ),
        pd.Series(delist_dates, name="simulated_delist_date"),
    )


def bias_simulation(
    scores: pd.DataFrame,
    data: Dataset,
    members: pd.DataFrame,
    deciles: pd.DataFrame,
    cfg,
    universe_cfg=None,
    n_trials: int = 20,
    annual_hazard: float = 0.05,
    terminal_return: float | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Distribution of strategy performance across simulated delisting histories.

    Compare the returned distribution against the survivor-only backtest. The
    gap between the survivor-only Sharpe and the median simulated Sharpe is the
    estimated bias; the spread across trials is its uncertainty. If the
    survivor-only Sharpe sits outside the simulated range, the headline result
    is largely an artefact of the missing dead companies.

    Pass ``universe_cfg`` to rebuild the point-in-time universe inside every
    trial. This matters more than it looks: a real index replaces a delisted
    constituent with the next eligible name, so holding the original membership
    fixed would shrink the book's cross-section trial after trial and report a
    concentration penalty as though it were survivorship bias.
    """
    from .backtest import run_backtest
    from .metrics import annualised_return, max_drawdown, sharpe_ratio
    from .universe import point_in_time_universe

    rng = np.random.default_rng(seed)
    rows = []

    for trial in range(n_trials):
        stressed, delisted = simulate_delistings(
            data,
            rng,
            deciles=deciles,
            annual_hazard=annual_hazard,
            terminal_return=terminal_return,
        )

        if universe_cfg is not None:
            trial_members, trial_deciles = point_in_time_universe(
                stressed.close, stressed.dollar_volume, universe_cfg
            )
        else:
            trial_members, trial_deciles = members, deciles

        result = run_backtest(scores, stressed, trial_members, trial_deciles, cfg)
        rows.append(
            {
                "trial": trial,
                "n_delisted": int(len(delisted)),
                "median_names": float(trial_members.sum(axis=1).median()),
                "ann_return": annualised_return(result.net_returns),
                "sharpe": sharpe_ratio(result.net_returns),
                "max_drawdown": max_drawdown(result.net_returns),
            }
        )

    return pd.DataFrame(rows).set_index("trial")


def bias_summary(
    results: dict[str, "object"],
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Stack performance across survivorship scenarios for one signal.

    ``results`` maps a scenario label to a :class:`~asxstat.backtest.BacktestResult`.
    The spread of Sharpe across the rows is the honest error bar on the
    headline number.
    """
    from .metrics import performance_summary

    rows = [
        performance_summary(
            net=result.net_returns,
            gross=result.gross_returns,
            turnover=result.turnover,
            benchmark=benchmark,
            name=label,
        )
        for label, result in results.items()
    ]
    return pd.concat(rows, axis=1).T
