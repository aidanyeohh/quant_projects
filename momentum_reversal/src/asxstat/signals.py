"""Momentum and reversal signal library for ASX equities.

Every signal returns a raw score frame aligned to the returns panel. Raw scores
are converted to portfolio weights by the backtester, not here, so a signal can
be studied on its own (information coefficient, decay, turnover) before any
portfolio construction decision is made.

Sign convention: a positive score means "expected to outperform", so reversal
signals already carry their minus sign.

Where the ideas come from
-------------------------
The course brief lists five routes to momentum and four to reversal. Their ASX
analogues, and the function implementing each, are:

* time horizon                 -> ``momentum``, ``reversal`` at any lookback
* new information / activity   -> ``abnormal_turnover`` as a conditioner
* seasonality                  -> ``financial_year_flags``, ``day_of_week``
* investment themes            -> ``factor_momentum`` on statistical factors
* uninformed trading           -> ``reversal`` conditioned on low turnover
* correlation                  -> ``residual_returns`` then ``reversal``
* macro dislocation            -> ``market_state`` regime conditioners

The Australian specifics that differ from the crypto brief are the 30 June tax
year end, which concentrates loss-taking selling into May and June and produces
a July bounce, and the February/August reporting and dividend season, which is
when new information arrives in bulk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- utilities


def cross_sectional_zscore(
    raw: pd.DataFrame,
    mask: pd.DataFrame | None = None,
    winsorise: float = 3.0,
    min_names: int = 20,
) -> pd.DataFrame:
    """Standardise each row across names, optionally inside a universe mask.

    Winsorising happens after standardising so a single extreme name cannot
    dominate the book, and rows with too few names are dropped entirely rather
    than producing a noisy two-stock bet.
    """
    scores = raw.where(mask) if mask is not None else raw.copy()

    counts = scores.notna().sum(axis=1)
    scores = scores.where(counts.ge(min_names), other=np.nan)

    mean = scores.mean(axis=1)
    std = scores.std(axis=1).replace(0.0, np.nan)
    z = scores.sub(mean, axis=0).div(std, axis=0)
    return z.clip(-winsorise, winsorise)


def cross_sectional_rank(
    raw: pd.DataFrame, mask: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Rank each row into [-0.5, 0.5]. Robust alternative to the z-score."""
    scores = raw.where(mask) if mask is not None else raw
    ranked = scores.rank(axis=1, pct=True)
    return ranked - 0.5


def _log1p(returns: pd.DataFrame) -> pd.DataFrame:
    """Log returns, tolerant of the NaNs left by untraded days."""
    return np.log1p(returns)


# ---------------------------------------------------------------- momentum


def momentum(
    returns: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    min_periods: float = 0.7,
) -> pd.DataFrame:
    """Cumulative log return over ``lookback`` days, skipping the last ``skip``.

    The skip is the standard defence against short-horizon reversal
    contaminating a momentum signal: the most recent month tends to mean-revert
    even when the preceding year trends, so 12-1 momentum is the classic
    parameterisation (``lookback=252, skip=21``).
    """
    logret = _log1p(returns)
    window = lookback - skip
    if window <= 0:
        raise ValueError("lookback must exceed skip")
    total = logret.rolling(window, min_periods=int(window * min_periods)).sum()
    return total.shift(skip)


def risk_adjusted_momentum(
    returns: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    vol_window: int = 63,
) -> pd.DataFrame:
    """Momentum divided by trailing volatility.

    Scaling by risk stops the signal from simply loading on the highest-beta
    small resource names, which on the ASX is otherwise most of what raw
    momentum selects.
    """
    raw = momentum(returns, lookback=lookback, skip=skip)
    vol = returns.rolling(vol_window, min_periods=vol_window // 2).std()
    return raw / vol.replace(0.0, np.nan)


def factor_momentum(
    returns: pd.DataFrame,
    loadings: pd.DataFrame,
    factor_returns: pd.DataFrame,
    lookback: int = 126,
    skip: int = 5,
) -> pd.DataFrame:
    """Momentum of the themes a stock is exposed to, rather than of the stock.

    This is the brief's "investment themes" idea made mechanical: statistical
    factors extracted from the return panel stand in for themes (resources,
    banks, rate-sensitives), a stock inherits the trailing performance of the
    factors it loads on, and the score is the loading-weighted factor momentum.
    """
    factor_logret = np.log1p(factor_returns)
    window = lookback - skip
    factor_mom = factor_logret.rolling(
        window, min_periods=window // 2
    ).sum().shift(skip)

    scores = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    common = factor_mom.index.intersection(scores.index)
    for date in common:
        if factor_mom.loc[date].isna().all():
            continue
        scores.loc[date] = loadings.mul(factor_mom.loc[date], axis=1).sum(axis=1)
    return scores


# ---------------------------------------------------------------- reversal


def reversal(returns: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Negative cumulative return over the last ``lookback`` days.

    Short lookbacks (1 to 5 days) capture liquidity-provision reversal; the
    one-month version (21 days) is the classic cross-sectional reversal factor.
    """
    logret = _log1p(returns)
    window = logret.rolling(lookback, min_periods=max(lookback // 2, 1)).sum()
    return -window


def residual_returns(
    returns: pd.DataFrame,
    n_factors: int = 3,
    window: int = 252,
    refit_every: int = 21,
    min_names: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strip common factors out of returns via rolling principal components.

    This is the brief's "Security A minus something correlated to it" applied
    without needing a sector classification. On the ASX the first component is
    effectively the market, and the next two typically separate resources from
    financials, which is the dominant axis of Australian equity co-movement.

    Betas are estimated on the trailing ``window`` and refit only every
    ``refit_every`` days, then applied forward, so residuals at time t use no
    information from after t.

    Returns ``(residuals, loadings, factor_returns)``.
    """
    filled = returns.copy()
    residuals = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    loadings = pd.DataFrame(
        np.nan, index=returns.columns, columns=[f"f{i}" for i in range(n_factors)]
    )
    factor_returns = pd.DataFrame(
        np.nan, index=returns.index, columns=[f"f{i}" for i in range(n_factors)]
    )

    dates = returns.index
    fit_points = range(window, len(dates), refit_every)

    for fit_idx in fit_points:
        fit_slice = filled.iloc[fit_idx - window : fit_idx]
        # Only names with a full, dense history can enter the covariance
        # estimate; everything else is priced off the fitted factors afterwards.
        dense = fit_slice.columns[fit_slice.notna().mean() > 0.9]
        if len(dense) < min_names:
            continue

        block = fit_slice[dense].fillna(0.0)
        demeaned = block - block.mean()

        # Economically the components are portfolios of stocks; the SVD of the
        # demeaned return block gives them directly.
        _, _, vt = np.linalg.svd(demeaned.values, full_matrices=False)
        weights = pd.DataFrame(
            vt[:n_factors].T, index=dense, columns=factor_returns.columns
        )

        apply_slice = dates[fit_idx : fit_idx + refit_every]
        if len(apply_slice) == 0:
            continue

        forward = filled.loc[apply_slice]
        f_ret = forward[dense].fillna(0.0) @ weights
        factor_returns.loc[apply_slice] = f_ret

        # Regress every name, dense or not, on the realised factor returns over
        # the fitting window to get its betas.
        hist_f = block @ weights
        design = np.column_stack([np.ones(len(hist_f)), hist_f.values])
        target = fit_slice.fillna(0.0).values
        betas, *_ = np.linalg.lstsq(design, target, rcond=None)
        beta_frame = pd.DataFrame(
            betas[1:].T, index=fit_slice.columns, columns=factor_returns.columns
        )
        loadings.loc[beta_frame.index] = beta_frame

        predicted = f_ret.values @ beta_frame.reindex(returns.columns).fillna(0.0).T.values
        residuals.loc[apply_slice] = forward.values - predicted

    return residuals.where(returns.notna()), loadings, factor_returns


# ------------------------------------------------------------- conditioners


def abnormal_turnover(
    dollar_volume: pd.DataFrame, short: int = 5, long: int = 60
) -> pd.DataFrame:
    """Log ratio of recent turnover to its own trailing baseline.

    High values mark heightened activity, which the brief associates with new
    information and therefore with momentum. Low values mark quiet, liquidity-
    driven trading, which is where reversal is strongest.
    """
    fast = dollar_volume.rolling(short, min_periods=max(short // 2, 1)).mean()
    slow = dollar_volume.rolling(long, min_periods=long // 2).mean()
    return np.log(fast.replace(0.0, np.nan) / slow.replace(0.0, np.nan))


def condition_signal(
    signal: pd.DataFrame,
    conditioner: pd.DataFrame,
    mask: pd.DataFrame | None = None,
    direction: int = 1,
    strength: float = 1.0,
) -> pd.DataFrame:
    """Tilt a signal by a second variable, cross-sectionally standardised.

    ``direction=1`` amplifies the signal where the conditioner is high (use for
    momentum with abnormal turnover); ``direction=-1`` amplifies where it is low
    (use for reversal with turnover, the uninformed-trading case).

    The multiplier is floored at zero so conditioning can switch a bet off but
    never silently flip its sign.
    """
    cond_z = cross_sectional_zscore(conditioner, mask=mask)
    multiplier = (1.0 + strength * direction * cond_z).clip(lower=0.0)
    return signal * multiplier


def market_state(
    returns: pd.DataFrame,
    mask: pd.DataFrame | None = None,
    window: int = 21,
) -> pd.DataFrame:
    """Daily measures of how dislocated the market is.

    The brief's macro section asks whether reversal is stronger when volatility
    and dispersion are high. These three series are the regime variables used
    to test that on the ASX cross-section.

    Columns: ``realised_vol`` (equal-weight index volatility, annualised),
    ``dispersion`` (cross-sectional standard deviation of daily returns) and
    ``pairwise_corr`` (average pairwise correlation, approximated from the
    ratio of index variance to average single-name variance).
    """
    panel = returns.where(mask) if mask is not None else returns

    index_ret = panel.mean(axis=1)
    realised_vol = index_ret.rolling(window, min_periods=window // 2).std() * np.sqrt(252)
    dispersion = panel.std(axis=1)

    avg_var = panel.var(axis=1).rolling(window, min_periods=window // 2).mean()
    index_var = index_ret.rolling(window, min_periods=window // 2).var()
    n_names = panel.notna().sum(axis=1).clip(lower=2)
    # var(index) = (1/n) * avg_var + (1 - 1/n) * avg_cov, solved for correlation.
    avg_cov = (index_var - avg_var / n_names) / (1.0 - 1.0 / n_names)
    pairwise_corr = (avg_cov / avg_var.replace(0.0, np.nan)).clip(-1.0, 1.0)

    return pd.DataFrame(
        {
            "realised_vol": realised_vol,
            "dispersion": dispersion,
            "pairwise_corr": pairwise_corr,
        }
    )


# ------------------------------------------------------------- seasonality


def financial_year_flags(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Australian calendar effects that have no crypto analogue.

    ``tax_loss_window`` covers the run into the 30 June financial year end,
    when realising capital losses is worth money to Australian investors and
    beaten-up names get sold beyond fundamentals. ``tax_loss_reversal`` covers
    the July bounce that pressure tends to produce. ``reporting_season`` marks
    February and August, when the bulk of ASX results and dividends land and
    when momentum should be strongest on the brief's new-information logic.
    """
    month = pd.Series(dates.month, index=dates)
    day = pd.Series(dates.day, index=dates)

    return pd.DataFrame(
        {
            "tax_loss_window": ((month == 5) | ((month == 6) & (day <= 30))).astype(int),
            "tax_loss_reversal": (month == 7).astype(int),
            "reporting_season": month.isin([2, 8]).astype(int),
            "month_end": (day >= 25).astype(int),
        },
        index=dates,
    )


def day_of_week(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Weekday dummies. Monday is 0.

    The brief's institution-versus-retail seasonality idea maps onto the ASX as
    a weekday question rather than a weekend one, since the market is shut at
    the weekend: Monday sessions carry the weekend's accumulated retail order
    flow, and Friday sessions carry institutional position squaring.
    """
    weekday = pd.Series(dates.dayofweek, index=dates)
    return pd.get_dummies(weekday, prefix="dow").astype(int)
