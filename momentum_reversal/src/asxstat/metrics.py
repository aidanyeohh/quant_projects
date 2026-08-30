"""Performance evaluation: the metrics the brief asks for, and the context.

Returns, volatility, Sharpe, maximum drawdown and alpha/beta are the required
set. Three more are reported alongside because a long-short book can look
excellent on the required five and still be untradable: turnover-adjusted
break-even cost, the ratio of net to gross return, and the worst calendar year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR


def annualised_return(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometric annualised return."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    total = float(np.expm1(np.log1p(clean).sum()))
    years = len(clean) / periods
    return (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else np.nan


def annualised_volatility(
    returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR
) -> float:
    return float(returns.dropna().std() * np.sqrt(periods))


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe.

    A dollar-neutral book is self-financing to a first approximation, so the
    default risk-free rate is zero; pass the RBA cash rate if evaluating a
    long-only or net-exposed variant.
    """
    excess = returns.dropna() - risk_free / periods
    if excess.empty or excess.std() == 0:
        return np.nan
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR
) -> float:
    clean = returns.dropna()
    downside = clean[clean < 0].std()
    if not downside:
        return np.nan
    return float(clean.mean() / downside * np.sqrt(periods))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown from running peak of the compounded equity curve."""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest run of days spent below a previous peak."""
    underwater = drawdown_series(returns) < 0
    if not underwater.any():
        return 0
    groups = (~underwater).cumsum()
    return int(underwater.groupby(groups).sum().max())


def alpha_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """OLS of strategy returns on benchmark returns, with Newey-West errors.

    Daily long-short returns are autocorrelated and heteroskedastic, so plain
    OLS standard errors overstate the significance of alpha. The lag length is
    the usual rule of thumb for daily data over a multi-year sample.
    """
    import statsmodels.api as sm

    joined = pd.concat([returns.rename("y"), benchmark.rename("x")], axis=1).dropna()
    if joined.shape[0] < 60:
        return {"alpha": np.nan, "beta": np.nan, "alpha_t": np.nan, "r_squared": np.nan}

    design = sm.add_constant(joined["x"])
    model = sm.OLS(joined["y"], design).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(joined.shape[0] ** 0.25)}
    )
    return {
        "alpha": float(model.params["const"] * periods),
        "beta": float(model.params["x"]),
        "alpha_t": float(model.tvalues["const"]),
        "r_squared": float(model.rsquared),
    }


def break_even_cost_bps(gross: pd.Series, turnover: pd.Series) -> float:
    """Round-trip cost in bps that would take the gross book to zero return.

    The single most useful number for judging whether a high-turnover reversal
    signal survives contact with the ASX spread.
    """
    total_turnover = turnover.sum()
    if total_turnover <= 0:
        return np.nan
    return float(gross.sum() / total_turnover * 10_000.0)


def performance_summary(
    net: pd.Series,
    gross: pd.Series | None = None,
    turnover: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    name: str = "strategy",
) -> pd.Series:
    """The full evaluation block for one return stream."""
    stats: dict[str, float] = {
        "ann_return": annualised_return(net),
        "ann_volatility": annualised_volatility(net),
        "sharpe": sharpe_ratio(net),
        "sortino": sortino_ratio(net),
        "max_drawdown": max_drawdown(net),
        "max_dd_days": max_drawdown_duration(net),
        "hit_rate": float((net.dropna() > 0).mean()),
        "skew": float(net.dropna().skew()),
        "n_days": int(net.dropna().shape[0]),
    }

    if gross is not None:
        stats["ann_return_gross"] = annualised_return(gross)
        stats["net_to_gross"] = (
            stats["ann_return"] / stats["ann_return_gross"]
            if stats["ann_return_gross"]
            else np.nan
        )
    if turnover is not None:
        stats["ann_turnover"] = float(turnover.mean() * TRADING_DAYS_PER_YEAR)
        if gross is not None:
            stats["break_even_bps"] = break_even_cost_bps(gross, turnover)
    if benchmark is not None:
        stats.update(alpha_beta(net, benchmark))

    return pd.Series(stats, name=name)


def yearly_table(net: pd.Series) -> pd.DataFrame:
    """Per calendar year return, volatility, Sharpe and drawdown."""
    by_year = net.groupby(net.index.year)
    return pd.DataFrame(
        {
            "return": by_year.apply(annualised_return),
            "volatility": by_year.apply(annualised_volatility),
            "sharpe": by_year.apply(sharpe_ratio),
            "max_drawdown": by_year.apply(max_drawdown),
        }
    )


def compare(summaries: list[pd.Series]) -> pd.DataFrame:
    """Stack several :func:`performance_summary` outputs side by side."""
    return pd.concat(summaries, axis=1)
