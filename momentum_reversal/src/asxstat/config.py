"""Central configuration for the ASX statistical-arbitrage research project.

All tunable research parameters live here so that backtests are reproducible
from a single object rather than from scattered literals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Yahoo Finance suffix for Australian Securities Exchange listings.
ASX_SUFFIX = ".AX"

# Benchmarks. ^AXJO = S&P/ASX 200, ^AXKO = S&P/ASX 300, ^AORD = All Ordinaries.
BENCHMARK = "^AXKO"
BENCHMARK_FALLBACKS = ("^AXJO", "^AORD")

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class CostModel:
    """Round-trip execution costs, expressed in basis points of traded notional.

    The course brief quotes 7 bps commission and 13 bps slippage for crypto.
    ASX equities are cheaper on commission but pay a bid-ask spread that widens
    sharply outside the top ~100 names, so cost is a function of liquidity rank.
    Costs below are charged per side, on turnover.
    """

    commission_bps: float = 8.0
    # Slippage is interpolated between these two anchors by liquidity decile,
    # decile 0 being the most liquid names in the point-in-time universe.
    slippage_bps_liquid: float = 7.0
    slippage_bps_illiquid: float = 35.0

    # "market" crosses the spread and pays commission plus slippage.
    #
    # "limit" pays commission only, which is what the brief assumes for passive
    # execution. It is the right model to *consider* for a reversal strategy,
    # because buying what just fell and selling what just rose is liquidity
    # provision and belongs on the passive side of the book. It is also
    # optimistic, and knowingly so: it assumes every order fills. In reality
    # the orders that do not fill are the ones the market ran away from, so the
    # fills are adversely selected and the realised edge is lower than this.
    # Treat a limit-order result as an upper bound, never as the headline.
    execution_style: str = "market"

    def slippage_bps_by_decile(self, decile: float) -> float:
        """Linear interpolation of slippage across liquidity deciles 0..9."""
        if self.execution_style == "limit":
            return 0.0
        weight = decile / 9.0
        return (
            self.slippage_bps_liquid * (1.0 - weight)
            + self.slippage_bps_illiquid * weight
        )


@dataclass(frozen=True)
class UniverseConfig:
    """Rules for rebuilding index membership point-in-time.

    Index constituent history is not freely available, and using today's
    S&P/ASX 300 membership would import exactly the survivorship bias this
    project is trying to avoid. Instead the universe is rebuilt each month
    from trailing liquidity, using only information available at that date.
    """

    size: int = 300
    # Trailing window used to rank liquidity, in trading days.
    liquidity_window: int = 60
    # A name must have traded on at least this fraction of the window.
    min_data_fraction: float = 0.8
    # A name must have traded within this many days of the rebalance to count
    # as alive. Guards against ranking a name that has already been delisted.
    max_staleness_days: int = 5
    # Minimum trailing median daily turnover in AUD to be eligible at all.
    min_dollar_volume: float = 250_000.0
    # Minimum price in AUD. Sub-cent stocks have quantised returns and
    # spreads measured in whole percent, which corrupts reversal signals.
    min_price: float = 0.10
    rebalance_freq: str = "ME"  # month-end


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters for the unconstrained (dollar-neutral) backtest."""

    # Gross book size. Long leg and short leg each get half of this.
    gross_leverage: float = 1.0
    # Winsorise cross-sectional z-scores at this many standard deviations.
    winsorise_z: float = 3.0
    # Lag between signal observation and trade. 1 = signal from close of day t
    # is traded at close of day t+1, which is the honest minimum.
    execution_lag_days: int = 1
    # Cap on any single name's share of gross book.
    max_weight: float = 0.05
    # How often the book is traded back to its target weights. Between
    # rebalances positions drift with the market and are left alone. Trading a
    # slow signal every day churns the book for no informational gain, which
    # is the single largest avoidable cost in a stat-arb backtest.
    # "D" daily, "W" weekly, "ME" month end.
    rebalance_freq: str = "W"
    # Average the raw signal over this many days before ranking. Smoothing
    # removes the day-to-day noise in a score that would otherwise be traded.
    signal_smoothing: int = 1
    costs: CostModel = field(default_factory=CostModel)


@dataclass(frozen=True)
class Config:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    start_date: str = "2005-01-01"
    end_date: str | None = None  # None = today


DEFAULT = Config()
