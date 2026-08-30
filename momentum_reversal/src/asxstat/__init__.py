"""Statistical arbitrage research on ASX-listed equities.

Momentum and reversal signals, a point-in-time universe reconstructed without
index-membership look-ahead, an unconstrained backtester and the performance
statistics required by the course brief.
"""

__all__ = [
    "config",
    "universe",
    "download",
    "clean",
    "signals",
    "backtest",
    "metrics",
    "survivorship",
]
