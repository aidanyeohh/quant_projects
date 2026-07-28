"""
construct_signal.py

Builds a rule-based composite "conviction" signal on top of
backtest_results.pkl (from backtest_signals.py) and validates it --
not just reports its full-sample mean return.

------------------------------------------------------------------------
THE RULE (fixed a priori -- see rationale below, not fit to the data)
------------------------------------------------------------------------
Each Buy event earns 1 point for each of the following it satisfies:

  1. is_cluster_buy       -- another insider bought the same stock nearby
  2. NOT a 10b5-1 plan     -- a discretionary decision, not pre-scheduled
  3. CEO or CFO            -- highest information-content roles
  4. pct_of_holdings >= 10%-- a meaningful increase to their own stake
  5. Direct ownership      -- personal purchase, not via a trust/entity
  6. NOT near a corporate event -- isolated signal, not confounded with
                                   an 8-K (earnings, M&A, etc.)

A "conviction score" of 0-6 comes out of this. You then pick a
threshold (default: fires at >=4/6) -- higher thresholds trade fewer,
higher-quality signals; lower thresholds trade more, noisier ones.

WHY THIS ORDER OF OPERATIONS MATTERS: the rule above was written before
looking at ANY performance numbers, based on backtest_signals.py's own
docstring reasoning (cluster buy, 10b5-1 status) and standard insider-
trading literature (C-suite > other officers, conviction size, direct
ownership, event-window confounds). This script only VALIDATES that
rule -- it never searches for the best-performing combination. If you
want to sanity-check any of these choices against your controlled
regression, run signal_regression.py and look at which terms are
positive AND significant; that's evidence for keeping/dropping a
condition, but changing the rule after seeing THIS script's output
defeats the point -- treat it as a one-way decision.
------------------------------------------------------------------------
WHAT "VALIDATED" MEANS HERE
------------------------------------------------------------------------
For each holding period and each candidate score threshold:
  - Pooled stats: n, mean/median abnormal return, hit rate, and a
    ticker-clustered t-stat (same clustering approach as
    signal_regression.py, for the same reason: overlapping windows on
    the same ticker are correlated, a plain t-test overstates
    significance).
  - Year-by-year mean return, so you can see whether the edge is
    stable, decaying, or concentrated in one lucky period.
  - A lift-vs-baseline comparison (score>=4 vs "all buys" vs "cluster
    buys only") so you can see the marginal value of stacking
    conditions, not just that "insider buying" works at all.
  - Liquidity/practicality: median transaction value and market cap of
    qualifying events, and events/year (capacity check).

None of this proves future performance -- see backtest_signals.py's own
caveats (entry-timing approximation, no transaction costs, current-not-
historical market cap). Treat this as a research tool, not a live
trading rule, without further out-of-sample testing on new data as it
arrives.

Usage:
    python3 construct_signal.py
    python3 construct_signal.py --holding-periods 20,60
    python3 construct_signal.py --thresholds 3,4,5,6
------------------------------------------------------------------------
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import form4_common as f4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_PKL = Path(f"{f4.data_output}/backtest_results.pkl")
SIGNAL_OUTPUT_PKL = Path(f"{f4.data_output}/composite_signal_results.pkl")

DEFAULT_HOLDING_PERIODS = [5, 10, 20, 60, 120]
DEFAULT_THRESHOLDS = [3, 4, 5, 6]
MIN_PCT_HOLDINGS = 0.10  # condition 4 cutoff
MIN_EVENTS_FOR_STATS = 10  # below this, don't bother reporting a t-stat


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def compute_conviction_score(df):
    """Return a 0-6 integer Series -- see module docstring for what each
    point requires. Missing/unknown values count AGAINST a point (e.g. an
    unknown 10b5-1 status doesn't get the "discretionary" point) rather
    than being silently dropped, so the score stays defined for every row."""
    score = pd.Series(0, index=df.index, dtype=int)
    score += df["is_cluster_buy"].fillna(False).astype(bool).astype(int)
    score += (df["is_10b5_1_plan"] != True).astype(int)  # noqa: E712 (False or NaN both count)
    score += df["position_bucket"].isin(["CEO", "CFO"]).astype(int)
    score += (df["pct_of_holdings"].fillna(0) >= MIN_PCT_HOLDINGS).astype(int)
    score += (df["ownership_form"] == "Direct").astype(int)
    score += (~df["near_corporate_event"].fillna(True)).astype(int)  # unknown treated as "near" (conservative)
    return score


CONDITION_LABELS = [
    "cluster_buy", "discretionary_not_10b5_1", "ceo_or_cfo",
    f"pct_holdings_gte_{int(MIN_PCT_HOLDINGS * 100)}pct", "direct_ownership", "not_near_8k",
]


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def clustered_mean_stats(values, tickers):
    """Mean, hit rate, and a ticker-clustered t-stat for a return series,
    via an intercept-only OLS with cluster-robust SEs (mirrors
    signal_regression.py's clustering approach applied to a single
    aggregate rather than a multivariate model)."""
    d = pd.DataFrame({"y": values, "ticker": tickers}).dropna()
    n = len(d)
    if n == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "hit_rate": np.nan, "t_stat_clustered": np.nan}
    mean = d["y"].mean()
    median = d["y"].median()
    hit_rate = (d["y"] > 0).mean()
    t_stat = np.nan
    if n >= MIN_EVENTS_FOR_STATS and d["ticker"].nunique() >= 2:
        try:
            model = smf.ols("y ~ 1", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
            t_stat = model.tvalues["Intercept"]
        except Exception as e:  # small/degenerate clusters can fail to invert -- don't crash the whole report
            log.warning(f"    clustered t-stat failed (n={n}): {e}")
    return {"n": n, "mean": mean, "median": median, "hit_rate": hit_rate, "t_stat_clustered": t_stat}


def evaluate_rule(df, mask, return_col, label):
    """Pooled stats + year-by-year breakdown for one subset of events."""
    sub = df[mask]
    stats = clustered_mean_stats(sub[return_col], sub["ticker"])
    stats["label"] = label

    by_year = (
        sub.dropna(subset=[return_col])
        .assign(year=sub["date_filed"].dt.year)
        .groupby("year")[return_col]
        .agg(n="count", mean="mean", hit_rate=lambda x: (x > 0).mean())
    )
    return stats, by_year


def report_holding_period(df, h, thresholds):
    return_col = f"abnormal_return_{h}d"
    if return_col not in df.columns:
        log.warning(f"  {return_col} not found -- skipping {h}d")
        return []

    df = df.dropna(subset=[return_col, "ticker", "date_filed"])
    rows = []

    print(f"\n{'=' * 78}\nHolding period: {h} trading days\n{'=' * 78}")

    baselines = {
        "all_buys": df["transaction_code"] == "Buy",
        "cluster_buys_only": (df["transaction_code"] == "Buy") & df["is_cluster_buy"].fillna(False),
    }
    for label, mask in baselines.items():
        stats, _ = evaluate_rule(df, mask, return_col, label)
        rows.append({**stats, "holding_days": h})
        _print_row(stats)

    score = compute_conviction_score(df)
    for t in thresholds:
        mask = (df["transaction_code"] == "Buy") & (score >= t)
        label = f"conviction_score_gte_{t}"
        stats, by_year = evaluate_rule(df, mask, return_col, label)
        events_per_year = by_year["n"].mean() if not by_year.empty else 0
        stats["events_per_year"] = events_per_year
        rows.append({**stats, "holding_days": h})
        _print_row(stats, events_per_year=events_per_year)

        if not by_year.empty and t == thresholds[len(thresholds) // 2]:  # show year detail for one mid threshold
            print(f"    year-by-year (score>={t}):")
            print(by_year.round(4).to_string())

        qualifying = df[mask]
        if not qualifying.empty:
            med_txn = qualifying["transaction_value"].median()
            med_cap = qualifying["market_cap"].median()
            print(f"    liquidity: median txn value ${med_txn:,.0f}, median market cap ${med_cap:,.0f}")

    return rows


def _print_row(stats, events_per_year=None):
    n, mean, median, hit, t = stats["n"], stats["mean"], stats["median"], stats["hit_rate"], stats["t_stat_clustered"]
    t_str = f"{t:.2f}" if pd.notna(t) else "n/a"
    extra = f", ~{events_per_year:.0f}/yr" if events_per_year is not None else ""
    print(
        f"  {stats['label']:<28} n={n:>6,}{extra:<14}  mean={mean * 100:6.2f}%  "
        f"median={median * 100:6.2f}%  hit_rate={hit * 100:5.1f}%  t(clustered)={t_str}"
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build and validate a composite insider-buy conviction signal.")
    parser.add_argument("--input", type=str, default=str(RESULTS_PKL))
    parser.add_argument("--holding-periods", type=str, default=None,
                         help="Comma-separated trading-day holding periods, e.g. 5,20,60")
    parser.add_argument("--thresholds", type=str, default=None,
                         help="Comma-separated conviction-score thresholds to test, e.g. 3,4,5,6")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"{input_path} not found -- run backtest_signals.py first.")

    df = pd.read_pickle(input_path)
    df["date_filed"] = pd.to_datetime(df["date_filed"], errors="coerce")

    holding_periods = DEFAULT_HOLDING_PERIODS
    if args.holding_periods:
        holding_periods = [int(x) for x in args.holding_periods.split(",")]
    thresholds = DEFAULT_THRESHOLDS
    if args.thresholds:
        thresholds = [int(x) for x in args.thresholds.split(",")]

    print(
        "Composite signal conditions (1 point each): "
        + ", ".join(CONDITION_LABELS)
    )

    all_rows = []
    for h in holding_periods:
        all_rows.extend(report_holding_period(df, h, thresholds))

    result_df = pd.DataFrame(all_rows)
    if not result_df.empty:
        result_df.to_pickle(SIGNAL_OUTPUT_PKL)
        log.info(f"\nSaved summary table to {SIGNAL_OUTPUT_PKL}")


if __name__ == "__main__":
    main()