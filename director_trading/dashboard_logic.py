"""
dashboard_logic.py

Pure pandas data-processing functions for dashboard.py, kept separate
from any Streamlit calls so they're independently testable/importable
without a Streamlit environment.
"""

import pandas as pd


def detect_holding_periods(df, prefix="abnormal_return_"):
    """Find available holding periods (in trading days) from columns
    named like 'abnormal_return_20d'."""
    periods = []
    for col in df.columns:
        if col.startswith(prefix) and col.endswith("d"):
            try:
                periods.append(int(col[len(prefix):-1]))
            except ValueError:
                pass
    return sorted(periods)


def apply_filters(df, positions=None, cluster_choice="All", plan_choice="All",
                   event_choice="All", ownership_choice="All", min_txn_value=0,
                   min_pct_holdings=0.0, tickers=None, txn_choice="All"):
    """Apply the dashboard's sidebar filter selections to the results
    DataFrame. Any filter left at its default ("All" / None / 0) is a
    no-op, so callers can pass only the filters they care about."""
    out = df.copy()

    if txn_choice == "Buy only":
        out = out[out["transaction_code"] == "Buy"]
    elif txn_choice == "Sell only":
        out = out[out["transaction_code"] == "Sell"]

    if positions:
        out = out[out["position_bucket"].isin(positions)]

    if cluster_choice == "Only cluster buys":
        out = out[out["is_cluster_buy"] == True]  # noqa: E712
    elif cluster_choice == "Only solo buys":
        out = out[out["is_cluster_buy"] == False]  # noqa: E712

    if plan_choice == "Discretionary only":
        out = out[out["is_10b5_1_plan"] == False]  # noqa: E712
    elif plan_choice == "10b5-1 plan only":
        out = out[out["is_10b5_1_plan"] == True]  # noqa: E712
    elif plan_choice == "Unknown only":
        out = out[out["is_10b5_1_plan"].isna()]

    if event_choice == "Exclude near 8-K":
        out = out[out["near_corporate_event"] == False]  # noqa: E712
    elif event_choice == "Only near 8-K":
        out = out[out["near_corporate_event"] == True]  # noqa: E712

    if ownership_choice != "All":
        out = out[out["ownership_form"] == ownership_choice]

    if min_txn_value:
        out = out[out["transaction_value"].fillna(0) >= min_txn_value]
    if min_pct_holdings:
        out = out[out["pct_of_holdings"].fillna(0) >= min_pct_holdings]

    if tickers:
        out = out[out["ticker"].isin(tickers)]

    return out


def add_previous_signal_return(df, return_col):
    """For each event, attach the return from the SAME director+ticker's
    immediately prior event -- this is what lets the 'latest filings'
    view show 'last time this insider bought, here's what happened'.

    Requires df sorted by date_filed internally (handled here); returns
    a new DataFrame with prev_return / prev_date_filed columns added.
    Rows with no prior event for that director+ticker get NaN (correctly
    -- there's no history to show yet)."""
    d = df.sort_values("date_filed").copy()
    d["prev_return"] = d.groupby(["director", "ticker"])[return_col].shift(1)
    d["prev_date_filed"] = d.groupby(["director", "ticker"])["date_filed"].shift(1)
    return d


def filter_by_director_quality(df, leaderboard, min_mean_return=None, min_hit_rate=None):
    """Restrict `df` to rows whose director appears in `leaderboard` (i.e.
    already meets leaderboard's own min_trades requirement) AND whose
    historical mean_return / hit_rate clear the given thresholds. Also
    attaches that director's mean_return / hit_rate / n_trades onto each
    returned row, so callers can display "why" a filing qualified.

    min_mean_return / min_hit_rate are on the same 0-1 scale as the
    leaderboard's mean_return / hit_rate columns; leave as None to skip
    that particular threshold."""
    qualifying = leaderboard.copy()
    if min_mean_return is not None:
        qualifying = qualifying[qualifying["mean_return"] >= min_mean_return]
    if min_hit_rate is not None:
        qualifying = qualifying[qualifying["hit_rate"] >= min_hit_rate]

    if qualifying.empty:
        return df.iloc[0:0].copy()

    stats_map = qualifying.set_index("director")[["mean_return", "hit_rate", "n_trades"]]
    out = df[df["director"].isin(stats_map.index)].copy()
    out = out.join(stats_map, on="director", rsuffix="_director")
    return out


def compute_director_leaderboard(df, return_col, min_trades=1):
    """Aggregate mean/median return and hit rate per director, requiring
    at least `min_trades` observed trades so a single lucky trade can't
    dominate the leaderboard. Sorted descending by mean return."""
    usable = df.dropna(subset=[return_col])
    if usable.empty:
        return pd.DataFrame(
            columns=["director", "tickers", "n_trades", "mean_return", "median_return", "hit_rate"]
        )

    stats = (
        usable.groupby("director")
        .agg(
            n_trades=(return_col, "count"),
            mean_return=(return_col, "mean"),
            median_return=(return_col, "median"),
            hit_rate=(return_col, lambda x: (x > 0).mean()),
            tickers=("ticker", lambda x: ", ".join(sorted(set(x)))),
        )
        .reset_index()
    )
    stats = stats[stats["n_trades"] >= min_trades]
    return stats.sort_values("mean_return", ascending=False).reset_index(drop=True)