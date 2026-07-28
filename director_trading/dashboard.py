"""
dashboard.py

Streamlit dashboard for NASDAQ insider Form 4 buy signals, built on
backtest_results.pkl (produced by backtest_signals.py, which itself
needs form4_common.py's near_corporate_event/market_cap columns).

Run with:
    streamlit run dashboard.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

import dashboard_logic as dl
import form4_common as f4

RESULTS_PATH = Path(f"{f4.data_output}/backtest_results.pkl")

st.set_page_config(page_title="Insider Form 4 Signal Dashboard", layout="wide")


@st.cache_data
def load_data(path):
    return pd.read_pickle(path)


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

if not RESULTS_PATH.exists():
    st.error(f"`{RESULTS_PATH}` not found. Run `python3 backtest_signals.py` first.")
    st.stop()

df = load_data(RESULTS_PATH)

holding_periods = dl.detect_holding_periods(df)
if not holding_periods:
    st.error("No `abnormal_return_*d` / `raw_return_*d` columns found in backtest_results.pkl.")
    st.stop()

st.title("📈 Insider Form 4 Signal Dashboard")
st.caption(
    "Historical backtest results, not investment advice. Past insider-buy returns are not a "
    "guarantee of future performance -- see backtest_signals.py / signal_regression.py for "
    "methodology and caveats (entry-timing assumptions, confound controls, statistical limits)."
)

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")

return_type = st.sidebar.radio(
    "Return type", ["Abnormal (vs. Nasdaq benchmark)", "Raw"],
    help="Abnormal return nets out the benchmark's move over the same window -- generally the "
         "more meaningful figure. Raw is the stock's own return, unadjusted.",
)
return_prefix = "abnormal_return_" if return_type.startswith("Abnormal") else "raw_return_"

default_hp = 20 if 20 in holding_periods else holding_periods[len(holding_periods) // 2]
holding_period = st.sidebar.selectbox(
    "Holding period (trading days)", holding_periods, index=holding_periods.index(default_hp)
)
return_col = f"{return_prefix}{holding_period}d"

position_options = sorted(df["position_bucket"].dropna().unique()) if "position_bucket" in df else []
positions = st.sidebar.multiselect("Position", position_options, default=position_options)

txn_choice = st.sidebar.selectbox(
    "Transaction type", ["Buy and Sell", "Buy only", "Sell only"],
    key="txn_choice",
    help="Note: backtest_signals.py defaults to Buy-only backtests (--include-sells enables Sell "
         "rows), so this filter only has an effect if backtest_results.pkl includes Sell rows. "
         "This selection is shared with the Director Price Chart page.",
)
txn_choice = {"Buy and Sell": "All", "Buy only": "Buy only", "Sell only": "Sell only"}[txn_choice]

cluster_choice = st.sidebar.selectbox(
    "Cluster buy", ["All", "Only cluster buys", "Only solo buys"],
    help="Cluster buy = another insider also bought the same stock within a nearby window. "
         "Generally considered a stronger signal than an isolated purchase.",
)
plan_choice = st.sidebar.selectbox(
    "10b5-1 status", ["All", "Discretionary only", "10b5-1 plan only", "Unknown only"],
    help="A 10b5-1 trade was pre-scheduled months in advance and reflects planning, not a fresh "
         "view -- generally a weaker signal than a discretionary purchase.",
)
event_choice = st.sidebar.selectbox(
    "Nearby 8-K filing", ["All", "Exclude near 8-K", "Only near 8-K"],
    help="An 8-K near the Form 4 filing date may indicate earnings, M&A, or other confounding "
         "company news around the same time as the trade.",
)
ownership_choice = st.sidebar.selectbox("Ownership form", ["All", "Direct", "Indirect"])

min_txn_value = st.sidebar.number_input(
    "Min transaction value ($)", min_value=0, value=0, step=10_000
)
min_pct_holdings = st.sidebar.slider(
    "Min % of holdings purchased", 0.0, 1.0, 0.0, step=0.01,
    help="Purchase size as a fraction of the insider's pre-trade stake -- a conviction proxy.",
)

ticker_options = sorted(df["ticker"].dropna().unique())
tickers = st.sidebar.multiselect("Ticker (optional)", ticker_options)

st.sidebar.divider()
st.sidebar.subheader("Director track record")
min_trades = st.sidebar.slider(
    "Min historical trades per director", 1, 20, 3,
    help="Requires at least this many trades before a director's stats are trusted (leaderboard "
         "and 'Latest Filings' table alike), so a single lucky trade can't dominate either.",
)
min_director_mean_return_pct = st.sidebar.slider(
    "Min director mean return (%)", -20.0, 20.0, 0.0, step=0.5,
    help="Only show 'Latest Filings' from directors whose historical average return (at the "
         "selected holding period) is at least this. Doesn't affect the leaderboard itself.",
)
min_director_hit_rate_pct = st.sidebar.slider(
    "Min director hit rate (%)", 0, 100, 50, step=5,
    help="Only show 'Latest Filings' from directors who were profitable at least this often "
         "historically. Doesn't affect the leaderboard itself.",
)
latest_window_days = st.sidebar.slider("'Latest filings' window (days)", 1, 30, 7)

# ---------------------------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------------------------

filtered = dl.apply_filters(
    df,
    positions=positions,
    cluster_choice=cluster_choice,
    plan_choice=plan_choice,
    event_choice=event_choice,
    ownership_choice=ownership_choice,
    min_txn_value=min_txn_value,
    min_pct_holdings=min_pct_holdings,
    tickers=tickers,
    txn_choice=txn_choice,
)

st.caption(f"{len(filtered):,} of {len(df):,} transactions match the current filters")

# Director leaderboard is used both to gate Section 1 (only show filings from
# directors with a strong track record) and to build Section 2's ranking, so
# it's computed once up front.
leaderboard = dl.compute_director_leaderboard(filtered, return_col, min_trades=min_trades)

# ---------------------------------------------------------------------------
# SECTION 1: LATEST FILINGS + RETURN AFTER THEIR PREVIOUS SIGNAL
# ---------------------------------------------------------------------------

st.header("1. Latest Form 4 Filings")
st.caption(
    f"Showing only filings from directors with >= {min_trades} historical trades, a mean "
    f"{holding_period}d return >= {min_director_mean_return_pct:.1f}%, and a hit rate >= "
    f"{min_director_hit_rate_pct}% -- adjust thresholds in the sidebar."
)

if filtered.empty or filtered["date_filed"].isna().all():
    st.info("No data matches the current filters.")
elif leaderboard.empty:
    st.info(
        f"No directors have at least {min_trades} qualifying trades under the current filters. "
        "Try lowering the 'min historical trades' filter."
    )
else:
    with_prev = dl.add_previous_signal_return(filtered, return_col)
    qualified = dl.filter_by_director_quality(
        with_prev, leaderboard,
        min_mean_return=min_director_mean_return_pct / 100,
        min_hit_rate=min_director_hit_rate_pct / 100,
    )

    if qualified.empty or qualified["date_filed"].isna().all():
        st.info(
            "No filings from directors meeting the mean-return / hit-rate thresholds. Try "
            "lowering them in the sidebar."
        )
    else:
        max_date = qualified["date_filed"].max()
        cutoff = max_date - pd.Timedelta(days=latest_window_days)
        latest = qualified[qualified["date_filed"] >= cutoff].sort_values("date_filed", ascending=False)

        avg_prev_return = latest["prev_return"].mean()
        n_with_history = int(latest["prev_return"].notna().sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("Filings in window", len(latest))
        col2.metric(
            f"Avg return after PREVIOUS signal ({holding_period}d)",
            f"{avg_prev_return * 100:.2f}%" if pd.notna(avg_prev_return) else "N/A",
            help="For each filing below, this is the average return you'd have earned had you bought "
                 "after that SAME director's PREVIOUS Form 4 purchase (not this new one).",
        )
        col3.metric("...with prior history", f"{n_with_history}/{len(latest)}")

        display_cols = [
            "date_filed", "ticker", "company_name", "director", "position",
            "shares", "price_per_share", "transaction_value", "market_cap",
            "is_cluster_buy", "is_10b5_1_plan", "near_corporate_event",
            "prev_date_filed", "prev_return",
            "mean_return", "hit_rate", "n_trades",
        ]
        display_cols = [c for c in display_cols if c in latest.columns]

        show_df = latest[display_cols].copy()
        if "prev_return" in show_df.columns:
            show_df["prev_return"] = show_df["prev_return"].map(
                lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "—"
            )
        if "mean_return" in show_df.columns:
            show_df = show_df.rename(columns={
                "mean_return": "director_mean_return", "hit_rate": "director_hit_rate",
                "n_trades": "director_n_trades",
            })
            show_df["director_mean_return"] = show_df["director_mean_return"].map(
                lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "—"
            )
            show_df["director_hit_rate"] = show_df["director_hit_rate"].map(
                lambda x: f"{x * 100:.0f}%" if pd.notna(x) else "—"
            )
        if "market_cap" in show_df.columns:
            show_df["market_cap"] = show_df["market_cap"].map(
                lambda x: f"${x / 1e9:.2f}B" if pd.notna(x) and x >= 1e9
                else (f"${x / 1e6:.0f}M" if pd.notna(x) else "—")
            )

        st.dataframe(show_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# SECTION 2: TOP 10 DIRECTORS BY AVERAGE RETURN
# ---------------------------------------------------------------------------

st.header("2. Top 10 Directors by Average Return")

top10 = leaderboard.head(10)

if top10.empty:
    st.info(
        f"No directors have at least {min_trades} qualifying trades under the current filters. "
        "Try lowering the 'min historical trades' filter."
    )
else:
    chart_df = top10.set_index("director")["mean_return"]
    st.bar_chart(chart_df)

    show_board = top10.copy()
    show_board["mean_return"] = show_board["mean_return"].map(lambda x: f"{x * 100:.2f}%")
    show_board["median_return"] = show_board["median_return"].map(lambda x: f"{x * 100:.2f}%")
    show_board["hit_rate"] = show_board["hit_rate"].map(lambda x: f"{x * 100:.0f}%")
    st.dataframe(
        show_board[["director", "tickers", "n_trades", "mean_return", "median_return", "hit_rate"]],
        use_container_width=True, hide_index=True,
    )

st.caption(
    "'Mean return' = average abnormal/raw return across all qualifying trades for that director "
    f"at the selected {holding_period}-day holding period. Requires >= {min_trades} trades to appear."
)