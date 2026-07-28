"""
pages/1_Director_Price_Chart.py

Second page of the Streamlit dashboard (auto-registered by Streamlit's
multipage mechanism because it lives in a `pages/` folder next to
dashboard.py). Lets the user pick one of the top-10 directors (ranked by
historical average return and hit rate, same ranking as dashboard.py
Section 2), then a ticker that director has traded, and plots that
stock's price history with the director's Form 4 buy/sell timings
overlaid.

Run (from the project root, same place you'd run dashboard.py):
    streamlit run dashboard.py
Then use the page picker in the sidebar to switch to this page.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import dashboard_logic as dl
import form4_common as f4
from backtest_signals import get_price_history

RESULTS_PATH = Path(f"{f4.data_output}/backtest_results.pkl")

st.set_page_config(page_title="Director Price Chart", layout="wide")


@st.cache_data
def load_data(path):
    return pd.read_pickle(path)


@st.cache_data(show_spinner=False)
def load_price_history(ticker, start, end):
    return get_price_history(ticker, start, end)


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

if not RESULTS_PATH.exists():
    st.error(f"`{RESULTS_PATH}` not found. Run `python3 backtest_signals.py` first.")
    st.stop()

df = load_data(RESULTS_PATH)

st.title("🧑‍💼 Director Trade Timing vs. Share Price")
st.caption(
    "Pick a top-10 director (same ranking as the main dashboard's leaderboard: historical "
    "average return and hit rate), then a ticker they've traded, to see their Form 4 buy/sell "
    "timings plotted against that stock's price history."
)

holding_periods = dl.detect_holding_periods(df)
if not holding_periods:
    st.error("No `abnormal_return_*d` / `raw_return_*d` columns found in backtest_results.pkl.")
    st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR: DIRECTOR + TICKER SELECTION
# ---------------------------------------------------------------------------

st.sidebar.header("Director & ticker")

txn_choice = st.sidebar.selectbox(
    "Transaction type", ["Buy and Sell", "Buy only", "Sell only"],
    key="txn_choice",
    help="Shared with the main dashboard page -- selecting Buy only/Sell only here (or there) "
         "filters the underlying data everywhere, including which directors qualify for the "
         "top-10 list below.",
)
if txn_choice == "Buy only":
    df = df[df["transaction_code"] == "Buy"]
elif txn_choice == "Sell only":
    df = df[df["transaction_code"] == "Sell"]

return_type = st.sidebar.radio("Return type", ["Abnormal (vs. Nasdaq benchmark)", "Raw"])
return_prefix = "abnormal_return_" if return_type.startswith("Abnormal") else "raw_return_"

default_hp = 20 if 20 in holding_periods else holding_periods[len(holding_periods) // 2]
holding_period = st.sidebar.selectbox(
    "Holding period (trading days)", holding_periods, index=holding_periods.index(default_hp)
)
return_col = f"{return_prefix}{holding_period}d"

min_trades = st.sidebar.slider(
    "Min historical trades to qualify for leaderboard", 1, 20, 3,
    help="Same qualification rule as the main dashboard's leaderboard.",
)

leaderboard = dl.compute_director_leaderboard(df, return_col, min_trades=min_trades)
top10 = leaderboard.head(10)

if top10.empty:
    st.info(
        f"No directors have at least {min_trades} qualifying trades under the current "
        f"transaction-type filter ({txn_choice}). Try lowering the 'min historical trades' "
        "filter or changing the transaction type."
    )
    st.stop()

selected_director = st.sidebar.selectbox("Director (top 10)", top10["director"].tolist())

director_stats = top10[top10["director"] == selected_director].iloc[0]
st.sidebar.metric(f"Mean return ({holding_period}d)", f"{director_stats['mean_return'] * 100:.2f}%")
st.sidebar.metric("Hit rate", f"{director_stats['hit_rate'] * 100:.0f}%")
st.sidebar.metric("Historical trades", int(director_stats["n_trades"]))

director_df = df[df["director"] == selected_director]
ticker_options = sorted(director_df["ticker"].dropna().unique())

if not ticker_options:
    st.info("This director has no trades with a resolvable ticker under the current filters.")
    st.stop()

selected_ticker = st.sidebar.selectbox("Ticker", ticker_options)

trades = director_df[director_df["ticker"] == selected_ticker].sort_values("date_filed")

if trades.empty:
    st.info("No trades found for this director/ticker combination.")
    st.stop()

# ---------------------------------------------------------------------------
# PRICE HISTORY + OVERLAY
# ---------------------------------------------------------------------------

price_start = trades["date_filed"].min() - pd.Timedelta(days=60)
price_end = pd.Timestamp.today()

with st.spinner(f"Loading price history for {selected_ticker}..."):
    price_df = load_price_history(selected_ticker, price_start, price_end)

if price_df is None or price_df.empty:
    st.error(f"No price history available for {selected_ticker}.")
    st.stop()

buys = trades[trades["transaction_code"] == "Buy"]
sells = trades[trades["transaction_code"] == "Sell"]


def _hover_text(row):
    parts = [f"{row['transaction_code']} filed {pd.Timestamp(row['date_filed']).date()}"]
    if pd.notna(row.get("shares")):
        parts.append(f"{row['shares']:,.0f} sh")
    if pd.notna(row.get("price_per_share")):
        parts.append(f"@ ${row['price_per_share']:.2f}")
    ret = row.get(return_col)
    if pd.notna(ret):
        parts.append(f"{holding_period}d return: {ret * 100:.1f}%")
    return "<br>".join(parts)


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=price_df.index, y=price_df["Close"], mode="lines", name=f"{selected_ticker} close",
    line=dict(color="#4C78A8", width=1.5),
))

if not buys.empty:
    fig.add_trace(go.Scatter(
        x=buys["date_filed"], y=buys["entry_price"], mode="markers", name="Buy",
        marker=dict(symbol="triangle-up", size=13, color="#2CA02C", line=dict(width=1, color="white")),
        text=[_hover_text(r) for _, r in buys.iterrows()], hoverinfo="text",
    ))

if not sells.empty:
    fig.add_trace(go.Scatter(
        x=sells["date_filed"], y=sells["entry_price"], mode="markers", name="Sell",
        marker=dict(symbol="triangle-down", size=13, color="#D62728", line=dict(width=1, color="white")),
        text=[_hover_text(r) for _, r in sells.iterrows()], hoverinfo="text",
    ))

fig.update_layout(
    title=f"{selected_ticker} price -- {selected_director}'s Form 4 trades",
    xaxis_title="Date", yaxis_title="Price ($)", height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    hovermode="closest",
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Buy/sell markers are plotted at `entry_price` (next trading day's open after the SEC "
    "filing date), matching the entry point used in backtest_signals.py -- not the exact "
    "transaction date/price reported on the Form 4 itself."
)

# ---------------------------------------------------------------------------
# TRADE LOG
# ---------------------------------------------------------------------------

st.subheader(f"{selected_director} -- {selected_ticker} trade log")

log_cols = [
    "date_filed", "transaction_code", "shares", "price_per_share", "transaction_value",
    "pct_of_holdings", "is_cluster_buy", "is_10b5_1_plan", "near_corporate_event",
    f"raw_return_{holding_period}d", f"abnormal_return_{holding_period}d",
]
log_cols = [c for c in log_cols if c in trades.columns]

show_trades = trades[log_cols].copy()
for col in show_trades.columns:
    if col.startswith("raw_return_") or col.startswith("abnormal_return_") or col == "pct_of_holdings":
        show_trades[col] = show_trades[col].map(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "—")

st.dataframe(show_trades, use_container_width=True, hide_index=True)