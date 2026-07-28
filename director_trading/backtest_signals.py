"""
backtest_signals.py

Event-study backtest of forward returns following NASDAQ insider Form 4
transactions in insider_purchases.pkl (produced by backfill_form4_history.py
/ daily_update_form4.py).

For each transaction and each holding period (default: 5, 10, 20, 60, 120
trading days), computes:
  - entry_date / entry_price   : next trading day's OPEN after the SEC
                                  filing date (date_filed) -- the earliest
                                  point the trade was public knowledge and
                                  a real order could plausibly be placed.
  - exit_date / exit_price     : close price `holding_days` trading days later
  - raw_return                 : (exit_price / entry_price) - 1
  - benchmark_return            : same-window return of a benchmark index
  - abnormal_return             : raw_return - benchmark_return

Results are saved at the individual-transaction level (backtest_results.pkl)
AND summarized by holding period x signal-quality slice (backtest_summary.pkl):
  - overall
  - is_cluster_buy (True/False)
  - is_10b5_1_plan (True/False/Unknown)
  - position_bucket (CEO/CFO/Chair/President/Director/10% Owner/Other/Unknown)
  - pct_of_holdings_bucket (<5%/5-20%/20-50%/50%+/Unknown)
  - ownership_form (Direct/Indirect)

------------------------------------------------------------------------
IMPORTANT STATISTICAL CAVEATS -- READ BEFORE TRUSTING ANY NUMBER HERE
------------------------------------------------------------------------
1. Entry timing is an approximation. We use next-day OPEN after the
   filing date. Real execution would face slippage, especially in
   small/illiquid names, that this doesn't model.
2. "Significance" here (t-stat = mean / standard error) is a rough
   diagnostic, NOT rigorous. Overlapping holding-period windows across
   transactions (especially the same ticker, or correlated market-wide
   periods) violate the independence assumption behind a plain t-test,
   which inflates apparent significance. Proper treatment needs clustered
   standard errors (by ticker and/or by calendar time) -- out of scope
   here, flagged so you don't over-trust a low p-value.
3. market_cap in the source pkl is CURRENT market cap applied to
   historical rows -- fine for "how big is this company today" but not
   for a company-size control at the time of the historical trade.
4. This is a historical backtest of a public, well-studied signal.
   Realistic expectations: use this to compare which slice of signals
   (cluster vs solo, discretionary vs 10b5-1, conviction size, etc.)
   showed the most consistent edge historically -- not as a guarantee of
   future performance.
------------------------------------------------------------------------
"""

import time
import argparse
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd

import form4_common as f4

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Install dependencies first: pip install yfinance pandas numpy")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------


INPUT_PKL = Path(f"{f4.data_output}/ceo_purchases.pkl")
RESULTS_PKL = Path(f"{f4.data_output}/backtest_results.pkl")
SUMMARY_PKL = Path(f"{f4.data_output}/backtest_summary.pkl")
PRICE_CACHE_DIR = Path(f"{f4.data_output}/price_cache")
PRICE_CACHE_DIR.mkdir(exist_ok=True)

HOLDING_PERIODS_TRADING_DAYS = [5, 10, 20, 60, 120]
BENCHMARK_TICKER = "^IXIC"  # Nasdaq Composite -- broad match for "NASDAQ-listed" universe

MAX_ENTRY_LAG_DAYS = 5  # if no trading day found within this many calendar days, skip the row


# ---------------------------------------------------------------------------
# PRICE DATA (cached to disk so re-runs don't re-hit yfinance every time)
# ---------------------------------------------------------------------------
 
def get_price_history(ticker, start, end):
    """Fetch daily OHLC history for `ticker` between start/end (date-like),
    cached to a local pickle so repeated backtest runs don't re-download."""
    cleaned_ticker = re.sub(r'[^a-zA-Z0-9 ]', '', ticker)
    cache_path = PRICE_CACHE_DIR / f"{cleaned_ticker.replace('/', '_')}.pkl"
    if cache_path.exists():
        df = pd.read_pickle(cache_path)
        if not df.empty and df.index.min() <= pd.Timestamp(start) and df.index.max() >= pd.Timestamp(end):
            return df
 
    try:
        df = yf.Ticker(ticker).history(
            start=pd.Timestamp(start) - pd.Timedelta(days=10),
            end=pd.Timestamp(end) + pd.Timedelta(days=10),
            auto_adjust=True,
        )
    except Exception as e:
        log.warning(f"  failed to download price history for {ticker}: {e}")
        return pd.DataFrame()
 
    if df.empty:
        return df
 
    df.index = df.index.tz_localize(None)
    df.to_pickle(cache_path)
    return df
 
 
def _next_trading_day_on_or_after(price_df, target_date):
    """Return the first index (trading date) in price_df on or after
    target_date, or None if none exists within MAX_ENTRY_LAG_DAYS."""
    candidates = price_df.index[price_df.index >= pd.Timestamp(target_date)]
    if candidates.empty:
        return None
    first = candidates[0]
    if (first - pd.Timestamp(target_date)).days > MAX_ENTRY_LAG_DAYS:
        return None
    return first
 
 
def compute_event_returns(price_df, filing_date, holding_periods):
    """Given a ticker's price history and a filing date, compute entry
    price (next trading day's open) and exit price/return for each
    holding period (in trading days from entry). Returns a dict."""
    if price_df.empty:
        return None
 
    entry_date = _next_trading_day_on_or_after(price_df, pd.Timestamp(filing_date) + pd.Timedelta(days=1))
    if entry_date is None:
        return None
 
    entry_idx = price_df.index.get_loc(entry_date)
    entry_price = price_df["Open"].iloc[entry_idx]
    if pd.isna(entry_price) or entry_price <= 0:
        return None
 
    out = {"entry_date": entry_date, "entry_price": entry_price}
    for h in holding_periods:
        exit_idx = entry_idx + h
        if exit_idx >= len(price_df):
            out[f"exit_date_{h}d"] = None
            out[f"exit_price_{h}d"] = None
            out[f"raw_return_{h}d"] = None
            continue
        exit_date = price_df.index[exit_idx]
        exit_price = price_df["Close"].iloc[exit_idx]
        out[f"exit_date_{h}d"] = exit_date
        out[f"exit_price_{h}d"] = exit_price
        out[f"raw_return_{h}d"] = (exit_price / entry_price) - 1 if pd.notna(exit_price) else None
    return out
 
 
# ---------------------------------------------------------------------------
# SIGNAL-QUALITY BUCKETING (for slicing the summary)
# ---------------------------------------------------------------------------
 
def bucket_position(position):
    if not isinstance(position, str) or not position.strip():
        return "Unknown"
    p = position.upper()
    if "CHIEF EXECUTIVE" in p or p == "CEO":
        return "CEO"
    if "CHIEF FINANCIAL" in p or p == "CFO":
        return "CFO"
    if "CHAIR" in p:
        return "Chair"
    if "PRESIDENT" in p:
        return "President"
    if p == "DIRECTOR":
        return "Director"
    if "10% OWNER" in p:
        return "10% Owner"
    return "Other Officer"
 
 
def bucket_pct_of_holdings(pct):
    if pd.isna(pct):
        return "Unknown"
    if pct < 0.05:
        return "<5%"
    if pct < 0.20:
        return "5-20%"
    if pct < 0.50:
        return "20-50%"
    return "50%+"
 
 
def bucket_10b5_1(val):
    if pd.isna(val):
        return "Unknown"
    return "10b5-1 Plan" if val else "Discretionary"
 
 
# ---------------------------------------------------------------------------
# MAIN BACKTEST
# ---------------------------------------------------------------------------
 
def run_backtest(input_path=INPUT_PKL, holding_periods=HOLDING_PERIODS_TRADING_DAYS,
                  benchmark_ticker=BENCHMARK_TICKER, only_buys=True, event_window_days=3):
    if not Path(input_path).exists():
        raise SystemExit(f"{input_path} not found -- run the backfill/daily scripts first.")
 
    df = pd.read_pickle(input_path)
    df = df[~df['market_cap'].isna()]
    if only_buys:
        df = df[df["transaction_code"] == "Buy"].copy()
    df = df.dropna(subset=["date_filed", "ticker"])
    log.info(f"Backtesting {len(df)} transactions across {df['ticker'].nunique()} tickers...")
 
    start = df["date_filed"].min()
    end = pd.Timestamp.today()
 
    log.info(f"Fetching benchmark ({benchmark_ticker}) price history...")
    benchmark_df = get_price_history(benchmark_ticker, start, end)
 
    # near_corporate_event: was there an 8-K (earnings, M&A, guidance,
    # etc) filed within +/-event_window_days of this Form 4? Lets you
    # separate "insider bought right after a strong earnings beat" from
    # a genuinely isolated insider-buy signal -- see signal_regression.py
    # for the fuller regression-based treatment of this same confound.
    nasdaq_ciks = f4.get_nasdaq_ciks()
    ticker_to_cik = {v["ticker"]: k for k, v in nasdaq_ciks.items()}
    log.info(f"Fetching 8-K filing dates ({start.date()} to {end.date()})...")
    cik_to_8k_dates = f4.fetch_8k_dates(nasdaq_ciks, start, end)
 
    results = []
    for i, (ticker, group) in enumerate(df.groupby("ticker"), 1):
        if i % 100 == 0:
            log.info(f"  ...{i}/{df['ticker'].nunique()} tickers processed")
 
        price_df = get_price_history(ticker, group["date_filed"].min(), end)
        time.sleep(0.05)  # light courtesy delay for yfinance
        if price_df.empty:
            continue
 
        cik = ticker_to_cik.get(ticker)
 
        for _, row in group.iterrows():
            event = compute_event_returns(price_df, row["date_filed"], holding_periods)
            if event is None:
                continue
 
            benchmark_event = compute_event_returns(benchmark_df, row["date_filed"], holding_periods) \
                if not benchmark_df.empty else None
 
            result = {
                "accession": row["accession"],
                "ticker": ticker,
                "company_name": row.get("company_name"),
                "market_cap": row.get("market_cap"),
                "director": row["director"],
                "position": row["position"],
                "position_bucket": bucket_position(row["position"]),
                "transaction_code": row["transaction_code"],
                "shares": row.get("shares"),
                "price_per_share": row.get("price_per_share"),
                "date_filed": row["date_filed"],
                "transaction_value": row["transaction_value"],
                "pct_of_holdings": row["pct_of_holdings"],
                "pct_of_holdings_bucket": bucket_pct_of_holdings(row["pct_of_holdings"]),
                "ownership_form": row["ownership_form"],
                "is_cluster_buy": row["is_cluster_buy"],
                "is_10b5_1_plan": row["is_10b5_1_plan"],
                "is_10b5_1_bucket": bucket_10b5_1(row["is_10b5_1_plan"]),
                "near_corporate_event": f4.compute_near_corporate_event(
                    cik, row["date_filed"], cik_to_8k_dates, window_days=event_window_days
                ),
                "entry_date": event["entry_date"],
                "entry_price": event["entry_price"],
            }
            for h in holding_periods:
                raw_ret = event.get(f"raw_return_{h}d")
                bench_ret = benchmark_event.get(f"raw_return_{h}d") if benchmark_event else None
                result[f"raw_return_{h}d"] = raw_ret
                result[f"benchmark_return_{h}d"] = bench_ret
                result[f"abnormal_return_{h}d"] = (
                    raw_ret - bench_ret if raw_ret is not None and bench_ret is not None else None
                )
            results.append(result)
 
    results_df = pd.DataFrame(results)
    if results_df.empty:
        log.warning("No events could be matched to price data -- nothing to summarize.")
        return results_df, pd.DataFrame()
 
    results_df.to_pickle(RESULTS_PKL)
    log.info(f"Saved {len(results_df)} event-level results to {RESULTS_PKL}")
 
    summary_df = summarize(results_df, holding_periods)
    summary_df.to_pickle(SUMMARY_PKL)
    log.info(f"Saved summary to {SUMMARY_PKL}")
 
    return results_df, summary_df
 
 
def summarize(results_df, holding_periods):
    """Build a long-format summary: one row per (slice_type, slice_value,
    holding_period) with mean/median abnormal return, hit rate, count, and
    a rough t-stat."""
    slice_cols = {
        "overall": None,
        "is_cluster_buy": "is_cluster_buy",
        "is_10b5_1_bucket": "is_10b5_1_bucket",
        "position_bucket": "position_bucket",
        "pct_of_holdings_bucket": "pct_of_holdings_bucket",
        "ownership_form": "ownership_form",
        "near_corporate_event": "near_corporate_event",
    }
 
    rows = []
    for slice_name, col in slice_cols.items():
        groups = [("all", results_df)] if col is None else list(results_df.groupby(col))
        for slice_value, group in groups:
            for h in holding_periods:
                ret_col = f"abnormal_return_{h}d"
                vals = group[ret_col].dropna()
                if len(vals) == 0:
                    continue
                mean = vals.mean()
                se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
                rows.append(
                    {
                        "slice_type": slice_name,
                        "slice_value": slice_value,
                        "holding_days": h,
                        "n": len(vals),
                        "mean_abnormal_return": mean,
                        "median_abnormal_return": vals.median(),
                        "hit_rate": (vals > 0).mean(),
                        "std_error": se,
                        "rough_t_stat": (mean / se) if se and se > 0 else np.nan,
                    }
                )
    return pd.DataFrame(rows)
 
 
def main():
    parser = argparse.ArgumentParser(description="Backtest forward returns after insider Form 4 signals.")
    parser.add_argument("--input", type=str, default=str(INPUT_PKL))
    parser.add_argument("--benchmark", type=str, default=BENCHMARK_TICKER)
    parser.add_argument("--include-sells", action="store_true",
                         help="Also backtest Sell transactions (default: Buys only)")
    parser.add_argument("--holding-periods", type=str, default=None,
                         help="Comma-separated trading-day holding periods, e.g. 5,10,20,60")
    parser.add_argument("--event-window", type=int, default=3,
                         help="Calendar days on either side counting as 'near' an 8-K")
    args = parser.parse_args()
 
    holding_periods = HOLDING_PERIODS_TRADING_DAYS
    if args.holding_periods:
        holding_periods = [int(x) for x in args.holding_periods.split(",")]
 
    results_df, summary_df = run_backtest(
        input_path=Path(args.input),
        holding_periods=holding_periods,
        benchmark_ticker=args.benchmark,
        only_buys=not args.include_sells,
        event_window_days=args.event_window,
    )
 
    if not summary_df.empty:
        pd.set_option("display.width", 160)
        print(summary_df.sort_values(["slice_type", "holding_days"]).to_string(index=False))
 
 
if __name__ == "__main__":
    main()