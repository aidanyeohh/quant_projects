"""
signal_regression.py

Multivariate regression to isolate the association between Form 4
insider-buy signal features and forward abnormal returns, CONTROLLING
for other explanatory factors simultaneously -- rather than excluding
"contaminated" events outright (which throws away data and can bias the
remaining sample if event timing isn't random).

For each holding period, fits:

    abnormal_return_Nd ~ is_cluster_buy + is_10b5_1_true + is_10b5_1_unknown
                        + pct_of_holdings + is_new_position
                        + log_transaction_value + C(position_bucket)
                        + is_indirect + near_corporate_event

using OLS with standard errors CLUSTERED BY TICKER -- not a naive t-test.
This addresses the "rough t-stat" caveat flagged in backtest_signals.py:
overlapping event windows for the same ticker are correlated with each
other, which inflates apparent significance under a plain t-test.
Clustering by ticker corrects for within-ticker correlation (though not
for cross-ticker/calendar-time correlation -- see caveats below).

------------------------------------------------------------------------
near_corporate_event: the confound control this script exists for
------------------------------------------------------------------------
True if the same company filed an 8-K within +/- EVENT_WINDOW_DAYS
calendar days of the Form 4's filing date. SEC requires an 8-K within 4
business days of most material events -- earnings releases, M&A,
guidance changes, restructuring, executive changes, etc -- so "was there
an 8-K nearby" is a blunt but broad, full-history (2000-present), and
essentially free proxy for "something else newsworthy happened right
around this trade" (exactly the earnings-announcement scenario you asked
about). It's built by reusing the SAME SEC master-index infrastructure
as the Form 4 pipeline, just filtered to form type 8-K -- and since we
only need filing DATES (not document content), no per-filing downloads
are needed at all; it's just local filtering of the same index files.

Caveats:
  - Blunt: an 8-K nearby could mean earnings, or could mean something
    unrelated (e.g. a routine executive appointment). It doesn't
    distinguish direction or magnitude of the news.
  - Two-way clustering (by ticker AND calendar time) would be more
    rigorous than clustering by ticker alone, but adds real complexity;
    out of scope here.
  - Regression coefficients show ASSOCIATION controlling for the other
    included variables, not proof of causation.
------------------------------------------------------------------------

Usage:
    python3 signal_regression.py
    python3 signal_regression.py --holding-periods 5,20,60
    python3 signal_regression.py --event-window 5
"""

import time
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
REGRESSION_OUTPUT_PKL = Path(f"{f4.data_output}/signal_regression_results.pkl")
EIGHTK_CACHE_PATH = Path(f"{f4.data_output}/eightk_dates_cache.pkl")
EVENT_WINDOW_DAYS = 3  # +/- this many calendar days counts as "near" an 8-K
HOLDING_PERIODS_TRADING_DAYS = [5, 10, 20, 60, 120]


# ---------------------------------------------------------------------------
# 8-K DATE DISCOVERY (the confound-proxy data)
# ---------------------------------------------------------------------------

def fetch_8k_dates(nasdaq_ciks, start_date, end_date, cache_path=EIGHTK_CACHE_PATH):
    """Return {cik: sorted numpy array of datetime64 8-K filing dates}
    covering [start_date, end_date]. Cached to disk since this covers the
    same span every run and is expensive to rebuild (~1 request/quarter,
    but each quarterly index file is several MB).

    Only filing DATES are collected -- no per-filing document download --
    since all we need is timing, not 8-K content."""
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if cached["date"].min() <= pd.Timestamp(start_date) and cached["date"].max() >= pd.Timestamp(end_date):
            log.info(f"Using cached 8-K filing dates ({cache_path})")
            return _to_cik_date_map(cached)
        log.info("8-K cache doesn't cover the requested range -- refreshing")

    start_year, start_quarter = start_date.year, (start_date.month - 1) // 3 + 1
    end_year, end_quarter = end_date.year, (end_date.month - 1) // 3 + 1

    all_rows = []
    for year, quarter in f4.iter_quarters(start_year, start_quarter, end_year, end_quarter):
        log.info(f"  fetching 8-K index for {year} QTR{quarter}...")
        filings = f4.fetch_quarterly_index(year, quarter, nasdaq_ciks, form_types=("8-K",))
        time.sleep(f4.REQUEST_DELAY_SEC)
        if not filings:
            continue
        for f in filings:
            all_rows.append({"cik": f["cik"], "date": f["date"]})

    df = pd.DataFrame(all_rows)
    if df.empty:
        log.warning("No 8-K filings found in this range.")
        return {}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates()
    df.to_pickly(cache_path)
    log.info(f"Cached {len(df)} 8-K filing dates to {cache_path}")
    return _to_cik_date_map(df)


def _to_cik_date_map(df):
    out = {}
    for cik, group in df.groupby("cik"):
        out[cik] = np.sort(group["date"].to_numpy())
    return out


def compute_near_corporate_event(cik, event_date, cik_to_8k_dates, window_days=EVENT_WINDOW_DAYS):
    """True if `cik` filed an 8-K within +/- window_days of event_date."""
    dates = cik_to_8k_dates.get(cik)
    if dates is None or len(dates) == 0 or pd.isna(event_date):
        return False
    event_date = np.datetime64(pd.Timestamp(event_date), "D")
    idx = np.searchsorted(dates, event_date)
    candidates = dates[max(0, idx - 1): idx + 2]  # the 1-2 nearest dates on either side
    if len(candidates) == 0:
        return False
    min_gap_days = np.min(np.abs((candidates - event_date).astype("timedelta64[D]").astype(int)))
    return bool(min_gap_days <= window_days)


# ---------------------------------------------------------------------------
# FEATURE PREP
# ---------------------------------------------------------------------------

def bucket_position(position):
    """Same bucketing as backtest_signals.py, duplicated here to keep this
    script runnable standalone against backtest_results.pickle."""
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


def prepare_regression_frame(results_df, cik_to_8k_dates, ticker_to_cik, window_days):
    df = results_df.copy()

    df["cik"] = df["ticker"].map(ticker_to_cik)
    df["near_corporate_event"] = df.apply(
        lambda r: compute_near_corporate_event(r["cik"], r["date_filed"], cik_to_8k_dates, window_days),
        axis=1,
    )

    # is_10b5_1_plan is True/False/NaN -- encode as two dummies so rows
    # with unknown status (common pre-2023) aren't dropped from the model.
    df["is_10b5_1_true"] = df["is_10b5_1_plan"] == True  # noqa: E712
    df["is_10b5_1_unknown"] = df["is_10b5_1_plan"].isna()

    # pct_of_holdings is NaN for "first-ever disclosed position" cases --
    # flag those explicitly rather than dropping them, and fill with 0 so
    # they don't reduce the regression sample.
    df["is_new_position"] = df["pct_of_holdings"].isna()
    df["pct_of_holdings_filled"] = df["pct_of_holdings"].fillna(0.0)

    df["is_indirect"] = df["ownership_form"] == "Indirect"
    df["position_bucket"] = df["position"].apply(bucket_position)

    df["log_transaction_value"] = np.log1p(df["transaction_value"].clip(lower=0))

    return df


# ---------------------------------------------------------------------------
# REGRESSION
# ---------------------------------------------------------------------------

FORMULA_TEMPLATE = (
    "{y} ~ is_cluster_buy + is_10b5_1_true + is_10b5_1_unknown "
    "+ pct_of_holdings_filled + is_new_position + log_transaction_value "
    "+ C(position_bucket) + is_indirect + near_corporate_event"
)


def run_regressions(df, holding_periods):
    all_coefs = []
    for h in holding_periods:
        y_col = f"abnormal_return_{h}d"
        if y_col not in df.columns:
            log.warning(f"  {y_col} not found -- skipping holding period {h}")
            continue

        model_df = df.dropna(subset=[y_col, "is_cluster_buy", "log_transaction_value"]).copy()
        if len(model_df) < 30:
            log.warning(f"  only {len(model_df)} usable rows for {h}d holding period -- skipping "
                        f"(too few for a meaningful regression)")
            continue

        formula = FORMULA_TEMPLATE.format(y=y_col)
        model = smf.ols(formula, data=model_df).fit(
            cov_type="cluster", cov_kwds={"groups": model_df["ticker"]}
        )

        log.info(f"\n{'=' * 70}\nHolding period: {h} trading days (n={len(model_df)})\n{'=' * 70}")
        log.info(f"\n{model.summary()}")

        for term, coef in model.params.items():
            all_coefs.append(
                {
                    "holding_days": h,
                    "n": len(model_df),
                    "term": term,
                    "coef": coef,
                    "std_err_clustered": model.bse[term],
                    "t_stat_clustered": model.tvalues[term],
                    "p_value_clustered": model.pvalues[term],
                    "r_squared": model.rsquared,
                }
            )
    return pd.DataFrame(all_coefs)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Regress forward abnormal returns on Form 4 signal features, "
                    "controlling for nearby-8-K confounds."
    )
    parser.add_argument("--input", type=str, default=str(RESULTS_PKL),
                         help="Path to backtest_results.pkl from backtest_signals.py")
    parser.add_argument("--holding-periods", type=str, default=None,
                         help="Comma-separated trading-day holding periods, e.g. 5,20,60")
    parser.add_argument("--event-window", type=int, default=EVENT_WINDOW_DAYS,
                         help="Calendar days on either side counting as 'near' an 8-K")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"{input_path} not found -- run backtest_signals.py first.")

    results_df = pd.read_pickle(input_path)
    date_cols = ["date_filed", "entry_date"] + [c for c in results_df.columns if c.startswith("exit_date_")]
    for col in date_cols:
        if col in results_df.columns:
            results_df[col] = pd.to_datetime(results_df[col], errors="coerce")

    holding_periods = HOLDING_PERIODS_TRADING_DAYS
    if args.holding_periods:
        holding_periods = [int(x) for x in args.holding_periods.split(",")]

    nasdaq_ciks = f4.get_nasdaq_ciks()
    ticker_to_cik = {v["ticker"]: k for k, v in nasdaq_ciks.items()}

    start_date = results_df["date_filed"].min()
    end_date = pd.Timestamp.today()
    log.info(f"Fetching 8-K filing dates from {start_date.date()} to {end_date.date()}...")
    cik_to_8k_dates = fetch_8k_dates(nasdaq_ciks, start_date, end_date)

    log.info("Preparing regression features...")
    reg_df = prepare_regression_frame(results_df, cik_to_8k_dates, ticker_to_cik, args.event_window)

    pct_near_event = reg_df["near_corporate_event"].mean() * 100
    log.info(f"{pct_near_event:.1f}% of events fall within {args.event_window} day(s) of an 8-K")

    log.info("Running regressions...")
    coef_df = run_regressions(reg_df, holding_periods)

    if coef_df.empty:
        log.warning("No regression results produced.")
        return

    coef_df.to_pkl(REGRESSION_OUTPUT_PKL)
    log.info(f"Saved coefficient table to {REGRESSION_OUTPUT_PKL}")

    pd.set_option("display.width", 160)
    focus = coef_df[coef_df["term"].isin(
        ["is_cluster_buy[T.True]", "near_corporate_event[T.True]",
         "is_10b5_1_true[T.True]", "pct_of_holdings_filled"]
    )]
    if not focus.empty:
        print("\nKey terms across holding periods:")
        print(focus[["holding_days", "term", "coef", "std_err_clustered", "p_value_clustered"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()