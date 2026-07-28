"""
form4_common.py

Shared logic for tracking NASDAQ insider (director/officer/10%-owner)
Form 4 BUY and SELL transactions via SEC EDGAR. Used by both:

  - backfill_form4_history.py  (one-time / resumable full-history load)
  - daily_update_form4.py      (daily incremental updater)

Output columns (see FINAL_COLUMNS):
  date, date_filed shares, price_per_share, transaction_value,
  transaction_code, ticker, company_name, director, position,
  title_of_security, ownership_form, market, market_cap, source_url,
  accession

------------------------------------------------------------------------
SEC REQUIRES a real contact email in the User-Agent or it will block you.
Set SEC_USER_AGENT below before running either script.
------------------------------------------------------------------------
"""

import time
import json
import logging
from pathlib import Path
from datetime import date
import xml.etree.ElementTree as ET
import pickle

import requests
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Install dependencies first: pip install yfinance pandas requests")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG (shared by both scripts)
# ---------------------------------------------------------------------------

# data_output = "~/Desktop/Python_data/quant_projects/director_trading/data_output"
data_output = "C:/Users/atsye/Desktop/Python_data/quant_projects/director_trading/data_output"

SEC_USER_AGENT = "yoyoman021.@gmail.com"
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT}

REQUEST_DELAY_SEC = 0.15  # ~6-7 req/sec, under SEC's 10 req/sec limit

DATA_PATH = Path(f"{data_output}/ceo_purchases.pkl")
STATE_PATH = Path(f"{data_output}/state.json")
NASDAQ_CACHE_PATH = Path(f"{data_output}/nasdaq_ciks_cache.json")
NASDAQ_CACHE_MAX_AGE_DAYS = 7  # NASDAQ's ticker/CIK list barely changes day to day
CLUSTER_WINDOW_DAYS = 7  # how many days on either side count as "clustered"
 
TRANSACTION_CODE_LABELS = {
    "P": "Buy",   # open-market or private purchase
    "S": "Sell",  # open-market or private sale
}
OWNERSHIP_FORM_LABELS = {
    "D": "Direct",
    "I": "Indirect",
}
 
FINAL_COLUMNS = [
    "date", "date_filed", "shares", "price_per_share", "transaction_value",
    "transaction_code", "ticker", "company_name", "director", "position",
    "title_of_security", "ownership_form", "market", "market_cap",
    "source_url", "accession",
    "is_10b5_1_plan", "footnotes", "shares_owned_after", "pct_of_holdings",
    "cluster_buy_count", "is_cluster_buy",
]

# ---------------------------------------------------------------------------
# NASDAQ TICKER / CIK UNIVERSE (cached -- this list changes rarely)
# ---------------------------------------------------------------------------
 
def get_nasdaq_ciks(cache_path=NASDAQ_CACHE_PATH, max_age_days=NASDAQ_CACHE_MAX_AGE_DAYS):
    """Return {cik (int): {"name": ..., "ticker": ...}} for NASDAQ companies.
    Cached to disk since this rarely changes."""
    if cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < max_age_days:
            log.info(f"Using cached NASDAQ ticker list ({cache_path}, {age_days:.1f}d old)")
            return {int(k): v for k, v in json.loads(cache_path.read_text()).items()}
 
    log.info("Refreshing NASDAQ ticker/CIK list from SEC...")
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["data"], columns=data["fields"])
    nasdaq = df[df["exchange"].str.lower() == "nasdaq"]
 
    mapping = {
        int(row["cik"]): {"name": row["name"], "ticker": row["ticker"]}
        for _, row in nasdaq.iterrows()
    }
    cache_path.write_text(json.dumps(mapping))
    return mapping
 
 
# ---------------------------------------------------------------------------
# MASTER INDEX PARSING (shared between daily and quarterly index files --
# both use the exact same CIK|Company Name|Form Type|Date Filed|Filename
# pipe-delimited format, SEC just publishes it at different granularities)
# ---------------------------------------------------------------------------
 
import re  # noqa: E402  (kept near point of use for readability)
 
 
def parse_master_index(raw_bytes, nasdaq_ciks, form_types=("4",)):
    """Parse a master.idx file's raw bytes (daily or quarterly -- same
    format) and return only filings of the given form_type(s) whose CIK
    is in nasdaq_ciks. Returns a list of dicts: {cik, company_name, date,
    accession}.
 
    form_types defaults to Form 4 only (the original use case). Other
    callers (e.g. signal_regression.py, which wants 8-K filing dates as a
    "was there other company news nearby" control) can pass a different
    set, like form_types=("8-K",)."""
    text = raw_bytes.decode("latin-1")  # SEC's .idx files are latin-1, not utf-8
    lines = text.splitlines()
 
    try:
        header_idx = next(i for i, l in enumerate(lines) if l.startswith("CIK|Company Name"))
    except StopIteration:
        log.warning("  unexpected master.idx format (no header found)")
        return []
    data_lines = lines[header_idx + 2:]  # skip header + '----' separator line
 
    filings = []
    for line in data_lines:
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik_str, company_name, form_type, date_filed, filename = parts
        if form_type not in form_types:
            continue
        cik = int(cik_str)
        if cik not in nasdaq_ciks:
            continue
 
        # filename looks like: edgar/data/1750/0000899243-24-000123.txt
        m = re.search(r"/(\d{10}-\d{2}-\d{6})\.txt$", filename)
        if not m:
            continue
        accession = m.group(1)
 
        filings.append(
            {"cik": cik, "company_name": company_name, "date": date_filed, "accession": accession}
        )
    return filings
 
 
def fetch_daily_index(day, nasdaq_ciks, form_types=("4",)):
    """Fetch SEC's daily master index for a single date. Returns None if
    no index was published for that date (weekend/holiday)."""
    quarter = (day.month - 1) // 3 + 1
    url = (
        f"https://www.sec.gov/Archives/edgar/daily-index/{day.year}/QTR{quarter}/"
        f"master.{day.strftime('%Y%m%d')}.idx"
    )
    r = requests.get(url, headers=SEC_HEADERS, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return parse_master_index(r.content, nasdaq_ciks, form_types=form_types)
 
 
def fetch_quarterly_index(year, quarter, nasdaq_ciks, form_types=("4",)):
    """Fetch SEC's quarterly full-index master file, covering ALL filings
    for that entire quarter in one download. This is the efficient way to
    do a multi-year backfill -- ~104 requests covers 2000-present, versus
    thousands of per-company or per-day requests."""
    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"
    r = requests.get(url, headers=SEC_HEADERS, timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return parse_master_index(r.content, nasdaq_ciks, form_types=form_types)
 
 
def iter_quarters(start_year, start_quarter, end_year, end_quarter):
    """Yield (year, quarter) tuples from start through end, inclusive."""
    y, q = start_year, start_quarter
    while (y, q) <= (end_year, end_quarter):
        yield y, q
        q += 1
        if q > 4:
            q = 1
            y += 1
 
 
def current_year_quarter():
    today = date.today()
    return today.year, (today.month - 1) // 3 + 1
 
 
# ---------------------------------------------------------------------------
# 8-K PROXIMITY (shared "was there other company news nearby" helper)
#
# SEC requires an 8-K within 4 business days of most material events --
# earnings releases, M&A, guidance changes, restructuring, executive
# changes, etc. "Was there an 8-K near this Form 4" is a blunt but broad,
# full-history, essentially-free proxy for "something else newsworthy
# happened right around this trade" -- reuses the exact same master-index
# infrastructure as Form 4 discovery, just filtered to form type 8-K, and
# only needs filing DATES (no per-filing document download).
# ---------------------------------------------------------------------------
 
EIGHTK_CACHE_PATH = Path("eightk_dates_cache.csv")
 
 
def fetch_8k_dates(nasdaq_ciks, start_date, end_date, cache_path=EIGHTK_CACHE_PATH):
    """Return {cik: sorted numpy array of datetime64 8-K filing dates}
    covering [start_date, end_date]. Cached to disk since rebuilding is
    ~1 request per quarter but each quarterly index file is several MB."""
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        if cached["date"].min() <= pd.Timestamp(start_date) and cached["date"].max() >= pd.Timestamp(end_date):
            log.info(f"Using cached 8-K filing dates ({cache_path})")
            return _to_cik_date_map(cached)
        log.info("8-K cache doesn't cover the requested range -- refreshing")
 
    start_year, start_quarter = start_date.year, (start_date.month - 1) // 3 + 1
    end_year, end_quarter = end_date.year, (end_date.month - 1) // 3 + 1
 
    all_rows = []
    for year, quarter in iter_quarters(start_year, start_quarter, end_year, end_quarter):
        log.info(f"  fetching 8-K index for {year} QTR{quarter}...")
        filings = fetch_quarterly_index(year, quarter, nasdaq_ciks, form_types=("8-K",))
        time.sleep(REQUEST_DELAY_SEC)
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
    df.to_csv(cache_path, index=False)
    log.info(f"Cached {len(df)} 8-K filing dates to {cache_path}")
    return _to_cik_date_map(df)
 
 
def _to_cik_date_map(df):
    out = {}
    for cik, group in df.groupby("cik"):
        out[cik] = np.sort(group["date"].to_numpy())
    return out
 
 
def compute_near_corporate_event(cik, event_date, cik_to_8k_dates, window_days=3):
    """True if `cik` filed an 8-K within +/- window_days calendar days of
    event_date. False (not NaN) if the CIK has no known 8-K history or
    event_date is missing -- absence of evidence isn't treated as a
    special case here, just "not flagged as near an event"."""
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
# PER-FILING LOOKUP + XML PARSING
# ---------------------------------------------------------------------------
 
def get_primary_doc_filename(cik, accession):
    """Look up the actual ownership-XML filename inside a specific
    accession folder via that folder's index.json. For Form 4 filings
    this folder normally contains exactly one .xml file (the ownership
    document itself), so this is unambiguous."""
    acc_nodash = accession.replace("-", "")
    base_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/"
    try:
        r = requests.get(base_url + "index.json", headers=SEC_HEADERS, timeout=30)
        r.raise_for_status()
        items = r.json().get("directory", {}).get("item", [])
    except (requests.RequestException, ValueError) as e:
        log.warning(f"  could not read index.json at {base_url}: {e}")
        return None
 
    xml_files = [i["name"] for i in items if i.get("name", "").lower().endswith(".xml")]
    if not xml_files:
        return None
    preferred = [n for n in xml_files if "primary_doc" in n.lower() or "form4" in n.lower()]
    return preferred[0] if preferred else xml_files[0]
 
 
def _parse_xml_bytes(content, source=""):
    """Parse XML bytes into an ElementTree root, handling stray BOM/
    whitespace, and logging (not swallowing) real parse failures."""
    cleaned = content.lstrip(b"\xef\xbb\xbf").strip()
    if b"<ownershipDocument" not in cleaned:
        log.warning(f"  {source} did not contain an <ownershipDocument> element")
        return None
    try:
        return ET.fromstring(cleaned)
    except ET.ParseError as e:
        log.warning(f"  XML parse error for {source}: {e}")
        return None
 
 
def _get_reporting_owner_position(owner_el):
    """Derive a human-readable position/title for a reportingOwner element,
    e.g. 'Chief Executive Officer', 'Director', '10% Owner', 'Other'."""
    rel = owner_el.find(".//reportingOwnerRelationship")
    if rel is None:
        return None
 
    title_el = rel.find("officerTitle")
    if title_el is not None and title_el.text and title_el.text.strip():
        return title_el.text.strip()
 
    def _flag(tag):
        el = rel.find(tag)
        return el is not None and el.text and el.text.strip() == "1"
 
    if _flag("isDirector"):
        return "Director"
    if _flag("isTenPercentOwner"):
        return "10% Owner"
    if _flag("isOther"):
        other_el = rel.find("otherText")
        return other_el.text.strip() if other_el is not None and other_el.text else "Other"
    return None
 
 
def _get_10b5_1_status(root):
    """Determine whether this filing's transactions were made under a
    Rule 10b5-1 plan.
 
    - Filings from 2023 onward have an explicit <aff10b5One> flag
      (document-level -- SEC's schema is "all or nothing" per filing, not
      per transaction).
    - Older filings lack that element entirely; the only signal is
      free-text footnotes mentioning "10b5-1" (a common but not
      guaranteed disclosure convention prior to the schema change).
 
    Returns (is_10b5_1: True/False/None, footnote_text: str or None).
    """
    footnote_texts = [
        (el.text or "").strip()
        for el in root.findall(".//footnotes/footnote")
        if el.text and el.text.strip()
    ]
    joined_footnotes = " | ".join(footnote_texts) if footnote_texts else None
    footnote_mentions_plan = any("10b5-1" in t or "10b5‑1" in t for t in footnote_texts)
 
    flag_el = root.find("aff10b5One")
    if flag_el is not None and flag_el.text is not None and flag_el.text.strip() != "":
        is_10b5_1 = flag_el.text.strip() == "1"
    elif footnote_mentions_plan:
        is_10b5_1 = True
    else:
        is_10b5_1 = None  # genuinely unknown for older filings with no footnote clue
 
    return is_10b5_1, joined_footnotes
 
 
def parse_form4_for_insider_transactions(cik, accession, doc):
    """Download a Form 4's ownership XML and pull out non-derivative BUY
    ('P') and SELL ('S') transactions for any insider (director, officer,
    10% owner, etc)."""
    acc_nodash = accession.replace("-", "")
    base_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/"
    xml_url = base_url + doc
 
    try:
        r = requests.get(xml_url, headers=SEC_HEADERS, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"  failed to download {xml_url}: {e}")
        return []
 
    root = _parse_xml_bytes(r.content, source=xml_url)
    if root is None:
        return []
 
    issuer = root.find(".//issuer")
    ticker_el = issuer.find("issuerTradingSymbol") if issuer is not None else None
    ticker = ticker_el.text.strip() if ticker_el is not None and ticker_el.text else None
 
    owner = root.find(".//reportingOwner")
    owner_name, position = None, None
    if owner is not None:
        name_el = owner.find(".//rptOwnerName")
        owner_name = name_el.text.strip() if name_el is not None and name_el.text else None
        position = _get_reporting_owner_position(owner)
 
    is_10b5_1, footnotes_text = _get_10b5_1_status(root)
 
    results = []
    for txn in root.findall(".//nonDerivativeTransaction"):
        code_el = txn.find(".//transactionCode")
        code = code_el.text if code_el is not None else None
        if code not in TRANSACTION_CODE_LABELS:
            continue
 
        shares_el = txn.find(".//transactionShares/value")
        shares = float(shares_el.text) if shares_el is not None and shares_el.text else None
 
        price_el = txn.find(".//transactionPricePerShare/value")
        price = None
        if price_el is not None and price_el.text:
            try:
                price = float(price_el.text)
            except ValueError:
                price = None
 
        date_el = txn.find(".//transactionDate/value")
        txn_date = date_el.text if date_el is not None else None
 
        title_el = txn.find(".//securityTitle/value")
        title_of_security = title_el.text.strip() if title_el is not None and title_el.text else None
 
        ownership_el = txn.find(".//ownershipNature/directOrIndirectOwnership/value")
        ownership_code = ownership_el.text.strip() if ownership_el is not None and ownership_el.text else None
        ownership_form = OWNERSHIP_FORM_LABELS.get(ownership_code, ownership_code)
 
        owned_after_el = txn.find(".//postTransactionAmounts/sharesOwnedFollowingTransaction/value")
        shares_owned_after = None
        if owned_after_el is not None and owned_after_el.text:
            try:
                shares_owned_after = float(owned_after_el.text)
            except ValueError:
                shares_owned_after = None
 
        # Conviction-size proxy: this transaction's shares as a % of the
        # insider's holdings immediately BEFORE it. A buy that doubles
        # someone's stake is a much stronger signal than the same share
        # count for someone who already held 50x that amount.
        pct_of_holdings = None
        if shares is not None and shares_owned_after is not None:
            label = TRANSACTION_CODE_LABELS.get(code, code)
            shares_before = shares_owned_after - shares if label == "Buy" else shares_owned_after + shares
            if shares_before and shares_before > 0:
                pct_of_holdings = shares / shares_before
            # shares_before <= 0 usually means this is the insider's
            # first-ever disclosed position -- pct_of_holdings left as
            # None rather than reporting a misleading infinite/undefined %.
 
        transaction_value = (shares * price) if (shares is not None and price) else None
 
        results.append(
            {
                # "date" and "date_notified" both come from
                # transactionDate/value (the date the insider actually
                # traded). If you want a separate "date SEC received the
                # filing" column, that's available from the index's
                # date_filed field -- ask and I can wire it through.
                "date": txn_date,
                "date_notified": txn_date,
                "shares": shares,
                "price_per_share": price,
                "transaction_value": transaction_value,
                "transaction_code": TRANSACTION_CODE_LABELS.get(code, code),
                "ticker": ticker,
                "director": owner_name,
                "position": position,
                "title_of_security": title_of_security,
                "ownership_form": ownership_form,
                "market": "NASDAQ",
                "source_url": xml_url,
                "accession": accession,
                "is_10b5_1_plan": is_10b5_1,
                "footnotes": footnotes_text,
                "shares_owned_after": shares_owned_after,
                "pct_of_holdings": pct_of_holdings,
            }
        )
    return results
 
 
# ---------------------------------------------------------------------------
# STATE MANAGEMENT (shared between backfill and daily scripts)
# ---------------------------------------------------------------------------
 
def load_state(state_path=STATE_PATH):
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"last_processed_date": None, "last_processed_quarter": None, "seen_accessions": []}
 
 
def save_state(state, state_path=STATE_PATH):
    state_path.write_text(json.dumps(state))
 
 
# ---------------------------------------------------------------------------
# MARKET CAP ENRICHMENT (yfinance)
# ---------------------------------------------------------------------------
 
def add_market_cap(df):
    """Adds a market_cap column by looking up each unique ticker via yfinance.
    NOTE: this is CURRENT market cap, applied retroactively to historical
    rows too -- fine as a "how big is this company today" reference, but
    it is not the market cap at the time of each historical transaction."""
    if df.empty:
        df["market_cap"] = []
        return df
 
    caps = {}
    for ticker in df["ticker"].dropna().unique():
        try:
            info = yf.Ticker(ticker).fast_info
            caps[ticker] = info.get("market_cap") or info.get("marketCap")
        except Exception as e:
            log.warning(f"  yfinance lookup failed for {ticker}: {e}")
            caps[ticker] = None
 
    df["market_cap"] = df["ticker"].map(caps)
    return df
 
 
# ---------------------------------------------------------------------------
# CLUSTER-BUY DETECTION (post-processing over the full dataset)
# ---------------------------------------------------------------------------
 
def add_cluster_signal(df, window_days=CLUSTER_WINDOW_DAYS):
    """For every BUY row, count how many OTHER distinct insiders also
    bought the SAME ticker within `window_days` on either side of this
    transaction's date. This is recomputed over the whole dataset (not
    just new rows) because a later filing can retroactively turn an
    earlier isolated buy into part of a cluster.
 
    Adds:
      cluster_buy_count - # of other distinct insiders buying nearby (Buy rows only)
      is_cluster_buy    - cluster_buy_count >= 1 (Buy rows only; None for Sell rows)
    """
    df = df.copy()
    df["cluster_buy_count"] = pd.NA
    df["is_cluster_buy"] = pd.NA
 
    buys = df[df["transaction_code"] == "Buy"].dropna(subset=["date", "ticker", "director"])
    window = pd.Timedelta(days=window_days)
 
    for ticker, group in buys.groupby("ticker"):
        dates = group["date"].to_numpy()
        directors = group["director"].to_numpy()
        idx = group.index.to_numpy()
        n = len(group)
        counts = [0] * n
        for i in range(n):
            in_window = pd.Series(dates - dates[i]).abs() <= window
            nearby_directors = set(directors[in_window.to_numpy()]) - {directors[i]}
            counts[i] = len(nearby_directors)
        df.loc[idx, "cluster_buy_count"] = counts
 
    buy_mask = df["transaction_code"] == "Buy"
    df.loc[buy_mask, "is_cluster_buy"] = df.loc[buy_mask, "cluster_buy_count"] >= 1
    return df
 
 
# ---------------------------------------------------------------------------
# DATA MERGE (append-safe, deduplicated) -- persisted as pickle so exact
# dtypes (datetimes, booleans, NaN/None distinctions) round-trip cleanly
# without needing parse_dates/converters on every read.
# ---------------------------------------------------------------------------
 
def merge_and_save(new_rows, data_path=DATA_PATH, compute_clusters=True):
    """Take a list of raw transaction dicts, enrich with market cap, merge
    with any existing pickle file (deduplicated), and save. Returns the
    combined DataFrame.
 
    `compute_clusters=False` skips the (whole-dataset) cluster-buy pass --
    useful for cheap intermediate checkpoints during a large backfill,
    where you'd rather run the cluster pass once at the end than on every
    250-filing checkpoint."""
    if not new_rows:
        log.info("No new transactions to save.")
        if data_path.exists():
            return pd.read_pickle(data_path)
        return pd.DataFrame()
 
    new_df = pd.DataFrame(new_rows)
    new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")
    new_df["date_notified"] = pd.to_datetime(new_df["date_notified"], errors="coerce")
    new_df["date_filed"] = pd.to_datetime(new_df["date_filed"], errors="coerce")
    new_df = new_df.dropna(subset=["shares"])
 
    log.info(f"Enriching {len(new_df)} new transaction(s) with market cap...")
    new_df = add_market_cap(new_df)
    new_df = new_df.reindex(columns=FINAL_COLUMNS)
 
    if data_path.exists():
        existing_df = pd.read_pickle(data_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["accession", "date", "shares", "price_per_share", "director", "transaction_code"]
        )
    else:
        combined = new_df
 
    if compute_clusters:
        log.info("Computing cluster-buy signal...")
        combined = add_cluster_signal(combined)
 
    combined = combined.sort_values("date", ascending=False).reset_index(drop=True)
    combined.to_pickle(data_path)
    log.info(f"Saved {len(combined)} total rows ({len(new_df)} new) to {data_path}")
    return combined