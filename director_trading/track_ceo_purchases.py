"""
track_ceo_purchases.py

Tracks CEO share PURCHASES for NASDAQ companies via SEC Form 4 filings
(EDGAR), and builds a pandas DataFrame with columns:

  date              - transaction date
  shares            - number of shares purchased
  price_per_share   - transaction price per share (USD)
  transaction_value - shares * price_per_share (USD)
  ticker            - NASDAQ ticker
  company_name      - issuer name
  director          - name of the CEO (reporting owner)
  market            - "NASDAQ"
  market_cap        - current market cap of the company (via yfinance)
  source_url        - link to the underlying Form 4 XML

------------------------------------------------------------------------
NOTES
------------------------------------------------------------------------
- Form 4 XML is structured and reliable. This script only keeps
  transactions where:
      transactionCode == "P"  (open-market/private purchase)
      reporting owner's officerTitle contains "CEO" / "CHIEF EXECUTIVE"
- Some "P" transactions (e.g. certain plan-related acquisitions) can have
  a price of 0 or be missing a price entirely -- these are kept but with
  price_per_share / transaction_value as None, so filter on that if you
  only want genuine open-market cash purchases.
- SEC requires a descriptive User-Agent with a working contact email. Set
  SEC_USER_AGENT below before running, or you will be blocked / rate-limited.
- Scanning every NASDAQ company means thousands of HTTP requests. Use
  `max_companies` to test on a small sample first. Respect rate limits
  (SEC: max ~10 req/sec, this script defaults to much slower).
------------------------------------------------------------------------
"""

import re
import time
import logging
import xml.etree.ElementTree as ET

import requests
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Install dependencies first: pip install yfinance pandas requests")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

#test

# SEC REQUIRES a real contact email in the User-Agent or it will block you.
SEC_USER_AGENT = "yoyoman021@gmail.com"
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT}

REQUEST_DELAY_SEC = 0.15   # ~6-7 req/sec, under SEC's 10 req/sec limit

# --- Tune these before running at scale ---
NASDAQ_MAX_COMPANIES = 5      # set higher / None to scan all NASDAQ (slow!)
NASDAQ_FILINGS_PER_COMPANY = 10
# ---------------------------------------------------------------------------
# NASDAQ / SEC FORM 4
# ---------------------------------------------------------------------------

def get_nasdaq_tickers(max_companies=None):
    """Pull the list of NASDAQ-listed tickers + CIK numbers from SEC's
    company_tickers_exchange.json (free, no auth)."""
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["data"], columns=data["fields"])
    nasdaq = df[df["exchange"].str.lower() == "nasdaq"].copy()
    if max_companies:
        nasdaq = nasdaq.head(max_companies)
    return nasdaq[["cik", "name", "ticker"]]


def get_recent_form4_filings(cik, lookback_n=40):
    """List recent Form 4 filing metadata for a given CIK via EDGAR's
    submissions API."""
    cik_str = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    if r.status_code != 200:
        return []
    data = r.json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    docs = recent.get("primaryDocument", [])

    filings = []
    for form, acc, date, doc in zip(forms, accessions, dates, docs):
        if form == "4":
            filings.append({"cik": cik, "accession": acc, "date": date, "doc": doc})
    return filings[:lookback_n]


def parse_form4_for_ceo_purchases(cik, accession, doc):
    """Download a Form 4 primary XML document and extract open-market
    PURCHASE (transactionCode == 'P') transactions made by the CEO,
    including price per share and computed transaction value."""
    acc_nodash = accession.replace("-", "")
    base_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/"

    # Primary doc is often already .xml; if it's .html, the sibling XML
    # usually shares the same basename.
    candidates = [doc]
    if not doc.lower().endswith(".xml"):
        candidates.append(re.sub(r"\.\w+$", ".xml", doc))

    root = None
    for candidate in candidates:
        try:
            r = requests.get(base_url + candidate, headers=SEC_HEADERS, timeout=30)
            if r.status_code == 200 and b"<ownershipDocument" in r.content:
                root = ET.fromstring(r.content)
                break
        except (requests.RequestException, ET.ParseError):
            continue

    if root is None:
        return []

    issuer = root.find(".//issuer")
    ticker_el = issuer.find("issuerTradingSymbol") if issuer is not None else None
    ticker = ticker_el.text.strip() if ticker_el is not None and ticker_el.text else None

    owner = root.find(".//reportingOwner")
    owner_name, is_ceo = None, False
    if owner is not None:
        name_el = owner.find(".//rptOwnerName")
        owner_name = name_el.text.strip() if name_el is not None and name_el.text else None

        rel = owner.find(".//reportingOwnerRelationship")
        if rel is not None:
            title_el = rel.find("officerTitle")
            if title_el is not None and title_el.text:
                t = title_el.text.upper()
                if "CEO" in t or "CHIEF EXECUTIVE" in t:
                    is_ceo = True

    if not is_ceo:
        return []

    results = []
    for txn in root.findall(".//nonDerivativeTransaction"):
        code_el = txn.find(".//transactionCode")
        code = code_el.text if code_el is not None else None
        if code != "P":  # P = open-market/private purchase
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
        date = date_el.text if date_el is not None else None

        transaction_value = (shares * price) if (shares is not None and price) else None

        results.append(
            {
                "date": date,
                "shares": shares,
                "price_per_share": price,
                "transaction_value": transaction_value,
                "ticker": ticker,
                "director": owner_name,
                "market": "NASDAQ",
                "source_url": base_url + doc,
            }
        )
    return results


def collect_nasdaq_ceo_purchases(max_companies=25, filings_per_company=20):
    """Loop over NASDAQ companies, pull recent Form 4s, keep CEO purchases."""
    companies = get_nasdaq_tickers(max_companies=max_companies)
    all_rows = []

    for _, row in companies.iterrows():
        cik, name, ticker = row["cik"], row["name"], row["ticker"]
        log.info(f"[NASDAQ] Checking {ticker} ({name})...")
        try:
            filings = get_recent_form4_filings(cik, lookback_n=filings_per_company)
        except Exception as e:
            log.warning(f"  failed to list filings for {ticker}: {e}")
            continue

        for f in filings:
            time.sleep(REQUEST_DELAY_SEC)
            try:
                rows = parse_form4_for_ceo_purchases(f["cik"], f["accession"], f["doc"])
            except Exception as e:
                log.warning(f"  failed to parse {f['accession']}: {e}")
                continue
            for r in rows:
                r["company_name"] = name
                if not r.get("ticker"):
                    r["ticker"] = ticker
                all_rows.append(r)

        time.sleep(REQUEST_DELAY_SEC)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# MARKET CAP ENRICHMENT (yfinance)
# ---------------------------------------------------------------------------

def add_market_cap(df):
    """Adds a market_cap column by looking up each unique ticker via yfinance."""
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
# MAIN
# ---------------------------------------------------------------------------

def main():


    log.info("Collecting NASDAQ CEO Form 4 purchases...")
    df = collect_nasdaq_ceo_purchases(
        max_companies=NASDAQ_MAX_COMPANIES,
        filings_per_company=NASDAQ_FILINGS_PER_COMPANY,
    )

    if df.empty:
        log.warning("No purchase transactions found.")
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["shares"])  # drop rows we couldn't extract a share count for
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    log.info("Enriching with market cap via yfinance...")
    df = add_market_cap(df)

    # Final tidy column order
    df = df[
        [
            "date",
            "shares",
            "price_per_share",
            "transaction_value",
            "ticker",
            "company_name",
            "director",
            "market",
            "market_cap",
            "source_url",
        ]
    ]

    out_path = "ceo_purchases.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved {len(df)} rows to {out_path}")

    print(df.head(20).to_string())
    return df


if __name__ == "__main__":
    main()
