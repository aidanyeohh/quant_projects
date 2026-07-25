"""
backfill_form4_history.py

ONE-TIME (or resumable) load of NASDAQ insider Form 4 BUY/SELL
transactions from a given start date (default: 2000-01-01) through the
present, using SEC's QUARTERLY full-index files:

    https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx

Each file lists EVERY filing SEC received that entire quarter, across
every company. We download ~1 file per quarter (roughly 104 requests to
cover 2000-present), filter locally to Form 4 filings from NASDAQ CIKs,
then fetch/parse only those specific filings.

------------------------------------------------------------------------
BE REALISTIC ABOUT RUNTIME
------------------------------------------------------------------------
Discovery is cheap (~104 requests total). But NASDAQ has ~3,000+ listed
companies, and over 25 years that adds up to a LARGE number of matching
Form 4 filings -- easily tens of thousands to low hundreds of thousands.
Each matching filing costs 2 requests (locate the XML, then fetch it),
rate-limited to stay under SEC's guidelines. Concretely: at ~3 requests/sec
sustained, 100,000 filings x 2 requests = ~18 hours of continuous runtime.
This is a fundamental consequence of the data volume, not something a
smarter script can avoid -- SEC doesn't offer a "give me the bulk data"
endpoint for this.

This script is written to be SAFELY INTERRUPTIBLE AND RESUMABLE:
  - Progress is checkpointed after each quarter completes.
  - Within a quarter, already-seen accession numbers (recorded in
    state.json) are skipped without a network call, so re-running after
    an interruption doesn't redo completed work.
  - Just re-run the same command to pick up where it left off.

Recommended: run this in the background (e.g. `nohup python3
backfill_form4_history.py &` or inside `tmux`/`screen`), and use
daily_update_form4.py for ongoing updates once it's done.
------------------------------------------------------------------------
"""

import time
import argparse
from datetime import date

import form4_common as f4

log = f4.log


def run_backfill(start_year=2000, start_quarter=1, end_year=None, end_quarter=None,
                  csv_path=f4.CSV_PATH, state_path=f4.STATE_PATH):
    if end_year is None or end_quarter is None:
        end_year, end_quarter = f4.current_year_quarter()

    state = f4.load_state(state_path)
    seen_accessions = set(state.get("seen_accessions", []))

    nasdaq_ciks = f4.get_nasdaq_ciks()
    log.info(f"Tracking {len(nasdaq_ciks)} NASDAQ-listed companies")

    quarters = list(f4.iter_quarters(start_year, start_quarter, end_year, end_quarter))

    # Resume support: skip quarters already fully processed in a prior run.
    last_q = state.get("last_processed_quarter")
    if last_q:
        last_year, last_quarter = last_q
        quarters = [
            (y, q) for (y, q) in quarters
            if (y, q) > (last_year, last_quarter)
        ]
        if quarters:
            log.info(f"Resuming: already completed through {last_year} QTR{last_quarter}")
        else:
            log.info("Nothing to do -- backfill already covers the requested range.")
            return

    for year, quarter in quarters:
        log.info(f"=== Processing {year} QTR{quarter} ===")
        filings = f4.fetch_quarterly_index(year, quarter, nasdaq_ciks)
        time.sleep(f4.REQUEST_DELAY_SEC)

        if filings is None:
            log.warning(f"  no quarterly index available for {year} QTR{quarter} -- skipping")
            _checkpoint(state, state_path, year, quarter, seen_accessions)
            continue

        log.info(f"  {len(filings)} NASDAQ Form 4 filing(s) found in this quarter")

        new_rows = []
        new_count = 0
        for i, filing in enumerate(filings, 1):
            if filing["accession"] in seen_accessions:
                continue

            doc = f4.get_primary_doc_filename(filing["cik"], filing["accession"])
            time.sleep(f4.REQUEST_DELAY_SEC)
            if doc is None:
                log.warning(f"  could not locate XML for accession {filing['accession']}")
                seen_accessions.add(filing["accession"])  # don't retry a known-bad filing forever
                continue

            rows = f4.parse_form4_for_insider_transactions(filing["cik"], filing["accession"], doc)
            time.sleep(f4.REQUEST_DELAY_SEC)

            for row in rows:
                row["company_name"] = filing["company_name"]
                row["date_filed"] = filing["date"]  # actual SEC filing date, distinct from transaction date
                if not row.get("ticker"):
                    row["ticker"] = nasdaq_ciks.get(filing["cik"], {}).get("ticker")
                new_rows.append(row)

            seen_accessions.add(filing["accession"])
            new_count += 1

            # Periodically flush progress to disk within a large quarter,
            # so a crash mid-quarter doesn't lose everything processed so far.
            # compute_clusters=False here: the cluster-buy pass scans the
            # WHOLE dataset, so running it on every 250-filing checkpoint
            # would be wasteful -- it's run once at quarter-end instead.

            if i % 250 == 0:
                log.info(f"  ...{i}/{len(filings)} filings checked in this quarter, "
                         f"{len(new_rows)} transaction rows found so far")
                f4.merge_and_save(new_rows, csv_path, compute_clusters=False)
                new_rows = []
                state["seen_accessions"] = sorted(seen_accessions)
                f4.save_state(state, state_path)

        f4.merge_and_save(new_rows, csv_path, compute_clusters=True)
        _checkpoint(state, state_path, year, quarter, seen_accessions)
        log.info(f"  quarter complete: {new_count} new filings processed")

    log.info("Backfill complete.")


def _checkpoint(state, state_path, year, quarter, seen_accessions):
    state["last_processed_quarter"] = [year, quarter]
    # Also advance last_processed_date to the quarter's end date, so
    # daily_update_form4.py knows to start the day after.
    quarter_end_month = quarter * 3
    if quarter_end_month == 12:
        end_date = date(year, 12, 31)
    else:
        next_month_first = date(year, quarter_end_month + 1, 1)
        from datetime import timedelta
        end_date = next_month_first - timedelta(days=1)
    state["last_processed_date"] = min(end_date, date.today()).isoformat()
    state["seen_accessions"] = sorted(seen_accessions)
    f4.save_state(state, state_path)


def main():
    parser = argparse.ArgumentParser(description="Backfill full NASDAQ insider Form 4 history.")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--start-quarter", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--end-quarter", type=int, default=None, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    run_backfill(
        start_year=args.start_year,
        start_quarter=args.start_quarter,
        end_year=args.end_year,
        end_quarter=args.end_quarter,
    )


if __name__ == "__main__":
    main()