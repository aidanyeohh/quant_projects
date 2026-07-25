"""
daily_update_form4.py

Daily incremental update of NASDAQ insider Form 4 BUY/SELL transactions.
Intended to run once a day (e.g. via cron) AFTER backfill_form4_history.py
has established the base dataset -- it picks up automatically from
`state.json`'s last_processed_date (shared with the backfill script) and
appends only new filings to the same CSV.

Uses SEC's DAILY index (one file per calendar day, listing every filing
by every company that day) rather than scanning per-company, since after
the initial backfill there's only ever a small number of new days to
check.

------------------------------------------------------------------------
Usage:
    python3 daily_update_form4.py                          # resume from state
    python3 daily_update_form4.py --start-date 2026-07-01   # manual override
    python3 daily_update_form4.py --end-date 2026-07-15
------------------------------------------------------------------------
"""

import time
import argparse
from datetime import date, timedelta

import form4_common as f4

log = f4.log


def run_daily_update(start_date=None, end_date=None, csv_path=f4.CSV_PATH, state_path=f4.STATE_PATH):
    state = f4.load_state(state_path)
    seen_accessions = set(state.get("seen_accessions", []))

    if start_date is None:
        if state.get("last_processed_date"):
            start_date = date.fromisoformat(state["last_processed_date"]) + timedelta(days=1)
        else:
            log.warning(
                "No prior state found -- have you run backfill_form4_history.py yet? "
                "Defaulting to a 7-day catch-up window."
            )
            start_date = date.today() - timedelta(days=7)
    if end_date is None:
        end_date = date.today() - timedelta(days=1)  # yesterday -- today's index isn't final yet

    if start_date > end_date:
        log.info(f"Nothing to do: start_date ({start_date}) is after end_date ({end_date}).")
        return

    nasdaq_ciks = f4.get_nasdaq_ciks()
    all_new_rows = []
    last_successful_date = None

    day = start_date
    while day <= end_date:
        log.info(f"Checking daily index for {day}...")
        filings = f4.fetch_daily_index(day, nasdaq_ciks)
        time.sleep(f4.REQUEST_DELAY_SEC)

        if filings is None:
            log.info(f"  no index published for {day} (weekend/holiday) -- skipping")
            last_successful_date = day
            day += timedelta(days=1)
            continue

        log.info(f"  {len(filings)} NASDAQ Form 4 filing(s) found")
        for filing in filings:
            if filing["accession"] in seen_accessions:
                continue

            doc = f4.get_primary_doc_filename(filing["cik"], filing["accession"])
            time.sleep(f4.REQUEST_DELAY_SEC)
            if doc is None:
                log.warning(f"  could not locate XML for accession {filing['accession']}")
                seen_accessions.add(filing["accession"])
                continue

            rows = f4.parse_form4_for_insider_transactions(filing["cik"], filing["accession"], doc)
            time.sleep(f4.REQUEST_DELAY_SEC)

            for row in rows:
                row["company_name"] = filing["company_name"]
                row["date_filed"] = filing["date"]  # actual SEC filing date, distinct from transaction date
                if not row.get("ticker"):
                    row["ticker"] = nasdaq_ciks.get(filing["cik"], {}).get("ticker")
                all_new_rows.append(row)

            seen_accessions.add(filing["accession"])

        last_successful_date = day
        day += timedelta(days=1)

    # Persist state even on a zero-result run, so we don't recheck the
    # same days next time.
    if last_successful_date is not None:
        state["last_processed_date"] = last_successful_date.isoformat()
    state["seen_accessions"] = sorted(seen_accessions)
    f4.save_state(state, state_path)

    combined = f4.merge_and_save(all_new_rows, csv_path)
    if not combined.empty and all_new_rows:
        print(combined.head(20).to_string())


def main():
    parser = argparse.ArgumentParser(description="Daily update of NASDAQ insider Form 4 transactions.")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD, overrides saved state")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD, defaults to yesterday")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date) if args.start_date else None
    end_date = date.fromisoformat(args.end_date) if args.end_date else None

    run_daily_update(start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    main()