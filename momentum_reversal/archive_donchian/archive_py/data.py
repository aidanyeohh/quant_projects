"""Download and cache daily OHLCV for every ASX-listed company plus benchmarks.

The ASX 500 universe is rebuilt point-in-time in `universe.py` from this panel
(top 500 by trailing traded value), because no free point-in-time constituent
list exists.
"""
import io
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
START = "2005-01-01"
BENCHMARKS = ["^AXJO", "^AORD"]  # ASX 200, All Ordinaries (~ASX 500)


def listed_companies() -> pd.DataFrame:
    path = DATA / "asx_listed.csv"
    if not path.exists():
        r = requests.get("https://www.asx.com.au/asx/research/ASXListedCompanies.csv",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), skiprows=1)
        df.columns = ["name", "code", "sector"]
        df.to_csv(path, index=False)
    return pd.read_csv(path)


def download(tickers, batch=100):
    frames = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        for attempt in range(3):
            try:
                d = yf.download(chunk, start=START, auto_adjust=False, actions=False,
                                progress=False, threads=True, group_by="column")
                break
            except Exception as e:
                print("retry", i, e)
                time.sleep(5)
        d = d.stack(level=1, future_stack=True).rename_axis(["date", "ticker"])
        frames.append(d.dropna(how="all"))
        print(f"{i + len(chunk)}/{len(tickers)}")
    return pd.concat(frames)


def build_panel():
    out = DATA / "ohlcv.parquet"
    if out.exists():
        return pd.read_parquet(out)
    codes = listed_companies()["code"].astype(str)
    tickers = sorted(c + ".AX" for c in codes)
    panel = download(tickers)
    panel.columns = [c.lower().replace(" ", "_") for c in panel.columns]
    panel.to_parquet(out)
    bench = download(BENCHMARKS)
    bench.columns = [c.lower().replace(" ", "_") for c in bench.columns]
    bench.to_parquet(DATA / "benchmarks.parquet")
    return panel


if __name__ == "__main__":
    p = build_panel()
    print(p.shape, p.index.get_level_values("ticker").nunique())
