"""Clean the raw Yahoo panel and build a point-in-time ASX 500 universe.

Universe rule (rebuilt at each month end, applied the following month):
  * ordinary equities only (ASX GICS group != "Not Applic", which drops ETFs/LICs/trusts)
  * >= 252 trading days of price history
  * last close >= $0.10
  * rank by 63-day median traded value (close * volume); keep the top 500

Known limitation: the ticker list is today's ASX listing, so companies that
delisted before 2026 (takeovers, failures) are missing. See README.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
N_UNIVERSE = 500
MIN_PRICE = 0.10
MIN_HISTORY = 252
SPIKE = np.log(1.5)  # a +/-50% log move reversed the next day is treated as a bad print


def _clean_returns(adj: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    lr = np.log(adj).diff()
    # Yahoo occasionally breaks the adjustment factor (e.g. TAH.AX 2009-02-10: adj x4.5,
    # raw close -6%). A big adjusted move the raw close does not share is an error;
    # genuine splits show the opposite pattern (big raw move, small adjusted move).
    raw_lr = np.log(close).diff()
    lr = lr.mask((lr.abs() > SPIKE) & ((lr - raw_lr).abs() > SPIKE))
    nxt = lr.shift(-1)
    spike = (lr.abs() > SPIKE) & (np.sign(lr) == -np.sign(nxt)) & (nxt.abs() > 0.7 * lr.abs())
    bad = spike | spike.shift(1, fill_value=False)
    lr = lr.mask(bad)
    lr = lr.mask(lr.abs() > np.log(5))  # >400% / <-80% in one day: treat as data error
    return np.expm1(lr)


def load(cache=True) -> dict:
    out = DATA / "clean.pkl"
    if cache and out.exists():
        return pd.read_pickle(out)

    raw = pd.read_parquet(DATA / "ohlcv.parquet")
    meta = pd.read_csv(DATA / "asx_listed.csv")
    equities = set(meta.loc[meta.sector != "Not Applic", "code"] + ".AX")
    raw = raw[raw.index.get_level_values("ticker").isin(equities)]

    wide = {c: raw[c].unstack() for c in ["adj_close", "close", "high", "low", "volume"]}
    close = wide["close"].where(wide["close"] > 0)
    adj = wide["adj_close"].where(wide["adj_close"] > 0)
    factor = adj / close  # split/dividend adjustment applied to high/low too
    high = (wide["high"].where(wide["high"] > 0) * factor)
    low = (wide["low"].where(wide["low"] > 0) * factor)
    # Yahoo sometimes reports high < close on illiquid days; keep the channel consistent
    high = np.maximum(high, adj)
    low = np.minimum(low, adj)
    vol = wide["volume"].fillna(0)
    # days with zero volume are stale quotes, not trades
    traded = vol > 0
    ret = _clean_returns(adj, close).where(traded | traded.shift(1, fill_value=False))

    dollar_vol = (close * vol).where(traded)
    adv = dollar_vol.rolling(63, min_periods=40).median()
    history = adj.notna().cumsum()

    month_end = adj.index.to_series().groupby(adj.index.to_period("M")).max()
    elig = (history >= MIN_HISTORY) & (close >= MIN_PRICE) & adv.notna()
    rank = adv.where(elig).loc[month_end.values].rank(axis=1, ascending=False)
    member = (rank <= N_UNIVERSE)
    # membership decided on month-end t applies from the next trading day
    member = member.reindex(adj.index).shift(1).ffill().fillna(False).astype(bool)

    sector = meta.assign(ticker=meta.code + ".AX").set_index("ticker")["sector"]
    bench = pd.read_parquet(DATA / "benchmarks.parquet")["adj_close"].unstack()
    bench_ret = bench.pct_change(fill_method=None).reindex(adj.index)

    d = dict(adj=adj, high=high, low=low, close=close, volume=vol, ret=ret, adv=adv,
             member=member, sector=sector.reindex(adj.columns), bench_ret=bench_ret)
    pd.to_pickle(d, out)
    return d


if __name__ == "__main__":
    d = load(cache=False)
    m = d["member"]
    print("members per year (mean daily count):")
    print(m.sum(axis=1).groupby(m.index.year).mean().round(0).to_string())
    print("distinct names ever in universe:", m.any().sum())
