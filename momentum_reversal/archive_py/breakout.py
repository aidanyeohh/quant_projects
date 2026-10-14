"""Channel-breakout momentum signals and an unconstrained daily backtester.

Conventions
  * Signals use data up to and including the close of day t.
  * Weights set at the close of t earn the return of t+1 (`lag=1`);
    `lag=2` delays trading by a further day as a robustness check.
  * Costs: `cost_bps` per unit of one-way turnover (20 bps = market orders,
    7 bps = limit orders, per the project brief).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

TRADING_DAYS = 252


# --------------------------------------------------------------------------- signals
def channels(d, n):
    """Upper/lower Donchian channel over the n days *before* t (excludes t)."""
    upper = d["high"].rolling(n, min_periods=n).max().shift(1)
    lower = d["low"].rolling(n, min_periods=n).min().shift(1)
    return upper, lower


def donchian_state(d, n, m=None, vol_filter=None):
    """Stateful Turtle-style breakout.

    Long  entry: close > n-day high.   Long  exit: close < m-day low.
    Short entry: close < n-day low.    Short exit: close > m-day high.
    m defaults to n // 2. Positions are forced flat outside the universe.

    vol_filter: None, "high" or "low" -- keep only entries whose breakout-day
    volume is above / below 1.5x its 50-day average (activity conditioning).
    """
    m = m or max(n // 2, 2)
    px = d["adj"]
    up_n, lo_n = channels(d, n)
    up_m, lo_m = channels(d, m)
    long_in, short_in = px > up_n, px < lo_n
    if vol_filter:
        rel = d["volume"] / d["volume"].rolling(50, min_periods=30).mean().shift(1)
        hot = rel > 1.5
        keep = hot if vol_filter == "high" else ~hot
        long_in, short_in = long_in & keep, short_in & keep
    member = d["member"]

    def state(enter, leave):
        s = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
        s = s.mask(leave | ~member, 0.0).mask(enter & member, 1.0)
        return s.ffill().fillna(0.0)

    long_s = state(long_in, px < lo_m)
    short_s = state(short_in, px > up_m)
    return long_s - short_s


def channel_position(d, n):
    """Continuous version: where the close sits in its n-day range, in [-1, 1]."""
    hi = d["high"].rolling(n, min_periods=n).max()
    lo = d["low"].rolling(n, min_periods=n).min()
    pos = 2 * (d["adj"] - lo) / (hi - lo) - 1
    return pos.where(d["member"])


# --------------------------------------------------------------------------- weights
def weights(signal, scheme="neutral", vol=None):
    """Turn a signal into portfolio weights.

    neutral : longs sum to +0.5, shorts to -0.5 (dollar neutral, gross 1)
    net     : w = s / sum|s| (gross 1, net exposure follows the signal)
    long    : longs only, sum to 1
    rank    : cross-sectional demeaned signal scaled to gross 1 (for continuous signals)
    vol: optional per-name volatility; signal is divided by it first (risk parity).
    """
    s = signal.copy()
    if vol is not None:
        s = s / vol.where(vol > 0)
    s = s.fillna(0.0)
    if scheme == "rank":
        r = signal.rank(axis=1, pct=True)
        r = r.sub(r.mean(axis=1), axis=0)
        return r.div(r.abs().sum(axis=1), axis=0).fillna(0.0)
    if scheme == "net":
        return s.div(s.abs().sum(axis=1), axis=0).fillna(0.0)
    pos, neg = s.clip(lower=0), (-s).clip(lower=0)
    if scheme == "long":
        return pos.div(pos.sum(axis=1), axis=0).fillna(0.0)
    w_long = 0.5 * pos.div(pos.sum(axis=1), axis=0)
    w_short = 0.5 * neg.div(neg.sum(axis=1), axis=0)
    return (w_long.fillna(0.0) - w_short.fillna(0.0))


# --------------------------------------------------------------------------- backtest
def backtest(w, d, cost_bps=20, lag=1):
    w = w.shift(lag - 1) if lag > 1 else w
    r = d["ret"].reindex_like(w).fillna(0.0)
    gross = (w.shift(1) * r).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    net = gross - turnover * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "net": net, "turnover": turnover,
                         "long": w.clip(lower=0).sum(axis=1), "short": w.clip(upper=0).sum(axis=1)})


def backtest_drift(signal, d, scheme="neutral", cost_bps=20, lag=2, scale=None):
    """Holdings-based backtest: positions drift with price and only entries/exits trade.

    `backtest` resets every name to equal weight each day and charges only for changes
    in *target* weights. On illiquid closing prices that daily reset quietly harvests
    bid-ask bounce (a random equal-weight portfolio rebalanced daily shows Sharpe ~2).
    Here a name keeps its drifted weight while held; a new entry gets gross/n_side and
    the rest of its side is scaled pro rata to keep the side's gross fixed. Every trade,
    including that pro-rata scaling, is charged.

    scheme: "neutral" (longs +0.5 / shorts -0.5) or "long" (longs sum to 1).
    scale: optional per-day multiplier on gross (vol targeting), known at t.
    """
    sig = np.sign(signal.shift(lag - 1).fillna(0.0)).values if lag > 1 else np.sign(signal.fillna(0.0)).values
    r = d["ret"].reindex_like(signal).fillna(0.0).values
    k = np.ones(len(sig)) if scale is None else scale.reindex(signal.index).fillna(0.0).values
    sides = [(1, 0.5), (-1, 0.5)] if scheme == "neutral" else [(1, 1.0)]
    h = np.zeros(sig.shape[1])
    gross, turn = np.zeros(len(sig)), np.zeros(len(sig))
    for t in range(len(sig)):
        if t:
            pnl = h @ r[t]
            gross[t] = pnl
            h = h * (1 + r[t]) / (1 + pnl)
        new = np.zeros_like(h)
        for side, g in sides:
            g = g * k[t]
            on = sig[t] == side
            if not on.any() or g == 0:
                continue
            keep = on & (np.sign(h) == side) & (h != 0)
            enter = on & ~keep
            n = on.sum()
            w_enter = g / n
            rest = g - w_enter * enter.sum()
            held = np.abs(h[keep]).sum()
            new[enter] = side * w_enter
            if keep.any():
                new[keep] = h[keep] * (rest / held if held > 0 else 0)
        turn[t] = np.abs(new - h).sum()
        h = new
    gross = pd.Series(gross, signal.index)
    turn = pd.Series(turn, signal.index)
    # pnl at t uses holdings set at t-1; cost of trades at t is paid at t
    return pd.DataFrame({"gross": gross, "net": gross - turn * cost_bps / 1e4, "turnover": turn,
                         "long": np.nan, "short": np.nan})


def stats(r, bench=None, name=None):
    r = r.dropna()
    ann_ret = r.mean() * TRADING_DAYS
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    out = {"ann_ret": ann_ret, "ann_vol": ann_vol,
           "sharpe": ann_ret / ann_vol if ann_vol else np.nan,
           "max_dd": dd, "calmar": ann_ret / -dd if dd else np.nan,
           "t_stat": r.mean() / r.std() * np.sqrt(len(r)), "days": len(r)}
    if bench is not None:
        b = bench.reindex(r.index).fillna(0.0)
        fit = sm.OLS(r, sm.add_constant(b)).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
        out.update(alpha=fit.params["const"] * TRADING_DAYS, alpha_t=fit.tvalues["const"],
                   beta=fit.params.iloc[1])
    return pd.Series(out, name=name)


def summarize(bt, bench, name, col="net"):
    s = stats(bt[col], bench, name)
    s["turnover"] = bt["turnover"].mean()
    s["avg_long"], s["avg_short"] = bt["long"].mean(), bt["short"].mean()
    return s
