"""Monte Carlo benchmark: the breakout strategies vs random buy-and-hold portfolios.

Each simulation picks K stocks at random from the ASX 500 members on the start date,
buys them equal weight (20 bps once), and holds them to the end with no rebalancing.
Two windows are run: the full sample (2006-) and the out-of-sample period (2017-).
Two portfolio sizes: K=30 (a typical concentrated investor) and K matched to the
average number of longs the breakout strategy holds.

Survivorship caveat: every ticker in the panel is still listed in 2026, so a random
2006 pick is a stock that survived 20 years. The random portfolios carry the same
bias as the strategy, which is why the comparison is like for like.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from breakout import backtest_drift, donchian_state
from universe import load

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "results"
N_SIMS, SEED, COST = 1000, 7, 20e-4
WINDOWS = {"FULL 2006-": "2006-01-03", "OOS 2017-": "2017-01-03"}
BLUE, BLUE_LIGHT, BLUE_MID, ORANGE, AQUA, YELLOW = "#2a78d6", "#cde2fb", "#86b6ef", "#eb6834", "#1baf7a", "#eda100"


def metrics(r, bench):
    """Annualised metrics for a matrix of daily returns (days x portfolios)."""
    r = np.asarray(r)
    if r.ndim == 1:
        r = r[:, None]
    eq = np.cumprod(1 + r, axis=0)
    years = len(r) / 252
    b = np.asarray(bench)
    bc = b - b.mean()
    beta = (bc[:, None] * (r - r.mean(0))).sum(0) / (bc ** 2).sum()
    alpha = (r.mean(0) - beta * b.mean()) * 252
    vol = r.std(0, ddof=1) * np.sqrt(252)
    return pd.DataFrame({
        "cagr": eq[-1] ** (1 / years) - 1,
        "ann_vol": vol,
        "sharpe": r.mean(0) * 252 / vol,
        "max_dd": (eq / np.maximum.accumulate(eq, axis=0) - 1).min(0),
        "alpha_axjo": alpha,
        "beta_axjo": beta,
    })


d = load()
ret = d["ret"].fillna(0.0)
axjo = d["bench_ret"]["^AXJO"].fillna(0.0)

# strategies (same definitions as research.py)
s = donchian_state(d, 250, 125)
rank = d["adv"].where(d["member"]).rank(axis=1, ascending=False)
top200 = (rank <= 200).shift(1, fill_value=False)
s_impl = s.where((s > 0) | top200, 0.0)
r_impl = backtest_drift(s_impl, d, "neutral", 20, 2)["net"]
scale = (0.10 / (r_impl.rolling(63).std() * np.sqrt(252))).shift(1).clip(upper=3).fillna(0)
strategies = {
    "Long-only breakout": backtest_drift(s, d, "long", 20, 2)["net"],
    "L/S breakout (implementable)": r_impl,
    "L/S breakout, 10% vol target": backtest_drift(s_impl, d, "neutral", 20, 2, scale=scale)["net"],
}
k_match = int(round((s > 0).sum(axis=1).loc["2006":].mean()))
rng = np.random.default_rng(SEED)

rows, dists, fans = [], {}, {}
for wname, start in WINDOWS.items():
    r_win = ret.loc[start:]
    b_win = axjo.loc[start:]
    pool = np.flatnonzero(d["member"].loc[start:].iloc[0].values)
    growth = np.cumprod(1 + r_win.values, axis=0)  # value of $1 in each stock
    for k in [30, k_match]:
        picks = np.array([rng.choice(pool, size=k, replace=False) for _ in range(N_SIMS)])
        value = growth[:, picks].mean(axis=2) * (1 - COST)  # days x sims, equal weight at start, drifting
        daily = np.vstack([value[:1] - 1, value[1:] / value[:-1] - 1])
        m = metrics(daily, b_win)
        dists[(wname, k)] = m
        fans[(wname, k)] = value
        # control: the same random picks reset to equal weight every day (no costs). This is
        # not investable -- it harvests bid-ask bounce on closing prices -- and shows why the
        # strategies are run with drifting holdings (breakout.backtest_drift).
        reset = metrics(r_win.values[:, picks].mean(axis=2), b_win)
        for label, mm in [(f"random B&H, K={k}", m), (f"control: random daily-reset EW, K={k}", reset)]:
            for p, row in mm.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).iterrows():
                rows.append({"window": wname, "portfolio": label, "stat": f"p{int(p * 100)}", **row})
    for sname, sr in strategies.items():
        ms = metrics(sr.loc[start:].values, b_win).iloc[0]
        rows.append({"window": wname, "portfolio": sname, "stat": "value", **ms})
        for k in [30, k_match]:
            pct = {f"pctile_{c}": (dists[(wname, k)][c] < ms[c]).mean() * 100 for c in ["cagr", "sharpe", "max_dd", "alpha_axjo"]}
            rows.append({"window": wname, "portfolio": sname, "stat": f"percentile vs K={k}", **pct})

table = pd.DataFrame(rows)
table.to_csv(OUT / "10_mc_buyhold.csv", index=False)
pd.set_option("display.width", 250)
print(f"K matched to average breakout long count = {k_match}\n")
print(table.round(3).to_string(index=False))

# chart 1: Sharpe and max-drawdown distributions, strategies marked -----------------
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
for i, wname in enumerate(WINDOWS):
    for j, (col, label) in enumerate([("sharpe", "Sharpe ratio"), ("max_dd", "Max drawdown")]):
        ax = axs[j, i]
        for k, c in [(30, BLUE_MID), (k_match, BLUE)]:
            ax.hist(dists[(wname, k)][col], bins=40, color=c, alpha=0.75, edgecolor="white", linewidth=0.5,
                    label=f"random buy & hold, K={k}")
        for (sname, sr), c in zip(strategies.items(), [AQUA, ORANGE, YELLOW]):
            v = metrics(sr.loc[WINDOWS[wname]:].values, axjo.loc[WINDOWS[wname]:]).iloc[0][col]
            ax.axvline(v, color=c, lw=2, label=sname)
        ax.set_title(f"{label}, {wname}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
axs[0, 0].set_ylabel("Simulations")
axs[1, 0].set_ylabel("Simulations")
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False)
fig.suptitle(f"Breakout strategies vs {N_SIMS:,} random buy-and-hold portfolios per size", fontsize=12)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "mc_distributions.png", dpi=130)

# chart 2: fan of random buy-and-hold equity curves (K matched), full window --------
wname = "FULL 2006-"
value = fans[(wname, k_match)]
idx = ret.loc[WINDOWS[wname]:].index
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(idx, np.percentile(value, 5, axis=1), np.percentile(value, 95, axis=1), color=BLUE_LIGHT,
                label=f"random B&H K={k_match}: 5th-95th pct")
ax.fill_between(idx, np.percentile(value, 25, axis=1), np.percentile(value, 75, axis=1), color=BLUE_MID,
                label="25th-75th pct")
ax.plot(idx, np.median(value, axis=1), color=BLUE, lw=1.5, label="median")
for (sname, sr), c in zip(strategies.items(), [AQUA, ORANGE, YELLOW]):
    ax.plot(idx, (1 + sr.loc[WINDOWS[wname]:]).cumprod(), color=c, lw=2, label=sname)
ax.set_yscale("log")
ax.set_ylabel("Growth of $1 (log scale)")
ax.set_title("Breakout strategies vs random buy-and-hold, 2006-2026, net of costs", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e5e5", lw=0.6)
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "mc_equity_fan.png", dpi=130)
print("\ncharts written to", OUT)
