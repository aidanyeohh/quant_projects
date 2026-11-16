"""Channel-breakout momentum on the ASX 500: full research run.

Writes every table and chart used in README.md to ./results.
Run order:  python data.py  ->  python universe.py  ->  python research.py
In-sample (IS) 2006-2016 is used for every design choice; 2017-2026 is the
out-of-sample (OOS) check.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from breakout import backtest_drift, donchian_state, summarize
from universe import load

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
IS, OOS = slice("2006-01-01", "2016-12-31"), slice("2017-01-01", None)
COLS = ["ann_ret", "ann_vol", "sharpe", "max_dd", "alpha", "alpha_t", "beta", "turnover"]

d = load()
axjo = d["bench_ret"]["^AXJO"]
in_univ = d["member"].shift(1, fill_value=False)
# Equal-weight ASX 500 with drifting positions (same survivorship as the strategy). A daily-reset
# equal weight would harvest bid-ask bounce on closing prices and is not investable; see backtest_drift.
ew = backtest_drift(d["member"].astype(float), d, "long", cost_bps=0, lag=1)["net"]


def split(bt, name, bench=ew, col="net"):
    return [summarize(bt.loc[sl], bench, f"{name} | {per}", col) for per, sl in [("IS", IS), ("OOS", OOS)]]


def save(rows, fname):
    t = pd.DataFrame(rows)
    t.to_csv(OUT / fname)
    print(f"\n== {fname}\n{t[[c for c in COLS if c in t]].round(3).to_string()}")
    return t


# 1. Benchmarks ------------------------------------------------------------------
save([summarize(pd.DataFrame({"net": ew, "turnover": 0, "long": 1, "short": 0}).loc[sl], axjo, f"EW ASX500 | {p}")
      for p, sl in [("IS", IS), ("OOS", OOS)]], "01_benchmark.csv")

# 2. Time-horizon sweep: where does breakout momentum live? ------------------------
rows = []
for n in [5, 10, 20, 50, 100, 150, 200, 250, 300, 400]:
    s = donchian_state(d, n)
    for lag in [1, 2]:
        bt = backtest_drift(s, d, "neutral", 20, lag)
        for st in split(bt, f"N{n} lag{lag}"):
            g = split(bt, "", col="gross")[0 if st.name.endswith("IS") else 1]
            st["gross_sharpe"] = g["sharpe"]
            st["n"], st["lag"], st["per"] = n, lag, st.name.split("| ")[1]
            rows.append(st)
hz = save(rows, "02_horizon_sweep.csv")

fig, ax = plt.subplots(figsize=(8, 4.5))
for (lag, per), g in hz.groupby(["lag", "per"]):
    ax.plot(g["n"], g["sharpe"], marker="o", ls="-" if per == "IS" else "--",
            label=f"{per}, trade {'same' if lag == 1 else 'next'}-day close")
ax.axhline(0, c="k", lw=0.8)
ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 50, 100, 250, 400]); ax.set_xticklabels([5, 10, 20, 50, 100, 250, 400])
ax.set_ylim(-6, 2); ax.set_xlabel("Channel length N (days)"); ax.set_ylabel("Net Sharpe (20 bps)")
ax.set_title("Long-short Donchian breakout: Sharpe by horizon"); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(OUT / "horizon_sweep.png", dpi=130)

# 3. Exit-channel ratio M/N on IS only (choose spec here) ---------------------------
rows = []
for n in [150, 200, 250, 300]:
    for ratio in [0.25, 0.5, 1.0]:
        bt = backtest_drift(donchian_state(d, n, int(n * ratio)), d, "neutral", 20, 2)
        st = summarize(bt.loc[IS], ew, f"N{n} M{int(n * ratio)} | IS")
        rows.append(st)
save(rows, "03_exit_grid_IS.csv")

# 4. Activity conditioning: breakouts on high vs low volume -------------------------
N, M = 250, 125
rows = []
for vf in [None, "high", "low"]:
    s = donchian_state(d, N, M, vol_filter=vf)
    rows += split(backtest_drift(s, d, "neutral", 20, 2), f"volume={vf or 'any'}")
save(rows, "04_volume_filter.csv")

# 5. Final strategy and variants ----------------------------------------------------
s = donchian_state(d, N, M)  # volume filter did not beat "any" on IS once drift is modelled
rank = d["adv"].where(d["member"]).rank(axis=1, ascending=False)
top200 = (rank <= 200).shift(1, fill_value=False)
s_impl = s.where((s > 0) | top200, 0.0)  # shorts only in the 200 most liquid names

variants = {
    "Research: L/S ASX500": (s, "neutral"),
    "Implementable: L ASX500 / S top200": (s_impl, "neutral"),
    "Long-only ASX500": (s, "long"),
}
bts = {k: backtest_drift(sig, d, sch, 20, 2) for k, (sig, sch) in variants.items()}
# vol-targeted implementable version: scale to 10% using trailing 63d realised vol (known at t)
r = bts["Implementable: L ASX500 / S top200"]["net"]
scale = (0.10 / (r.rolling(63).std() * np.sqrt(252))).shift(1).clip(upper=3).fillna(0)
bts["Implementable, 10% vol target"] = backtest_drift(s_impl, d, "neutral", 20, 2, scale=scale)

rows = []
for k, bt in bts.items():
    for st in split(bt, k):
        st["alpha_axjo"], st["beta_axjo"] = summarize(bt.loc[IS if "IS" in st.name else OOS], axjo, "")[["alpha", "beta"]]
        rows.append(st)
    full = summarize(bt.loc["2006":], ew, f"{k} | FULL")
    full["alpha_axjo"], full["beta_axjo"] = summarize(bt.loc["2006":], axjo, "")[["alpha", "beta"]]
    rows.append(full)
final = save(rows, "05_final_strategies.csv")
print(final[["alpha_axjo", "beta_axjo"]].round(3).to_string())

# 6. Costs and execution ----------------------------------------------------------
w = s_impl
rows = []
for c in [0, 7, 20, 50]:
    for lag in [1, 2]:
        rows += split(backtest_drift(w, d, "neutral", c, lag), f"{c} bps, trade {'same' if lag == 1 else 'next'} day")
save(rows, "06_costs.csv")

# 7. Legs and calendar years --------------------------------------------------------
bt = bts["Implementable: L ASX500 / S top200"]
legs = {"long leg": backtest_drift(w.clip(lower=0), d, "neutral", 20, 2)["net"],
        "short leg": backtest_drift(w.clip(upper=0), d, "neutral", 20, 2)["net"]}
yearly = pd.DataFrame({"L/S strategy": bt["net"], **legs, "EW ASX500": ew, "ASX200": axjo}).loc["2006":]
yearly = yearly.groupby(yearly.index.year).apply(lambda x: (1 + x).prod() - 1)
yearly.to_csv(OUT / "07_yearly.csv")
print("\n== 07_yearly.csv\n", yearly.round(3).to_string())

# 8. Liquidity of what we hold ------------------------------------------------------
held_adv = (d["adv"].where(w != 0).median(axis=1) / 1e6).loc["2006":]
held_adv.groupby(held_adv.index.year).median().rename("median ADV of held names, A$m").to_csv(OUT / "08_liquidity.csv")

# 9. Size control: the implementable book is long small / short large names -------
import statsmodels.api as sm
ew200 = backtest_drift((d["member"] & top200).astype(float), d, "long", cost_bps=0, lag=1)["net"]
factors = pd.DataFrame({"ew_asx500": ew, "size (ew500 - ew200)": ew - ew200})
rows = []
for k in ["Research: L/S ASX500", "Implementable: L ASX500 / S top200"]:
    for per, sl in [("IS", IS), ("OOS", OOS)]:
        fit = sm.OLS(bts[k]["net"].loc[sl], sm.add_constant(factors.loc[sl].fillna(0))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 10})
        rows.append(pd.Series({"alpha": fit.params["const"] * 252, "alpha_t": fit.tvalues["const"],
                               **{f"beta {c}": fit.params[c] for c in factors}}, name=f"{k} | {per}"))
t = pd.DataFrame(rows)
t.to_csv(OUT / "09_size_control.csv")
print("\n== 09_size_control.csv\n", t.round(3).to_string())

# Charts ---------------------------------------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
for k in ["Research: L/S ASX500", "Implementable: L ASX500 / S top200", "Implementable, 10% vol target"]:
    eq = (1 + bts[k]["net"].loc["2006":]).cumprod()
    axs[0].plot(eq, label=k)
    axs[1].plot(eq / eq.cummax() - 1, label=k)
axs[0].plot((1 + axjo.loc["2006":].fillna(0)).cumprod(), c="grey", lw=1, label="ASX 200 (reference)")
axs[0].axvline(pd.Timestamp("2017-01-01"), c="k", ls=":", lw=1)
axs[0].set_yscale("log"); axs[0].set_title("Channel breakout N=250/M=125, net of 20 bps, dollar neutral")
axs[0].legend(fontsize=8); axs[1].set_ylabel("Drawdown")
fig.tight_layout(); fig.savefig(OUT / "equity_curves.png", dpi=130)
print("\nresults written to", OUT)
