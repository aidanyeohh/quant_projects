# ASX Statistical Arbitrage: Momentum and Reversal

Course project for The Wall Street Quants, transplanted from crypto to
Australian equities. The goal is unchanged: find cross-sectional price-volume
patterns that predict returns, trade them in an unconstrained dollar-neutral
book, and report performance honestly net of costs.

## Why the ASX changes the problem

The brief was written for crypto, where the market never closes, there is no
corporate action to speak of, and every asset that ever traded still has a
price series. Three of those assumptions break on the ASX.

**Companies die.** Roughly half the names that traded on the ASX in 2010 are no
longer listed. If the backtest universe is today's S&P/ASX 300, the sample
contains only the companies that survived, and a momentum backtest on that
sample is close to guaranteed to look good: the names momentum would have been
short are exactly the ones that later delisted and dropped out of the list. See
[Survivorship](#survivorship) below - it is the central design constraint of
this repository, not a footnote.

**The market is shut most of the time.** The brief's seasonality section asks
about weekday-versus-weekend and institutional-versus-retail hours. On the ASX
the weekend question disappears and is replaced by a sharper one: the 30 June
Australian financial year end concentrates tax-loss selling into May and June
and produces a July bounce, and the February and August reporting seasons are
when new information arrives in bulk. `signals.financial_year_flags` exposes
both.

**The cross-section is small and top-heavy.** BHP, CBA and CSL are a large
share of the index, and the resources-versus-financials axis dominates
co-movement. A naive long-short book is really a bet on iron ore. The residual
signals in `signals.residual_returns` strip statistical factors out before the
reversal signal is computed, which is the brief's "Security A minus something
correlated to it" idea done without needing a sector classification.

## Layout

```
src/asxstat/
  config.py        parameters: universe rules, cost model, backtest settings
  universe.py      ticker-space recovery and point-in-time membership
  download.py      bulk daily OHLCV into wide parquet panels
  clean.py         bad ticks, untraded days, delisting detection
  signals.py       momentum, reversal, conditioners, seasonality
  backtest.py      dollar-neutral book, costs, information coefficient
  metrics.py       returns, vol, Sharpe, drawdown, alpha/beta
  survivorship.py  bias measurement and stress scenarios
scripts/
  01_probe_universe.py      scan AAA..ZZZ for every ASX code with history
  02_download_prices.py     pull daily data for the recovered pool
  03_run_baseline.py        signals, IC, cost sweep, survivorship stress
  04_momentum_robustness.py momentum across universe sizes
tests/
  test_pipeline.py       end-to-end smoke test on synthetic data
```

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/test_pipeline.py -q     # 3 seconds, run this first
python scripts/01_probe_universe.py            # ~1 hr, resumable
python scripts/02_download_prices.py --start 2005-01-01
python scripts/03_run_baseline.py --start 2010-01-01 --universe-size 300
python scripts/04_momentum_robustness.py --start 2010-01-01
```

Steps 1 and 2 cache to `data/raw` after every chunk, so an interrupted run
resumes rather than restarting.

Yahoo silently returns empty frames for perfectly valid symbols when it decides
a request is too large or too frequent, which would be recorded as a delisting
and quietly corrupt the whole exercise. Step 1 defends against this by sending a
canary ticker in every chunk and retrying with backoff when the canary comes
back empty; step 2 repeats its passes until no new tickers arrive. Both are
worth keeping if the code is reused.

## Survivorship

Three separate problems hide under the one label. The repository treats them
separately because they need different fixes.

**1. Universe selection.** Solved structurally. `universe.point_in_time_universe`
rebuilds membership at each month end from trailing 60-day median turnover,
using only data available at that date. Index membership is never consulted, so
no forward-looking constituent list can leak in. `survivorship.survivor_only_members`
builds the biased version deliberately, so the size of the effect is a measured
number in the results rather than a claim in the write-up.

**2. Missing dead companies. Attempted, and it failed - this is the important
result in the repository.** Almost every ASX ordinary share has a three-letter
code, so `universe.probe_ticker_space` scans all 17,576 of AAA through ZZZ with
cheap monthly bars. That candidate set is independent of survival, unlike any
constituent list, so in principle it recovers companies no constituent list
would mention.

It does not, because the data is not there to recover. The full scan found
**1,637 ASX tickers with at least two years of history, of which 8 have stopped
trading.** Yahoo has purged essentially every delisted Australian name: `OZL`,
`SYD`, `CCL`, `CWN`, `WSA` and `SLR` were all acquired between 2021 and 2024 and
all return nothing. Stooq, the obvious free alternative, now gates scripted
access behind a JavaScript challenge.

So the sample here **is** survivor-only and cannot be fixed with free data.
Fixing it properly needs a vendor that keeps delisted securities and historical
index membership - Norgate Data is the Australian one built for exactly this
job. Everything below is what to do instead when that is not available.

**3. Bounding the bias instead of removing it.** Two independent bounds, both in
`scripts/03_run_baseline.py`.

*Simulated deaths.* `survivorship.simulate_delistings` injects delistings into
the survivor-only panel at a configurable annual hazard, then reruns the
strategy. The hazard is deliberately not uniform: it is multiplied by quintile
of trailing 12-month return, worst quintile 3x and best 0.4x, because companies
die after their price falls and a random simulation would understate the bias
badly. Across trials this gives a distribution of Sharpe under realistic
censoring. The gap between the survivor-only Sharpe and the simulated median is
the estimated bias; if the survivor-only number sits outside the simulated
range, the headline is mostly artefact.

*Universe size.* Large-cap ASX delistings are overwhelmingly takeovers, which
resolve at a premium already in the price, so the missing terminal return is
near zero. Small-cap delistings are collapses, where it is near -100%.
Delisting risk, borrow availability and spread therefore all worsen together as
the universe widens, and `run_universe_size_comparison` runs every signal at 300
names and at 100. **An edge that appears only at 300 names is the one to
distrust.**

`survivorship.apply_delisting_returns` remains for the handful of genuinely
recovered dead names, booking an explicit terminal return and treating a >20%
run-up into the exit as a takeover.

The bias does not run the same way for both strategy families. Momentum is
short the falling names, which are the ones that delist, so censoring their
final move makes momentum look *worse* than it was - survivor-only momentum
results are a **lower** bound. Reversal is long them, so survivor-only reversal
results are an **upper** bound. Quoting a single number for either, without the
simulation range beside it, is not a result.

## Costs

The brief assumes 20 bps all-in for crypto market orders. ASX equities are
cheaper on commission and more expensive on spread, and the spread is a strong
function of size. `config.CostModel` charges 8 bps commission plus slippage
that scales from 7 bps in the most liquid decile to 35 bps in the least, and
the backtest applies it per side on drift-adjusted turnover.

`metrics.break_even_cost_bps` reports the round-trip cost that would take the
gross book to zero. For a five-day reversal signal this is the number that
decides whether the strategy is real, and it is the first thing to check before
being pleased about a Sharpe.

## Results so far

Sample: 1,635 ASX tickers, 2010-01 to 2026-08, universe rebuilt monthly by
trailing turnover. Note that the liquidity and price floors bind before 300
names do, so the "ASX 300" universe is really a median of **281** eligible
names; asking for 500 returns the same 281.

**Momentum works and is tradable.** Risk-adjusted 12-1 momentum, rebalanced
monthly with five-day signal smoothing, market orders:

| | value |
|---|---|
| net return | 4.8% p.a. |
| volatility | 6.8% p.a. |
| Sharpe | 0.70 |
| max drawdown | -14.8% |
| beta to ASX 300 | -0.07 |
| alpha | 3.9% p.a., t = 1.96 |
| turnover | 5.4x p.a. |
| break-even cost | 122 bps round trip |

The break-even number is the one that matters: at 122 bps against a cost model
charging roughly 25, the result does not depend on the cost assumptions. The
alpha t-statistic of 1.96 is a different story - it is significant by the
thinnest possible margin over sixteen years, and should be quoted that way.

**Rebalancing frequency was worth more than any signal change.** The same
signal traded daily returns -1.3% net; traded monthly it returns +4.8%. Gross
return barely moves (5.8% to 6.5%) while turnover falls from 23.6x to 5.4x.
Before tuning a signal, check what the book is paying to chase it.

**Reversal does not survive costs.** Five-day reversal has a genuinely
predictive signal - information coefficient 0.026 at one day, t = 12, decaying
to zero by 21 days, and a top-minus-bottom quintile spread of 25% annualised
gross. It still loses money at every frequency tested. Break-even is 8 to 10
bps against a cost model charging roughly 25, and slowing the book down destroys
the gross faster than it saves cost, because a five-day signal traded monthly is
stale by the time it trades. Under limit-order execution the best variant makes
1.5% a year at a Sharpe of 0.20, and that figure assumes every passive order
fills, which is exactly the assumption a liquidity-providing strategy is least
entitled to make. **Verdict: no tradable reversal edge on the ASX at daily
frequency.** Neither the low-turnover tilt nor factor-neutralisation rescues it;
all three reversal variants are correlated above 0.89 and fail together.

**Survivorship, momentum.** Survivor-only Sharpe 0.38 against a simulated median
of 0.67, 5th-95th percentile 0.51 to 0.79, at a 5% annual hazard with terminal
returns varying by liquidity decile (both at the weekly configuration). Injecting
realistic deaths *helps* momentum, which is the direction theory predicts: the
book is short the names that die, and the survivor-only sample censors its best
trades. The reported momentum result is therefore a **lower** bound. Note the
hazard used is aggressive - it kills about 1,054 of 1,635 names over the sample
and shrinks the median book from 281 names to 190 - so treat the simulated
figures as a stress, not a forecast.

**The momentum edge lives in the smaller names, and this is the real
vulnerability.** Monthly rebalance, five-day smoothing, market orders:

| universe | names | net return | Sharpe | 21-day IC | IC t-stat |
|---|---|---|---|---|---|
| top 500 | 281 | 5.3% | 0.81 | 0.046 | 13.4 |
| top 300 | 281 | 4.8% | 0.70 | 0.044 | 12.4 |
| top 200 | 200 | 4.2% | 0.61 | 0.038 | 10.1 |
| top 100 | 100 | 1.8% | 0.27 | 0.025 | 5.6 |
| top 50 | 50 | 1.3% | 0.20 | 0.017 | 3.5 |

The decay is monotone and, importantly, it is in the **information coefficient**,
not in the costs - break-even stays between 58 and 130 bps at every size. So
this is not a cost artefact: ASX momentum is genuinely weaker among large caps,
which matches the equity momentum literature. It is still the finding that
should temper enthusiasm, because the names carrying the edge are the ones that
are hardest to borrow, widest to trade and most likely to delist. The honest
summary is that the strategy is real at the ASX 200-300 level and marginal at
the ASX 100 level.

## Research agenda

The baseline run covers the four signals below. The brief's remaining threads
are listed after them, with the function that supports each.

Baseline, in `scripts/03_run_baseline.py`:

| Signal | Idea |
|---|---|
| `mom_12_1` | risk-adjusted 12-month momentum, skipping the last month |
| `rev_5d` | five-day cross-sectional reversal |
| `rev_5d_quiet` | reversal tilted towards low-turnover, uninformed flow |
| `rev_resid_5d` | reversal on factor-neutralised returns |

Not yet run, and the natural next steps:

- **Horizon sweep.** `backtest.information_coefficient` across lookbacks from
  1 to 252 days, to locate the crossover where reversal becomes momentum on the
  ASX. The two points measured so far put it somewhere between 5 and 63 days.
- **Why momentum is weak in large caps.** The size decay above is the most
  interesting open question in the results. Is it index-arbitrage flow, analyst
  coverage, or simply that the ASX 100 is twelve stocks and a lot of noise?
- **Reporting season.** Condition momentum on `financial_year_flags`
  `reporting_season`; the new-information logic predicts momentum is stronger
  in February and August.
- **Tax-loss effect.** Test whether the May-June losers bounce in July, using
  `tax_loss_window` and `tax_loss_reversal`.
- **Macro regimes.** Split reversal performance by the terciles of
  `signals.market_state` realised volatility, dispersion and pairwise
  correlation, which is the brief's macro section directly.
- **Theme momentum.** `signals.factor_momentum` on the statistical factors, to
  test whether ASX themes trend even when individual names do not.

## Known limitations

- Yahoo's ASX history has occasional unadjusted split bars. `clean.py` discards
  single-day moves beyond 80%, which removes most but not all of them.
- The point-in-time universe is liquidity-ranked, not index membership. It
  correlates highly with the real ASX 300 but is not identical, so results are
  not directly comparable to published ASX 300 factor studies.
- **The sample is survivor-only.** See above. Every number this repository
  produces is a bound, not an estimate, until the data source changes.
- No borrow cost or short availability constraint. Small ASX names are often
  hard to borrow, so the short leg of any book weighted towards them is
  optimistic. The `--compare-sizes 300,100` run shows how much of the result
  depends on names that could actually be shorted.
- Daily closes only. Anything the brief says about intraday execution or
  front-running scheduled institutional flow needs a minute-bar feed.
