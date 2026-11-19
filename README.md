# Saturating logit-curve trophy rarity forecast

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Lagabara77/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Lagabara77/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one — a
[make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty entry.

## The model

`main.py` (Python 3.13, standard library only) reads one case
(`makefun/gta6-trophies-case/v1`) — every Exophase read of the game's trophy
page before the checkpoint — and forecasts each trophy's earn rate at the
target as five quantiles (5%, 25%, 50%, 75%, 95%), scored by pinball loss on
the logit scale.

Per trophy, on the logit scale:

- **Smoothed rates.** Each read's rate is smoothed as
  `(E + 0.5) / (N + 1)`, where `E` is the implied earners count and `N` the
  players tracked; only reads whose `(rate, N)` pair changes count as
  distinct points.
- **Median ensemble.** A weighted median of four members — persistence
  (0.10), a 24-hour log-time linear fit (0.25), a weighted quadratic fit
  (0.30; refit linear if the `x²` coefficient is positive), and a
  saturating-exponential grid fit over `k` (0.35) — each clamped to
  `[logit(last) − 1.5, logit(last) + 3.0]`. With fewer than 3 distinct
  points the median is persistence alone.
- **Refresh-due shift.** Exophase refreshes a game's counts on its own
  schedule, not continuously: launch night sat unchanged for hours, then
  refreshed with every rate but the 100% one lower. When the page has gone
  stale relative to its own refresh cadence, the median logit is pulled down
  by `−0.7 × stale_factor × exp(−horizon_days / 0.5)`.
- **Spread.** `sigma` combines the quadratic fit's residual, the
  extrapolation distance in log time and the horizon, floored at 0.9 (1.2
  with fewer than 3 distinct points) and capped at 2.2, with a wider lower
  tail when a refresh is due.

On the practice cases (which straddle launch night's first refresh) this
scores a pinball loss of ~0.53 against the inferred answers, versus ~1.50
for the house persistence baseline.

The program is fully deterministic: same input, same bytes out, every run.

## Run it offline

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `<id>.in` in the input directory (in name order) and writes
`<id>.out` for each to the output directory. Outside the grader,
`MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` stand in for `/input` / `/output`.

## License

MIT — see [LICENSE](LICENSE).
