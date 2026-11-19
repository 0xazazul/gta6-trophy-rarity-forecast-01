# GTA VI trophy-rarity forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0xazazul/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0xazazul/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
the case target (2026-11-26T00:00Z for the final), as five quantiles
(5/25/50/75/95%) per trophy, for make.fun bounty
`gta6-trophy-rarity-forecast-01`. Scored by pinball loss on the logit scale.

## What it does

Exophase does not update continuously: between its refreshes the trophy
page is frozen, and at a refresh the tracked-player count jumps and every
rate moves toward the earn rate among the newly tracked players. So the
only informative history is the sequence of refreshes.

`main.py` (Python 3.13, standard library only, no network, deterministic):

1. **Detects refreshes** in the case history (tracked-count jumps or rate
   changes between consecutive reads).
2. **Recovers each trophy's new-player earn fraction** `f` from the counts
   at each refresh — `f = (r2·N2 − r1·N1) / (N2 − N1)` — pooled across
   refreshes with recency weights.
3. **Projects the dilution forward**: over the expected number of further
   refreshes before the target (horizon / observed refresh cadence) the
   tracked base grows by the median per-refresh influx, and the median rate
   is the player-weighted average of the current rate and `f`. As the base
   grows each later refresh dilutes less, so the rate asymptotes to `f`.
4. **Spreads honestly**: the quantiles come from a mixture over the number
   of refreshes before the target (Poisson counts), with a random-walk term,
   uncertainty in `f`, and per-refresh growth uncertainty, all on the logit
   scale. With no refresh in the history (launch night) the median stays at
   the last rate and the spread carries the refresh risk — one refresh can
   dilute a non-universal trophy several-fold, while the trophy held by
   ~everyone stays ~100%.

Nothing is fitted to any practice answer (none are in the kit) and no
practice date, rate or value is hardcoded.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<dir with *.in> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input dir and writes `<id>.out` beside it in
the output dir: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Local measurements (see the build report for details)

- Valid on both practice cases (55 rows, non-decreasing, 0–100, ≤65,536 B),
  byte-identical across two runs, ~0.04 s per case (limit 8 s).
- Flat held-out slices (forecast within the frozen launch-night history):
  loss 0.082 vs persistence 0.025 — the honest price of refresh insurance
  when nothing moves.
- Synthetic single-refresh answer (aggregate-implied: non-100% rates
  ×0.2207): loss 0.85 vs persistence 1.40, coverage 0.98 vs 0.00.
- Synthetic multi-refresh final shape (3 refreshes in history, 6 more
  before a 4.5-day target): loss 0.25 vs persistence 0.31; the median
  tracks the dilution toward the new-player fraction (e.g. 0.08 → 0.044
  against a true 0.04).
- The official practice loss cannot be computed locally — the answers are
  not in the kit; the site scores them on submit.

---

<sub>Entered on [make.fun](https://make.fun/?ref=jvhn9fta&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards · Bounty: https://make.fun/bounty/gta6-trophy-rarity-forecast-01</sub>
