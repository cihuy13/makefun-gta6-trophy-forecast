# GTA VI trophy rarity forecast

Forecasts every GTA VI (PS5) trophy's earn rate on Exophase at a
case's target time, as five quantiles per trophy.

## Model: counts + dilution + saturating growth

1. **Counts, not rates.** Each read's rate is converted to an earner
   count (`earners = rate / 100 * players_tracked`). The player base
   grows and dilutes rates, so a rate can fall while its earners
   still grow; fitting rates directly would miss that.
2. **Saturating fits in log-time.** Per trophy, the earner counts are
   fitted against `x = ln(1 + hours since 2026-11-19T00:00Z)` with
   two candidate shapes — a saturating exponential `A * (1 - exp(-k * x))`
   (least squares over a log-spaced `k` grid) and an exponential in
   log-time `exp(a + b * x)` — and the better fit wins. The same fit
   is applied to `players_tracked`. Both are projected to the target;
   the median rate is projected earners / projected players. Earners
   and players are cumulative, so projections never fall below the
   last observed value.
3. **Fallbacks.** A flat history (or fewer than 2 usable reads) falls
   back to persistence (the last rate). A trophy at 100% stays near
   100%. Missing rates are skipped; a trophy never listed gets a
   tier prior with a wide spread.
4. **Quantiles.** Logit-normal around the median. The spread combines
   the fit's residual scatter in logit space, a horizon term growing
   with `sqrt(horizon in days)`, a small-count term (rare trophies
   are more volatile), and an extra term when the page never moved
   at all (a refresh is then overdue). Quantiles are clamped to
   0.01–99.99 and non-decreasing.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<dir with *.in> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` file in `$MAKEFUN_INPUT_DIR` (else `/input/`)
and writes `/output/<id>.out` (`$MAKEFUN_OUTPUT_DIR` else `/output/`)
per case: one JSON object `{"trophies": [{"id": n, "q": [q05, q25,
q50, q75, q95]}]}`. Standard library only, no network, no
randomness — two runs on the same case are byte-identical.
