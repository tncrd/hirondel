# Benchmarks

Thin scripts that pass the released package through the official benchmark harnesses. Results are written in each
benchmark's own submission format under `benchmarks/results/`.

## The adapter

Every model, ours or another, enters a benchmark through one function:

```python
forecast_fn(contexts: list[np.ndarray], prediction_length: int) -> np.ndarray  # [n_series, prediction_length, 9]
```

returning the quantiles 0.1, ..., 0.9 on the raw scale (contexts are 1-D float arrays, NaN = missing).
`benchmarks.adapter.GluonTSAdapter(forecast_fn, prediction_length)` turns it into a GluonTS predictor, checks shapes and
finiteness, and emits `QuantileForecast` objects. `hirondel_forecast_fn(forecaster)` is the released model with its default
inference procedure (flip on, zero clipping of non-negative series on).

## GIFT-Eval

```bash
git clone https://github.com/SalesforceAIResearch/gift-eval
pip install ".[bench]" && pip install --no-deps ./gift-eval
export GIFT_EVAL=<folder of the Salesforce/GiftEval dataset>
python -m benchmarks.gift --gift-eval gift-eval --out benchmarks/results/gift --model <folder or repo id> --device cuda
```

97 tasks (55 short, 21 medium, 21 long), the metrics, dataset names and columns of the official notebooks. The headline
is the leaderboard's: geometric mean over the 97 tasks of MASE[0.5] and CRPS (`mean_weighted_sum_quantile_loss`), each
divided by the Seasonal Naive result in `gift-eval/results/seasonal_naive/all_results.csv`.

gift-eval declares `gluonts~=0.15.1` and `numpy~=1.26`; it is installed here without its dependencies, next to
gluonts 0.16.3 and numpy 2 (needed for recent Python versions); the numbers in `results/` were computed that way.

## fev-bench

```bash
git clone https://github.com/autogluon/fev && git -C fev checkout ceb25ae
pip install ".[bench]" "fev==0.9.0"
python -m benchmarks.fev_bench --fev-bench fev/benchmarks/fev_bench --out benchmarks/results/fev --device cuda
```

All 100 tasks, univariate and covariate-blind (every target column forecast as its own series). Skill score, win rate
and rank are computed as on the fev-bench site (`fev.analysis.leaderboard` with the models it shows by default, from
`results/` and `models.yaml` of the fev repository). No task is declared as trained on: our training data contains none
of the fev-bench datasets (`DATA.md`).

## TIME

`timebench` pins older numpy / datasets / gluonts, so TIME's loader and scorer run in their own environment and the
model in this package's; they meet through files (`python -m benchmarks.time_bench --help`). The headline is the one of
TIME's `scripts/compute_local_leaderboard.py` (per task, mean over windows and series divided by Seasonal Naive's;
geometric mean over the 98 tasks), written unrounded to `results/time/headline.json`, with the per-task values in
`results/time/tasks.csv`.

## Series with no observed value

Some test contexts have no observed value at all, while their future values are scored:

| benchmark | contexts | datasets |
|---|---|---|
| GIFT-Eval | 649 | bitbrains_fast_storage and bitbrains_rnd (5T and H; short, medium, long), electricity/W |
| fev-bench | 230 of 1 980 | gvar |
| TIME | 0 | |

`HirondelForecaster.forecast()` refuses such a series with a clear error: a forecast from nothing is not something a user
should get silently. A benchmark still has to score it, so `hirondel_forecast_fn` runs the model on the empty context
through the same internal path as the reference implementation (its output is finite but carries no information).
Among the competitors, only TimesFM-2.5's package refuses an empty context; it then receives a single observed 0
(`benchmarks.competitors.no_empty`). The other packages receive the empty context as it is.

## Competitors

`competitors.yaml` lists the models we run ourselves, in forced univariate mode, with the inference settings of their
official GIFT-Eval notebooks or model cards. The list, sizes and revisions were committed before any of them was scored.
Each model runs in its own environment with its public package:

```bash
python -m benchmarks.competitors gift --name Chronos-2 --gift-eval gift-eval --device cuda
```

Results: `results/gift_competitors/<model>/` (`all_results.csv`, and `headline.json` with the package version and the
date). These are "same protocol, computed by us" numbers, not the leaderboards' official ones.

## Zero clipping (`clip_nonnegative`)

The released default clamps the quantiles of non-negative series at 0, after the flip average. It was measured before
being adopted, with `benchmarks/experiments/clip_zero.py` (variant `after_flip`, which the package default reproduces
exactly: `tests/test_clip_zero.py`). The results in `results/{gift,fev,time}` are those of the default; the same model
without clipping is kept in `results/experiments/clip_zero/off/`.

| | GIFT-Eval MASE / CRPS | fev-bench skill MASE / SQL (%) | TIME MASE / CRPS |
|---|---|---|---|
| without clipping | 0.7011 / 0.4825 | 32.80 / 44.80 | 0.6555 / 0.5527 |
| with clipping (default) | 0.7003 / 0.4816 | 32.91 / 44.90 | 0.6551 / 0.5520 |

The gain is small but goes the same way on every benchmark and metric. The variant `per_pass` (clipping the direct pass
before the average) gains less on GIFT-Eval (0.7008 / 0.4821) and was not kept.
