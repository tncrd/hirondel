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
finiteness, and emits `QuantileForecast` objects. `nom_forecast_fn(forecaster)` is the released model with its default
inference procedure (flip on).

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
