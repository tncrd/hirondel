---
license: apache-2.0
library_name: nom
pipeline_tag: time-series-forecasting
tags:
  - time-series
  - forecasting
  - foundation-model
  - zero-shot
  - probabilistic-forecasting
model-index:
  - name: Nom
    results:
      - task: {type: time-series-forecasting, name: Probabilistic forecasting}
        dataset: {type: Salesforce/GiftEval, name: GIFT-Eval (97 tasks)}
        metrics:
          - {type: mase, name: MASE (normalised by Seasonal Naive), value: 0.7011}
          - {type: crps, name: CRPS (normalised by Seasonal Naive), value: 0.4825}
      - task: {type: time-series-forecasting, name: Probabilistic forecasting}
        dataset: {type: autogluon/fev_datasets, name: fev-bench (100 tasks)}
        metrics:
          - {type: skill-score, name: Skill score, MASE (%), value: 32.80}
          - {type: skill-score, name: Skill score, SQL (%), value: 44.80}
      - task: {type: time-series-forecasting, name: Probabilistic forecasting}
        dataset: {type: Real-TSF/TIME, name: TIME (98 tasks)}
        metrics:
          - {type: mase, name: MASE (normalised by Seasonal Naive), value: 0.6555}
          - {type: crps, name: CRPS (normalised by Seasonal Naive), value: 0.5527}
---

![Nom](showcase/figures/banner.png)

# Nom

A small foundation model for zero-shot probabilistic time-series forecasting: 7.5M parameters, 9 quantiles from 0.1 to
0.9, any horizon, CPU-friendly. Code and weights under Apache-2.0.

**The most accurate forecasting model under 10M parameters with public weights and a permissive license, on GIFT-Eval, fev-bench and TIME** (official leaderboards, read on 2026-10-22; how this was checked: `showcase/README.md`).

## Results

<!-- results:start -->
| Benchmark | MASE | CRPS (fev-bench: SQL) |
|---|---|---|
| GIFT-Eval (97 tasks, normalised by Seasonal Naive; lower is better) | 0.7011 | 0.4825 |
| fev-bench (100 tasks, skill score in %; higher is better) | 32.80 | 44.80 |
| TIME (98 tasks, normalised by Seasonal Naive; lower is better) | 0.6555 | 0.5527 |
<!-- results:end -->

On the fev-bench leaderboard this ranks 7 of 33 (SQL) and 6 of 33 (MASE) among the models the site shows by default (2026-10-22).

![Accuracy against size, univariate setting](showcase/figures/univariate.png)

Univariate setting: every model forecasts each series alone, without covariates (competitors run by us in forced
univariate mode with their public packages; TiRex and TimesFM-2.5 univariate by design). Pareto fronts over models with
a permissive license, the others shown hollow.

| Model | Parameters | License | GIFT-Eval MASE | GIFT-Eval CRPS | TIME MASE | TIME CRPS | fev-bench SQL skill |
|---|---|---|---|---|---|---|---|
| Timestral-5M | 5.2M | Modified MIT (revenue cap) | 0.7102 | 0.4836 | 0.6571 | 0.5490 | 44.36 |
| **Nom (ours)** | 7.5M | Apache-2.0 | 0.7011 | 0.4825 | 0.6555 | 0.5527 | 44.80 |
| Toto-2.0-22m | 21.9M | Apache-2.0 | 0.7226 | 0.5009 | 0.6751 | 0.5654 | 42.37 |
| Timestral-28M | 27.8M | Modified MIT (revenue cap) | 0.6817 | 0.4641 |  |  | 45.86 |
| TiRex | 35.0M | NX-AI Community License | 0.7213 | 0.4921 | 0.6831 | 0.5731 | 42.58 |
| Chronos-2 | 119.5M | Apache-2.0 | 0.7037 | 0.4911 | 0.6659 | 0.5584 | 43.97 |
| TimesFM-2.5 | 231.3M | Apache-2.0 | 0.7050 | 0.4903 | 0.6686 | 0.5674 | 42.20 |

![Accuracy against size, official leaderboards](showcase/figures/pareto.png)

Official leaderboards, read on 2026-10-22 (some models use covariates or joint multivariate forecasting there).
Sources: `showcase/data/` in the code repository.

## Usage

```python
import numpy as np
from nom import NomForecaster

forecaster = NomForecaster.from_pretrained("account/Nom-7M")
t = np.arange(1000)
context = np.sin(2 * np.pi * t / 24) + 0.1 * np.random.default_rng(0).normal(size=t.size)
forecast = forecaster.forecast(context, prediction_length=48)
print(forecast.quantiles.shape)   # (1, 48, 9)
```

Install, inference details, tail levels, hardware and numerical parity: see the code repository's README.

## Training data

Real data under licenses that allow this release, plus synthetic data. Every source, its license, its attribution and
the changes made are listed in `DATA.md`; `NOTICE` gathers the required attributions. No dataset of GIFT-Eval's test
split, fev-bench or TIME is in the training data.

## Limitations

- Univariate: no covariates, no cross-series information.
- Context: 2 048 points at full resolution plus a coarse reading of up to 8 192 points.
- Tail levels 0.01 / 0.05 / 0.95 / 0.99 are extrapolated, not learned.

## License

Apache-2.0 for code and weights. `NOTICE` must be kept when redistributing.
