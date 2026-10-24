---
license: apache-2.0
library_name: hirondel
pipeline_tag: time-series-forecasting
tags:
  - time-series
  - forecasting
  - foundation-model
  - zero-shot
  - probabilistic-forecasting
model-index:
  - name: Hirondel-7M
    results:
      - task: {type: time-series-forecasting, name: Probabilistic forecasting}
        dataset: {type: Salesforce/GiftEval, name: GIFT-Eval (97 tasks)}
        metrics:
          - {type: mase, name: MASE (normalised by Seasonal Naive), value: <!-- gen:gift_mase -->}
          - {type: crps, name: CRPS (normalised by Seasonal Naive), value: <!-- gen:gift_crps -->}
      - task: {type: time-series-forecasting, name: Probabilistic forecasting}
        dataset: {type: autogluon/fev_datasets, name: fev-bench (100 tasks)}
        metrics:
          - {type: skill-score, name: Skill score, MASE (%), value: <!-- gen:fev_mase -->}
          - {type: skill-score, name: Skill score, SQL (%), value: <!-- gen:fev_sql -->}
      - task: {type: time-series-forecasting, name: Probabilistic forecasting}
        dataset: {type: Real-TSF/TIME, name: TIME (98 tasks)}
        metrics:
          - {type: mase, name: MASE (normalised by Seasonal Naive), value: <!-- gen:time_mase -->}
          - {type: crps, name: CRPS (normalised by Seasonal Naive), value: <!-- gen:time_crps -->}
---

![Hirondel](showcase/figures/banner.png)

# Hirondel-7M: zero-shot probabilistic time-series forecasting

*Hirondel — from French* hirondelle *(swallow), the bird said to tell the weather.*

Hirondel is a 7M-parameter time-series forecasting model for zero-shot probabilistic forecasting: 9 quantiles from 0.1
to 0.9, any horizon, CPU-friendly. Code and weights under Apache-2.0: https://github.com/tncrd/hirondel.

<!-- gen:sentence -->

## Results

<!-- gen:results -->

![Accuracy against size, univariate setting](showcase/figures/univariate.png)

Univariate setting: every model forecasts each series alone, without covariates (competitors run by us in forced
univariate mode with their public packages; TiRex, TimesFM-2.5, FlowState-r1.1 and Moirai-2.0 univariate by design). Pareto fronts over models with
a permissive license, the others shown hollow.

<!-- gen:univariate -->

### Industrial data: IND-Eval

![IND-Eval, accuracy against size](showcase/figures/indeval.png)

Five industrial sources of [IND-Eval](https://huggingface.co/datasets/cognitedata/IND-Eval), none in the training
data (top row), then the nine of the paper's aggregate (bottom row). Hirondel: 0.839 / 0.536 (nMASE / nCRPS) on the
five. With every model forecasting each channel alone (circles, our runs with the same harness), Hirondel is ahead of
Chronos-2, TimesFM-2.5 and TiRex, level with Timestral-5M and Toto-2.0-22m, and behind Timestral-28M. Forecasting the
channels jointly with PRONTO's covariates (crosses) moves Chronos-2 from 0.895 to 0.822. Tables per source, protocol
and model versions: [README, IND-Eval](https://github.com/tncrd/hirondel#industrial-data-ind-eval).

![Accuracy against size, official leaderboards](showcase/figures/pareto.png)

Official leaderboards, read on 2026-10-23 (some models use covariates or joint multivariate forecasting there).
Sources: `showcase/data/` in the code repository.

## Usage

```python
import numpy as np
from hirondel import HirondelForecaster

forecaster = HirondelForecaster.from_pretrained("Tncrd/Hirondel-7M")
t = np.arange(1000)
context = np.sin(2 * np.pi * t / 24) + 0.1 * np.random.default_rng(0).normal(size=t.size)
forecast = forecaster.forecast(context, prediction_length=48)
print(forecast.quantiles.shape)   # (1, 48, 9)
```

Install with `pip install "hirondel @ git+https://github.com/tncrd/hirondel@v1.0.0"` (\* `pip install hirondel`:
PyPI release in preparation). Inference details, tail levels, hardware and numerical parity: see the code
repository's README (https://github.com/tncrd/hirondel).

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

## Citation

```bibtex
@misc{hirondel2026,
  title        = {Hirondel-7M: zero-shot probabilistic time-series forecasting},
  author       = {{The Hirondel Author (Tncrd)}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/Tncrd/Hirondel-7M}}
}
```
