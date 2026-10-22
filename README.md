<p align="center"><img src="showcase/figures/banner.png" alt="Nom" width="100%"></p>

# Nom

A small foundation model for zero-shot probabilistic time-series forecasting: **7.5M parameters**, 9 quantiles from 0.1
to 0.9 (tails to 0.01 and 0.99 by extrapolation), any horizon, CPU-friendly, **Apache-2.0** code and weights.

**The most accurate forecasting model under 10M parameters with public weights and a permissive license, on GIFT-Eval, fev-bench and TIME** (official leaderboards, read on 2026-10-22; how this was checked: `showcase/README.md`).

## Results

Headline numbers of this package (float32), computed with each benchmark's official harness
(`benchmarks/`, per-task details in `benchmarks/results/`):

<!-- results:start -->
| Benchmark | MASE | CRPS (fev-bench: SQL) |
|---|---|---|
| GIFT-Eval (97 tasks, normalised by Seasonal Naive; lower is better) | 0.7011 | 0.4825 |
| fev-bench (100 tasks, skill score in %; higher is better) | 32.80 | 44.80 |
| TIME (98 tasks, normalised by Seasonal Naive; lower is better) | 0.6555 | 0.5527 |
<!-- results:end -->

On the fev-bench leaderboard this ranks 7 of 33 (SQL) and 6 of 33 (MASE) among the models the site shows by default (2026-10-22).

<p align="center"><img src="showcase/figures/pareto.png" alt="Accuracy against size on GIFT-Eval, TIME and fev-bench" width="100%"></p>

Official leaderboard results of well-known models, read on 2026-10-22, against their size. The Pareto fronts are drawn
over models with a permissive license; models under a non-commercial or restricted license are shown hollow. Every
value, with its source, is in `showcase/data/official.csv`. Some of these models use covariates or forecast the
variables of a series jointly on the leaderboards; this model forecasts each series alone.

### Same protocol, computed by us

The same comparison with every model in the same setting: each series forecast alone, no covariates, public packages,
GIFT-Eval (97 tasks). Settings and the list (written before any score) are in `benchmarks/competitors.yaml`.

| Model | Parameters | License | GIFT-Eval MASE | GIFT-Eval CRPS |
|---|---|---|---|---|
| Timestral-28M | 27.8M | Modified MIT (revenue cap) | 0.6817 | 0.4641 |
| **Nom (ours)** | 7.5M | Apache-2.0 | 0.7011 | 0.4825 |
| Chronos-2 | 119.5M | Apache-2.0 | 0.7037 | 0.4911 |
| Timestral-5M | 5.2M | Modified MIT (revenue cap) | 0.7102 | 0.4836 |
| TiRex | 35.0M | NX-AI Community License | 0.7213 | 0.4921 |
| Toto-2.0-22m | 21.9M | Apache-2.0 | 0.7226 | 0.5009 |

<p align="center"><img src="showcase/figures/same_protocol.png" alt="Same protocol on GIFT-Eval" width="85%"></p>

## Install

From a clone of this repository (Python 3.10 or newer, PyTorch 2.4 or newer):

```bash
pip install .            # core: torch, numpy, safetensors, huggingface_hub
pip install ".[pandas]"  # adds forecast_df for pandas DataFrames
```

## Quick start

```python
import numpy as np
from nom import NomForecaster

forecaster = NomForecaster.from_pretrained("account/Nom-7M")  # or a local folder holding config.json and model.safetensors

t = np.arange(1000)
context = np.sin(2 * np.pi * t / 24) + 0.1 * np.random.default_rng(0).normal(size=t.size)

forecast = forecaster.forecast(context, prediction_length=48)
print(forecast.quantiles.shape)       # (1, 48, 9): series, steps, quantile levels 0.1 ... 0.9
print(forecast.median[0, :4])         # point forecast
print(forecast.quantile(0.9)[0, :4])  # upper 90% quantile
```

`forecast()` takes one series (1-D array), a batch of equal-length series (2-D array) or a list of series of any
lengths; NaN values are treated as missing. By default (`flip=True`) each forecast averages the model run on the series
and on its negation; `flip=False` runs one pass instead of two. `forecast_df()` does the same for a long pandas
DataFrame (`item_id`, `timestamp`, `target`) and returns one row per future timestamp.

### Tail levels 0.01, 0.05, 0.95, 0.99

`forecast.quantile(q)` also accepts 0.01, 0.05, 0.95 and 0.99 (and `forecast_df(..., tail_quantiles=True)` adds their
columns). **These levels are extrapolated, not learned**: the network predicts the 9 levels 0.1 ... 0.9; each tail is
then extended by an exponential fitted on its two outermost levels (the `ExponentialTailApproximation` of GluonTS),
after the flip average. No extra weights; the 9 predicted levels are unchanged.

Measured coverage P(y ≤ q), this model with `flip=True`:

| data | 0.01 | 0.05 | 0.95 | 0.99 |
|---|---|---|---|---|
| fresh 2026 series (424 windows, H = 96) | 0.009 | 0.046 | 0.950 | 0.985 |
| GIFT-Eval, test windows (H 48-900) | 0.007 | 0.039 | 0.947 | 0.984 |
| TIME (98 tasks) | 0.009 | 0.046 | 0.947 | 0.988 |
| fev-bench (100 tasks) | 0.012 | 0.044 | 0.938 | 0.979 |

Source: our measurement on the released weights, averaged over windows (fresh series, GIFT-Eval) or over tasks (TIME,
fev-bench). The upper tail runs slightly short, most on fev-bench (spiky and intermittent series).

## Hardware

Tested on CPU (Linux, PyTorch 2.12.1 and 2.4.1) and on an AMD GPU with ROCm 7.2 (RX 7900 XTX, PyTorch 2.12.1,
`device="cuda"`). CUDA GPUs are expected to work the same way but have not been tested.

## Numerical parity

`tests/test_parity.py` checks `forecast()` against outputs of the reference implementation on fixed
inputs (CPU, float32, one thread). With torch 2.12.1 every case matches bit for bit and the test is
strict (1e-6). With other torch versions small rounding differences remain; measured with torch
2.4.1 (the oldest supported version), the largest gap over all cases is 1.5e-5 standard deviations
of the input series; on the ROCm GPU above (float32), 2.7e-5.

## Limitations

- Univariate: each series is forecast on its own; covariates and cross-series information are not used. Models that use
  them (for example Chronos-2 on GIFT-Eval, Timestral with covariates on fev-bench) can do better on such tasks.
- Context: the last 2 048 points at full resolution, plus a coarse reading of up to 8 192 points. Very long seasonal
  histories are read coarsely.
- Tail levels 0.01, 0.05, 0.95, 0.99 are extrapolated, not learned; the upper tail runs slightly short.
- Tested on CPU and on one AMD GPU (ROCm); not yet on CUDA.

## Reproducing the numbers

`benchmarks/README.md` gives the commands for GIFT-Eval, fev-bench, TIME and the competitors, and `showcase/README.md`
how this page and its figures are regenerated (`python -m showcase.build`).

## License and data

Code and weights: Apache-2.0 (`LICENSE`). The training data and the attribution each source requires are listed in
`DATA.md`; `NOTICE` must be kept when redistributing.

## Citation

See `CITATION.cff`.

## Contributing

See `CONTRIBUTING.md`.
