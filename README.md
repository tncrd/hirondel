# nom

A small univariate time-series foundation model for zero-shot probabilistic forecasting: 7.5M parameters,
9 quantiles from 0.1 to 0.9, any horizon, CPU-friendly.

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

## Hardware

Tested on CPU (Linux, PyTorch 2.12.1 and 2.4.1). GPUs (ROCm, CUDA) are expected to work through the `device` argument
but have not been tested with this package yet.

## Numerical parity

`tests/test_parity.py` checks `forecast()` against outputs of the reference implementation on fixed
inputs (CPU, float32, one thread). With torch 2.12.1 every case matches bit for bit and the test is
strict (1e-6). With other torch versions small rounding differences remain; measured with torch
2.4.1 (the oldest supported version), the largest gap over all cases is 1.5e-5 standard deviations
of the input series.

## License and data

Code and weights: Apache-2.0 (`LICENSE`). The training data and the attribution each source requires are listed in
`DATA.md`; `NOTICE` must be kept when redistributing.

## Citation

See `CITATION.cff`.

## Contributing

See `CONTRIBUTING.md`.
