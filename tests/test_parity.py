"""Parity with the reference implementation the weights were trained with.

`tests/data/parity_reference.npz` holds fixed input series and the quantiles the reference
implementation returned for them (float32, CPU, one thread). `forecast()` must reproduce them to 1e-6.

The weights are read from `NOM_WEIGHTS` (a local folder or a Hugging Face repo id), or from
`./weights` when present. Without weights the test is skipped.
"""
import json
import os

import numpy as np
import pytest
import torch

from nom import NomForecaster

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "data", "parity_reference.npz")


def _weights() -> str | None:
    if os.environ.get("NOM_WEIGHTS"):
        return os.environ["NOM_WEIGHTS"]
    local = os.path.join(HERE, "..", "weights")
    return local if os.path.isfile(os.path.join(local, "model.safetensors")) else None


@pytest.fixture(scope="module")
def forecaster():
    src = _weights()
    if src is None:
        pytest.skip("no weights: set NOM_WEIGHTS to a folder or a Hugging Face repo id")
    torch.set_num_threads(1)                     # the reference was computed on one thread
    return NomForecaster.from_pretrained(src)


def _cases():
    ref = np.load(REFERENCE)
    return [(c, ref) for c in json.loads(str(ref["cases"]))]


@pytest.mark.parametrize("case, ref", _cases(), ids=lambda x: x["name"] if isinstance(x, dict) else "")
def test_forecast_matches_reference(forecaster, case, ref):
    context = [ref[f"{case['name']}/context_{j}"] for j in range(case["n_series"])]
    out = forecaster.forecast(context, case["prediction_length"], freq=case["freq"], flip=case["flip"])
    np.testing.assert_allclose(out.quantiles, ref[f"{case['name']}/quantiles"], rtol=1e-6, atol=1e-6)


def test_config_matches_reference_config(forecaster):
    with open(os.path.join(HERE, "data", "config.json")) as f:
        assert json.load(f)["quantile_levels"] == list(forecaster.cfg.quantile_levels)
