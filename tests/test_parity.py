"""Parity with the reference implementation the weights were trained with.

`tests/data/parity_reference.npz` holds fixed input series and the quantiles the reference
implementation returned for them (float32, CPU, one thread, the torch version stored in the file).
`forecast()` must reproduce them to 1e-6. This holds with that same torch version on one thread; other
torch versions agree to a few 1e-6 of the series' standard deviation (rounding differences), which is
reported but not required.

The weights are read from `NOM_WEIGHTS` (a local folder or a Hugging Face repo id), or from
`./weights` when present. Without weights the test is skipped.
"""
import json
import os

import numpy as np
import pytest
import torch

from nom import NomForecaster
from nom.forecaster import Forecast

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
    out = forecaster.forecast(context, case["prediction_length"], flip=case["flip"])
    expected = ref[f"{case['name']}/quantiles"]
    scale = np.array([np.nanstd(c) for c in context])[:, None, None]
    gap = float((np.abs(out.quantiles - expected) / scale).max())
    np.testing.assert_allclose(
        out.quantiles, expected, rtol=1e-6, atol=1e-6,
        err_msg=f"torch {torch.__version__} vs reference torch {ref['torch_version']}; "
                f"max gap {gap:.2e} context standard deviations")


def test_config_matches_reference_config(forecaster):
    with open(os.path.join(HERE, "data", "config.json")) as f:
        assert json.load(f)["quantile_levels"] == list(forecaster.cfg.quantile_levels)


def test_reference_quantiles_never_cross():
    ref = np.load(REFERENCE)
    for case in json.loads(str(ref["cases"])):
        q = ref[f"{case['name']}/quantiles"]
        assert np.isfinite(q).all() and (np.diff(q, axis=-1) >= 0).all(), case["name"]


@pytest.mark.parametrize("case, ref", _cases(), ids=lambda x: x["name"] if isinstance(x, dict) else "")
def test_forecast_never_crosses_on_reference_inputs(forecaster, case, ref):
    context = [ref[f"{case['name']}/context_{j}"] for j in range(case["n_series"])]
    q = forecaster.forecast(context, case["prediction_length"], flip=case["flip"]).quantiles
    assert (np.diff(q, axis=-1) >= 0).all(), case["name"]


def test_reference_quantiles_with_tails_never_cross():
    """The 13 levels (9 predicted + 4 extrapolated) stay ordered on every reference case."""
    ref = np.load(REFERENCE)
    with open(os.path.join(HERE, "data", "config.json")) as f:
        levels = tuple(json.load(f)["quantile_levels"])
    for case in json.loads(str(ref["cases"])):
        out = Forecast(ref[f"{case['name']}/quantiles"], levels)
        all13 = np.stack([out.quantile(q) for q in out.all_levels], -1)
        assert len(out.all_levels) == 13 and (np.diff(all13, axis=-1) >= 0).all(), case["name"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU (CUDA or ROCm)")
def test_gpu_agrees_with_reference():
    """On a GPU (float32) the reference is not bit-exact; measured on ROCm 7.2 (RX 7900 XTX): 2.7e-5 standard deviations
    of the input series at most, no crossing."""
    src = _weights()
    if src is None:
        pytest.skip("no weights: set NOM_WEIGHTS to a folder or a Hugging Face repo id")
    gpu = NomForecaster.from_pretrained(src, device="cuda")
    ref = np.load(REFERENCE)
    for case in json.loads(str(ref["cases"])):
        context = [ref[f"{case['name']}/context_{j}"] for j in range(case["n_series"])]
        q = gpu.forecast(context, case["prediction_length"], flip=case["flip"]).quantiles
        scale = np.array([np.nanstd(c) for c in context])[:, None, None]
        assert float((np.abs(q - ref[f"{case['name']}/quantiles"]) / scale).max()) < 1e-4, case["name"]
        assert (np.diff(q, axis=-1) >= 0).all(), case["name"]
