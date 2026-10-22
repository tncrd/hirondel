"""Other models through the same adapter as ours: one `forecast_fn(contexts, prediction_length) -> [n, H, 9]` per model
of `competitors.yaml`, in forced univariate mode (each series alone, no covariates), with the inference settings of the
model's official GIFT-Eval notebook or model card. Each model runs in its own environment with its public package:

    pip install -e . "gluonts==0.16.3" pandas pyyaml <package of the model>
    python -m benchmarks.competitors gift --name Chronos-2 --gift-eval gift-eval --device cuda

Weights are loaded from safetensors where the model ships them; TiRex ships a `.ckpt` that its package loads with
`torch.load(weights_only=True)` (tensors only, no code).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
HERE = os.path.dirname(os.path.abspath(__file__))


def load_list(path: str = os.path.join(HERE, "competitors.yaml")) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def entry(lst: dict, name: str, benchmark: str) -> dict:
    """The model's entry, with the benchmark-specific overrides of the list applied (e.g. a leakage-free checkpoint)."""
    base = next(m for m in lst["competitors"] if m["name"] == name)
    return {**base, **(lst.get(f"{benchmark}_eval_overrides") or lst.get(f"{benchmark}_overrides") or {}).get(name, {})}


def left_pad(contexts: list) -> tuple[np.ndarray, np.ndarray]:
    """Ragged series -> [n, longest] left-padded with NaN, and the observed mask (False on padding and NaN)."""
    T = max(len(c) for c in contexts)
    x = np.full((len(contexts), T), np.nan, np.float32)
    for i, c in enumerate(contexts):
        x[i, T - len(c):] = c
    return x, np.isfinite(x)


def level_indices(levels) -> list[int]:
    """Positions of 0.1, ..., 0.9 in a model's own list of quantile levels."""
    out = []
    for q in LEVELS:
        hit = [i for i, x in enumerate(levels) if abs(float(x) - q) < 1e-9]
        if not hit:
            raise ValueError(f"level {q} is not among the model's levels {list(levels)}")
        out.append(hit[0])
    return out


def out_dir(benchmark: str, line: str) -> str:
    return os.path.join(HERE, "results", f"{benchmark}_competitors", line)


def no_empty(contexts: list) -> list:
    """A context with no observed value (649 in GIFT-Eval: bitbrains, electricity/W) becomes a single observed 0, for the
    packages that refuse such a context (TimesFM-2.5). The other packages receive it as it is."""
    return [c if np.isfinite(c).any() else np.zeros(1, np.float32) for c in contexts]


def _by_length(contexts: list, batch: int):
    """Batches of similar lengths (less padding); yields (indices, contexts)."""
    order = np.argsort([len(c) for c in contexts], kind="stable")
    for lo in range(0, len(order), batch):
        idx = order[lo:lo + batch]
        yield idx, [contexts[i] for i in idx]


# ---- one loader per model --------------------------------------------------------------------------------------------
def timestral(m: dict, device: str):
    """Timestral forecaster defaults (context up to 32 768, symmetric averaging, non-negative clipping), one variate."""
    import torch
    from timestral import TimestralForecaster
    fc = TimestralForecaster.from_pretrained(m["repo_id"], device=device, revision=m["revision"])
    idx = level_indices(fc.quantile_levels)

    def fn(contexts, H):
        out = np.empty((len(contexts), H, 9), np.float32)
        for rows, chunk in _by_length(contexts, 64):
            x, _ = left_pad(chunk)
            with torch.no_grad():
                q = fc.forecast(x[:, None, :], H).quantiles[:, 0][..., idx]
            out[rows] = q.float().cpu().numpy()
        return out
    return fn


def chronos2(m: dict, device: str):
    """Chronos-2 pipeline, `predict_quantiles` on univariate inputs, no cross-series learning (as its notebook)."""
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(m["repo_id"], revision=m["revision"], device_map=device)

    def fn(contexts, H):
        q, _ = pipe.predict_quantiles([{"target": np.asarray(c, np.float32)} for c in contexts],
                                      prediction_length=H, quantile_levels=list(LEVELS), batch_size=256)
        return np.stack([t[0].float().cpu().numpy() for t in q])
    return fn


def tirex(m: dict, device: str, variant: str | None = None):
    """TiRex `forecast` as on its model card, torch backend (the custom-kernel backend needs CUDA). Variant "resample":
    `resample_strategy="frequency"` as in TiRex's GIFT-Eval notebook (the context is downsampled so that one dominant
    period spans about 64 points, the forecast is made at that rate and brought back)."""
    import torch
    from tirex import load_model
    model = load_model(m["repo_id"], device=device, backend="torch", hf_kwargs={"revision": m["revision"]})

    def fn(contexts, H):
        kw = {"resample_strategy": "frequency"} if variant == "resample" else {}
        q, _ = model.forecast(context=[torch.tensor(c) for c in contexts], prediction_length=H, batch_size=256, **kw)
        return np.asarray(q.float().cpu().numpy())[:, :H]
    return fn


def timesfm25(m: dict, device: str):
    """TimesFM 2.5 with the ForecastConfig of its GIFT-Eval notebook (context up to 15 360, flip invariance, ...), except
    per_core_batch_size 32 instead of 128: at a 15 360-point context, 128 series do not fit in 24 GB. The batch size
    changes memory use only; every series is forecast on its own."""
    import timesfm
    tf = timesfm.TimesFM_2p5_200M_torch.from_pretrained(m["repo_id"], revision=m["revision"])
    p = tf.model.p

    def fn(contexts, H):
        mc = min(15360, (max(len(c) for c in no_empty(contexts)) + p - 1) // p * p)
        tf.compile(forecast_config=timesfm.ForecastConfig(
            max_context=mc, max_horizon=max(1024, H), infer_is_positive=True, use_continuous_quantile_head=True,
            fix_quantile_crossing=True, force_flip_invariance=True, return_backcast=False, normalize_inputs=True,
            per_core_batch_size=32))
        _, q = tf.forecast(horizon=H, inputs=[np.asarray(c, np.float64) for c in no_empty(contexts)])
        return np.asarray(q)[:, :H, 1:]                                  # [mean, 0.1 ... 0.9] -> 0.1 ... 0.9
    return fn


def toto2(m: dict, device: str):
    """Toto 2.0 through its own GluonTS predictor with the settings of its GIFT-Eval notebook (context 4096), one
    variate per series. The start timestamps given to it are placeholders: the model reads values only."""
    import pandas as pd
    from toto2 import Toto2Model
    from toto2.configuration import Toto2GluonTSModelConfig
    from toto2.model import Toto2GluonTSModel
    model = Toto2Model.from_pretrained(m["repo_id"], revision=m["revision"], map_location=device).to(device).eval()
    predictors = {}

    def fn(contexts, H):
        if H not in predictors:
            cfg = Toto2GluonTSModelConfig(prediction_length=H, context_length=4096, target_dim=1)
            predictors[H] = Toto2GluonTSModel(model, cfg).to(device).eval().create_predictor(batch_size=512,
                                                                                               device=device)
        start = pd.Period("2000-01-01 00:00", freq="h")
        fcs = predictors[H].predict([{"target": np.asarray(c, np.float32), "start": start} for c in contexts])
        return np.stack([np.stack([f.quantile(q) for q in LEVELS], -1) for f in fcs])
    return fn


VARIANTS = {("TiRex", "resample"): "TiRex (resampling)"}


def result_name(name: str, variant: str | None) -> str:
    """Name of a result line: the model, or the model with an inference variant of its own official notebook."""
    if variant is None:
        return name
    if (name, variant) not in VARIANTS:
        raise ValueError(f"{name} has no variant {variant!r}")
    return VARIANTS[(name, variant)]


LOADERS = {"Timestral-5M": timestral, "Timestral-28M": timestral, "Chronos-2": chronos2, "TiRex": tirex,
           "TimesFM-2.5": timesfm25, "Toto-2.0-22m": toto2}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", choices=["gift", "fev", "time"])
    ap.add_argument("--name", required=True, choices=sorted(LOADERS))
    ap.add_argument("--gift-eval", help="gift: clone of github.com/SalesforceAIResearch/gift-eval")
    ap.add_argument("--fev-bench", help="fev: benchmarks/fev_bench of a clone of github.com/autogluon/fev")
    ap.add_argument("--work", help="time: work folder with the contexts of `benchmarks.time_bench dump`")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", action="store_true", help="skip the tasks already in all_results.csv")
    ap.add_argument("--variant", default=None, help="inference variant from the model's own notebook (TiRex: resample)")
    a = ap.parse_args()

    from importlib.metadata import version
    from benchmarks import gift
    from benchmarks.adapter import GluonTSAdapter
    m = entry(load_list(), a.name, a.benchmark)
    line = result_name(a.name, a.variant)
    fn = LOADERS[a.name](m, a.device, a.variant) if a.variant else LOADERS[a.name](m, a.device)
    out = a.out or out_dir(a.benchmark, line)
    os.makedirs(out, exist_ok=True)
    package = m["package"].split()[0].split(">")[0].split("=")[0]
    import datetime
    meta = {"model": line, "variant": a.variant, "repo_id": m["repo_id"], "revision": m["revision"], "package": package,
            "package_version": version(package), "date": datetime.date.today().isoformat(),
            "protocol": "forced univariate, computed by us"}
    if a.benchmark == "gift":
        with open(os.path.join(a.gift_eval, "notebooks", "dataset_properties.json")) as f:
            properties = json.load(f)
        out_csv = os.path.join(out, "all_results.csv")
        gift.evaluate(lambda pl: GluonTSAdapter(fn, pl, batch_size=256), line, properties, out_csv, resume=a.resume)
        h = gift.headline(out_csv, os.path.join(a.gift_eval, "results", "seasonal_naive", "all_results.csv"))
        meta.update(h)
        print(f"\n{line} GIFT-Eval ({h['n_tasks']} tasks): MASE {h['MASE']:.4f}  CRPS {h['CRPS']:.4f}")
    elif a.benchmark == "fev":
        from benchmarks import fev_bench
        run = f"{line} [univariate]"                 # distinct from the model's own entry on the leaderboard
        s = fev_bench.evaluate(fn, run, a.fev_bench)
        with open(os.path.join(out, "summaries.json"), "w") as f:
            json.dump(s, f, indent=1, default=str)
        h = fev_bench.headline(s, run, a.fev_bench, out)
        meta.update({k: v["skill_score"] for k, v in h.items()})
        print(f"\n{line} fev-bench: skill SQL {h['SQL']['skill_score']:.2f}  MASE {h['MASE']['skill_score']:.2f}")
    else:
        from benchmarks import time_bench
        time_bench.forecast(a.work, fn, line)
        print(f"\n{line} TIME: quantiles written; score with `benchmarks.time_bench score --name '{line}'`")
    with open(os.path.join(out, "headline.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
