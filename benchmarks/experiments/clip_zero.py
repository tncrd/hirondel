"""Measurement only, outside the published inference path: zero clipping of the quantiles of series whose observed context
is non-negative (Timestral's `clip_nonnegative` rule). Two variants, as forecast functions for the benchmark adapters:

- `after_flip`: the package's forecast (flip average included), then every quantile clipped at 0;
- `per_pass`: the two passes of the flip computed separately, the direct pass clipped before the average (the earlier
  test); the pass on -x is never clipped since its context is non-positive.
"""
from __future__ import annotations

import numpy as np

from benchmarks.adapter import nom_forecast_fn


def non_negative(contexts) -> np.ndarray:
    """Per series: at least one observed value, and every observed value >= 0."""
    out = []
    for c in contexts:
        c = np.asarray(c, np.float64)
        obs = np.isfinite(c)
        out.append(bool(obs.any() and (c[obs] >= 0).all()))
    return np.array(out, bool)


def after_flip(forecaster):
    def fn(contexts, prediction_length):
        q = nom_forecast_fn(forecaster)(contexts, prediction_length)
        keep = non_negative(contexts)[:, None, None]
        return np.where(keep, np.maximum(q, 0), q).astype(np.float32)
    return fn


def per_pass(forecaster, clip: bool = True):
    def fn(contexts, prediction_length):
        one_pass = nom_forecast_fn(forecaster, flip=False)              # handles contexts with no observed value
        up = one_pass(contexts, prediction_length)
        down = one_pass([-np.asarray(c) for c in contexts], prediction_length)
        if clip:
            up = np.where(non_negative(contexts)[:, None, None], np.maximum(up, 0), up)
        return (0.5 * (up - down[..., ::-1])).astype(np.float32)
    return fn


def main():
    """python -m benchmarks.experiments.clip_zero {gift,fev,time} --variant {after_flip,per_pass} [paths]"""
    import argparse
    import json
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["gift", "fev", "time"])
    ap.add_argument("--variant", required=True, choices=["after_flip", "per_pass"])
    ap.add_argument("--gift-eval")
    ap.add_argument("--fev-bench")
    ap.add_argument("--work", help="TIME work folder (contexts from `benchmarks.time_bench dump`)")
    ap.add_argument("--model", default="weights")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from nom import NomForecaster
    f = NomForecaster.from_pretrained(a.model, device=a.device)
    fn = after_flip(f) if a.variant == "after_flip" else per_pass(f)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "experiments",
                       "clip_zero", a.variant, a.bench)
    os.makedirs(out, exist_ok=True)
    name = f"nom_clip_{a.variant}"
    if a.bench == "gift":
        from benchmarks import gift
        from benchmarks.adapter import GluonTSAdapter
        with open(os.path.join(a.gift_eval, "notebooks", "dataset_properties.json")) as fp:
            props = json.load(fp)
        csv_path = os.path.join(out, "all_results.csv")
        gift.evaluate(lambda pl: GluonTSAdapter(fn, pl), name, props, csv_path)
        h = gift.headline(csv_path, os.path.join(a.gift_eval, "results", "seasonal_naive", "all_results.csv"))
        print(f"GIFT-Eval {a.variant}: MASE {h['MASE']:.4f}  CRPS {h['CRPS']:.4f}")
    elif a.bench == "fev":
        from benchmarks import fev_bench
        s = fev_bench.evaluate(fn, name, a.fev_bench)
        with open(os.path.join(out, "summaries.json"), "w") as fp:
            json.dump(s, fp, indent=1, default=str)
        h = fev_bench.headline(s, name, a.fev_bench, out)
        with open(os.path.join(out, "headline.json"), "w") as fp:
            json.dump(h, fp, indent=1)
        print(f"fev-bench {a.variant}: " + ", ".join(f"{m} skill {v['skill_score']:.2f}" for m, v in h.items()))
    else:
        from benchmarks import time_bench
        time_bench.forecast(a.work, fn, name)
        print(f"TIME {a.variant}: quantiles written; score with `benchmarks.time_bench score --name {name}`")


if __name__ == "__main__":
    main()
