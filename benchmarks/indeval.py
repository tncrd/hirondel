"""IND-Eval (cognitedata, arXiv 2610.16652): industrial forecasting tasks, scored as in the paper's appendix.

    python -m benchmarks.indeval --data <folder holding data/<source>/*.parquet> --out benchmarks/results/indeval \
        [--model weights] [--device cuda]

Data: `huggingface.co/datasets/cognitedata/IND-Eval` at revision `REV` (tag v0.1.5), the parquet files of the five
industrial sources below; windows are read as published (context up to 4 096 points, future of H points).

Scoring, as in the paper's appendix:
  - a pair is one TARGET channel of one window (channels with role "covariate" are never read nor scored);
  - MASE of a pair = MAE of the median / mean |x_t - x_{t-m}| over the window's context (m = `seasonality` column);
    MASE of a task = mean over its pairs; CRPS = 2/K sum of the pinball losses at the 9 levels, summed over pairs and steps;
  - nMASE, nCRPS = model / Seasonal Naive (StatsForecast's SeasonalNaive, rewritten here: last value imputation, last
    season repeated, symmetric Gaussian intervals with sigma_h = sigma sqrt(floor((h-1)/m) + 1));
  - per source: geometric mean over tasks; aggregate: geometric mean over sources, equal weights.
Choices of ours, stated:
  - univariate: each target channel is forecast alone from its own past; no covariate is read (PRONTO's control
    channels included), whereas the paper forecasts the targets of a window jointly, with PRONTO's covariates;
  - pairs with a constant context (MASE scale 0, frequent in Mesa del Sol) are left out of the MASE on both sides of the
    ratio and counted; the paper does not say how it handles them;
  - the five industrial sources only: MetroPT-3 is part of TIME and Electricity, Solar, ETT of GIFT-Eval.
This reproduces the paper's Seasonal Naive normalisation to a few thousandths (Mesa del Sol, naive: 1.278 / 1.405 here,
1.280 / 1.410 in the paper), not bit for bit.
"""
from __future__ import annotations

import glob
import json
import os
import time
from statistics import NormalDist

import numpy as np

REPO, REV = "cognitedata/IND-Eval", "9b0dce3fd91a97b9428ab1da532f08403f4106e1"   # tag v0.1.5
SOURCES = ("neats", "3w", "pronto", "mds", "tablet_compression")
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
MEDIAN = 4
CONTEXT_MAX = 4096
Z = np.array([NormalDist().inv_cdf(float(q)) for q in LEVELS])
NAIVE = "seasonal naive"


def task_files(data: str, source: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data, "data", source, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no task for {source} under {data}/data/{source}")
    return files


def iter_pairs(path: str, batch: int = 4096):
    """Batches {inst, ctx (float32, at most CONTEXT_MAX), y, m, H} of the TARGET channels of a task file."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    icol = "instance_id" if "instance_id" in names else "source_instance"
    cols = [icol, "context", "target", "seasonality", "prediction_length"] + (["role"] if "role" in names else [])
    for rb in pf.iter_batches(batch_size=batch, columns=cols):
        d = rb.to_pydict()
        keep = [i for i in range(rb.num_rows) if d.get("role", ["target"] * rb.num_rows)[i] == "target"]
        if keep:
            yield {"inst": [d[icol][i] for i in keep],
                   "ctx": [np.asarray(d["context"][i], np.float64)[-CONTEXT_MAX:].astype(np.float32) for i in keep],
                   "y": [np.asarray(d["target"][i], np.float64) for i in keep],
                   "m": [int(d["seasonality"][i]) for i in keep], "H": int(d["prediction_length"][keep[0]])}


def seasonal_error(ctx, m: int) -> float:
    """MASE scale: mean |x_t - x_{t-m}| over the context (m = 1 when the context is too short), NaN pairs ignored."""
    c = np.asarray(ctx, np.float64)
    m = int(m) if len(c) > int(m) else 1
    d = np.abs(c[m:] - c[:-m])
    d = d[np.isfinite(d)]
    return float(d.mean()) if len(d) else float("nan")


def snaive_quantiles(ctx, H: int, m: int) -> np.ndarray:
    """Seasonal Naive with Gaussian intervals -> (H, 9)."""
    c = np.asarray(ctx, np.float64).copy()
    ok = np.isfinite(c)
    if not ok.any():
        return np.zeros((H, 9))
    idx = np.where(ok, np.arange(len(c)), 0)
    np.maximum.accumulate(idx, out=idx)
    c = c[idx]
    c[:np.argmax(ok)] = c[np.argmax(ok)]
    m = int(m) if len(c) >= int(m) else 1
    mean = np.resize(c[-m:], H)
    res = c[m:] - c[:-m]
    sigma = float(np.sqrt(np.sum(res ** 2) / (len(c) - m))) if len(c) > m else 0.0
    sh = sigma * np.sqrt(np.floor(np.arange(H) / m) + 1)
    return mean[:, None] + Z[None, :] * sh[:, None]


def pair_stats(q: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """q (H, 9), y (H,) -> (sum |y - median|, sum of 2/K sum_k pinball, valid steps). Missing future values are masked."""
    ok = np.isfinite(y)
    yv, qv = y[ok], np.asarray(q, np.float64)[ok]
    d = yv[:, None] - qv
    pin = np.maximum(LEVELS * d, (LEVELS - 1) * d)
    return float(np.abs(yv - qv[:, MEDIAN]).sum()), float(2 * pin.mean(1).sum()), int(ok.sum())


def fill_nan_last(q, ctx) -> np.ndarray:
    """A non-finite output is replaced by the last observed value of the context."""
    c = np.asarray(ctx, np.float64)
    c = c[np.isfinite(c)]
    last = float(c[-1]) if len(c) else 0.0
    return np.nan_to_num(np.asarray(q, np.float64), nan=last, posinf=last, neginf=last)


def score_task(path: str, models: dict, chunk: int = 8192) -> dict:
    """{model: (MASE, CRPS)} of one task, with the pair counts. `models`: name -> fn(contexts, H, futures) -> (H, 9)s."""
    sae = {k: [] for k in models}
    sql = {k: [] for k in models}
    scale, n_ok = [], []
    buf = []

    def flush():
        if not buf:
            return
        C = [c for b in buf for c in b["ctx"]]
        Y = [y for b in buf for y in b["y"]]
        H = buf[0]["H"]
        for name, fn in models.items():
            for i, (q, y, c) in enumerate(zip(fn(C, H, Y), Y, C)):
                a, s, n = pair_stats(fill_nan_last(q, c), y)
                sae[name].append(a)
                sql[name].append(s)
                if name == NAIVE:
                    n_ok.append(n)
        buf.clear()
    for b in iter_pairs(path):
        scale += [seasonal_error(c, m) for c, m in zip(b["ctx"], b["m"])]
        buf.append(b)
        if sum(len(x["y"]) for x in buf) >= chunk:
            flush()
    flush()
    scale, n_ok = np.array(scale), np.array(n_ok, np.float64)
    ok_crps = n_ok > 0
    ok_mase = ok_crps & np.isfinite(scale) & (scale > 0)
    out = {}
    for k in models:
        a, s = np.array(sae[k]), np.array(sql[k])
        out[k] = (float(np.mean(a[ok_mase] / n_ok[ok_mase] / scale[ok_mase])), float(s[ok_crps].sum()))
    return {"scores": out, "n_pairs": int(len(scale)), "n_left_out_of_mase": int((~ok_mase).sum())}


def _gmean(x) -> float:
    x = np.asarray(x, np.float64)
    return float(np.exp(np.mean(np.log(x))))


def run(data: str, sources, models: dict, out: str, log=print) -> dict:
    """Scores every task of each source and writes `<out>/<source>.json` as each source ends, then `headline.json`."""
    models = {NAIVE: None} | models                # filled per task with each pair's period (`_with_period`)
    os.makedirs(out, exist_ok=True)
    res, t0 = {"sources": {}}, time.time()
    for s in sources:
        tasks = {}
        for f in task_files(data, s):
            tasks[os.path.basename(f).removesuffix(".parquet")] = score_task(f, _with_period(models, f))
        norm = {t: {k: (v["scores"][k][0] / v["scores"][NAIVE][0], v["scores"][k][1] / v["scores"][NAIVE][1])
                    for k in models} for t, v in tasks.items()}
        res["sources"][s] = {k: [_gmean([n[k][0] for n in norm.values()]), _gmean([n[k][1] for n in norm.values()])]
                             for k in models}
        with open(os.path.join(out, f"{s}.json"), "w") as fh:
            json.dump({"source": s, "scores": res["sources"][s], "tasks": norm,
                       "n_pairs": sum(v["n_pairs"] for v in tasks.values()),
                       "n_left_out_of_mase": sum(v["n_left_out_of_mase"] for v in tasks.values())}, fh, indent=1)
        log(f"[{s}] " + "  ".join(f"{k} {v[0]:.3f} / {v[1]:.3f}" for k, v in res["sources"][s].items())
            + f"  ({time.time() - t0:.0f} s)")
    res["aggregate"] = {k: [_gmean([res["sources"][s][k][0] for s in sources]),
                            _gmean([res["sources"][s][k][1] for s in sources])] for k in models}
    res.update({"sources_order": list(sources), "revision": REV, "seconds": round(time.time() - t0, 1)})
    with open(os.path.join(out, "headline.json"), "w") as fh:
        json.dump({"sources": list(sources), "revision": REV, "nMASE_nCRPS": res["aggregate"],
                   "per_source": res["sources"], "univariate": True, "covariates": "none"}, fh, indent=1)
    return res


def _with_period(models: dict, path: str) -> dict:
    """The seasonal naive reads each pair's period m; the other models only see contexts and H."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    cols = ["seasonality"] + (["role"] if "role" in names else [])
    t = pf.read(columns=cols).to_pydict()
    ms = [m for i, m in enumerate(t["seasonality"]) if t.get("role", ["target"] * len(t["seasonality"]))[i] == "target"]
    state = {"i": 0}

    def naive(C, H, Y):
        i = state["i"]
        state["i"] += len(C)
        return [snaive_quantiles(c, H, m) for c, m in zip(C, ms[i:i + len(C)])]
    return {k: (naive if k == NAIVE else v) for k, v in models.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder holding data/<source>/*.parquet (revision REV)")
    ap.add_argument("--out", default="benchmarks/results/indeval")
    ap.add_argument("--model", default="weights", help="local folder or Hugging Face repo id")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from benchmarks.adapter import hirondel_forecast_fn
    from hirondel import HirondelForecaster
    fn = hirondel_forecast_fn(HirondelForecaster.from_pretrained(a.model, device=a.device))
    res = run(a.data, SOURCES, {"Hirondel": lambda C, H, Y: list(fn(C, H))}, a.out, log=lambda *x: print(*x, flush=True))
    m, c = res["aggregate"]["Hirondel"]
    print(f"IND-Eval (5 industrial sources): nMASE {m:.3f}  nCRPS {c:.3f}  ({res['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
