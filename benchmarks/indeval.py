"""IND-Eval (cognitedata, arXiv 2610.16652): industrial forecasting tasks, scored as in the paper's appendix.

    python -m benchmarks.indeval --data <folder holding data/<source>/*.parquet> --out benchmarks/results/indeval \
        [--sources industrial|all] [--model weights] [--device cuda]

Data: `huggingface.co/datasets/cognitedata/IND-Eval` at revision `REV` (tag v0.1.5), the parquet files of the sources
below; windows are read as published (context up to 4 096 points, future of H points). ETT is not distributed (CC BY-ND):
its tasks are rebuilt locally with the dataset's own `raw/ett/scripts/prepare.py --output-dir <data>/data/ett`.

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
  - headline: the five industrial sources (`SOURCES`); MetroPT-3 is part of TIME and Electricity, Solar, ETT of
    GIFT-Eval (`SHARED`), so they are scored apart, and with them the nine-source aggregate of the paper (`--sources all`);
  - the GIFT-Eval-derived sources ship no `seasonality` column: m is the paper's period of their frequency (`PERIODS`).
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
SHARED = ("metropt3", "ett", "electricity", "solar")
PERIODS = {"10min": 144, "15min": 96, "h": 24}          # paper's appendix; daily and weekly: m = 1
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
MEDIAN = 4
CONTEXT_MAX = 4096
Z = np.array([NormalDist().inv_cdf(float(q)) for q in LEVELS])
NAIVE = "seasonal naive"


def task_files(data: str, source: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data, "data", source, "**", "*.parquet"), recursive=True))   # ETT: ett1/, ett2/
    if not files:
        raise FileNotFoundError(f"no task for {source} under {data}/data/{source}")
    return files


def iter_pairs(path: str, batch: int = 4096):
    """Batches {inst, ctx (float32, at most CONTEXT_MAX), y, m, H} of the TARGET channels of a task file."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    icol = "instance_id" if "instance_id" in names else "source_instance"
    pcol = _period_column(names)
    cols = [icol, "context", "target", pcol, "prediction_length"] + (["role"] if "role" in names else [])
    for rb in pf.iter_batches(batch_size=batch, columns=cols):
        d = rb.to_pydict()
        keep = [i for i in range(rb.num_rows) if d.get("role", ["target"] * rb.num_rows)[i] == "target"]
        if keep:
            yield {"inst": [d[icol][i] for i in keep],
                   "ctx": [np.asarray(d["context"][i], np.float64)[-CONTEXT_MAX:].astype(np.float32) for i in keep],
                   "y": [np.asarray(d["target"][i], np.float64) for i in keep],
                   "m": [period(d[pcol][i]) for i in keep], "H": int(d["prediction_length"][keep[0]])}


def iter_windows(path: str, batch: int = 4096):
    """The windows of a task file, in order: {"targets": [context, ...], "covariates": [(context, future), ...], "H"}.
    Targets come in the order of `iter_pairs`; covariates (PRONTO's control channels) with their known future. For the
    models that forecast the targets of a window jointly and read its covariates, as in the paper."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    icol = "instance_id" if "instance_id" in names else "source_instance"
    cols = [icol, "window", "context", "target", "prediction_length"] + (["role"] if "role" in names else [])
    key, cur = None, None
    for rb in pf.iter_batches(batch_size=batch, columns=cols):
        d = rb.to_pydict()
        for i in range(rb.num_rows):
            if (d[icol][i], d["window"][i]) != key:
                if cur:
                    yield cur
                key, cur = (d[icol][i], d["window"][i]), {"targets": [], "covariates": [],
                                                          "H": int(d["prediction_length"][i])}
            ctx = np.asarray(d["context"][i], np.float64)[-CONTEXT_MAX:].astype(np.float32)
            if d.get("role", ["target"] * rb.num_rows)[i] == "covariate":
                cur["covariates"].append((ctx, np.asarray(d["target"][i], np.float32)))
            else:
                cur["targets"].append(ctx)
    if cur:
        yield cur


def _period_column(names) -> str:
    return "seasonality" if "seasonality" in names else "freq"


def period(v) -> int:
    """Seasonal period m of a pair: its `seasonality` value, or the period of its frequency (`PERIODS`, else 1)."""
    return PERIODS.get(v, 1) if isinstance(v, str) else int(v)


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
    """{model: (MASE, CRPS)} of one task, with the pair counts. `models`: name -> fn(contexts, H, futures) -> (H, 9)s,
    or a joint model, whose `task(path)` yields the (H, 9) of every target pair of the file in order (`iter_windows`)."""
    joint = {k: iter(fn.task(path)) for k, fn in models.items() if hasattr(fn, "task")}
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
            qs = [next(joint[name]) for _ in C] if name in joint else fn(C, H, Y)
            for q, y, c in zip(qs, Y, C):
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
    res["aggregate"] = aggregate(res["sources"], sources, models)
    res.update({"sources_order": list(sources), "revision": REV, "seconds": round(time.time() - t0, 1)})
    head = {"sources": list(sources), "revision": REV, "nMASE_nCRPS": res["aggregate"]}
    if set(SOURCES) < set(sources):                 # the headline stays the five industrial sources
        head["nMASE_nCRPS_industrial"] = aggregate(res["sources"], SOURCES, models)
    with open(os.path.join(out, "headline.json"), "w") as fh:
        json.dump(head | {"per_source": res["sources"], "univariate": True, "covariates": "none"}, fh, indent=1)
    return res


def aggregate(per_source: dict, sources, models) -> dict:
    """Geometric mean over the sources, equal weights, per model."""
    return {k: [_gmean([per_source[s][k][0] for s in sources]), _gmean([per_source[s][k][1] for s in sources])]
            for k in models}


def _with_period(models: dict, path: str) -> dict:
    """The seasonal naive reads each pair's period m; the other models only see contexts and H."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    pcol = _period_column(names)
    t = pf.read(columns=[pcol] + (["role"] if "role" in names else [])).to_pydict()
    ms = [period(m) for i, m in enumerate(t[pcol]) if t.get("role", ["target"] * len(t[pcol]))[i] == "target"]
    state = {"i": 0}

    def naive(C, H, Y):
        i = state["i"]
        state["i"] += len(C)
        return [snaive_quantiles(c, H, m) for c, m in zip(C, ms[i:i + len(C)])]
    return {k: (naive if k == NAIVE else v) for k, v in models.items()}


def covariate_model(forecaster, mode: str, batch: int = 256):
    """Hirondel with the known-future covariates of each window (PRONTO's control channels): every target channel is
    still forecast alone, with `forecast(..., future_covariates, covariate_mode=mode)`. Windows without covariates
    get the univariate forecast, bit for bit. Contexts with no observed value go through `hirondel_forecast_fn`."""
    from benchmarks.adapter import hirondel_forecast_fn
    plain = hirondel_forecast_fn(forecaster)

    def task(path):
        for w in iter_windows(path):
            H, targets = w["H"], w["targets"]
            covs = []
            for ctx in targets:
                cols = {}
                for j, (cctx, cfut) in enumerate(w["covariates"]):
                    past = np.full(len(ctx), np.nan, np.float32)    # aligned on the end of the target's context
                    m = min(len(ctx), len(cctx))
                    past[len(ctx) - m:] = cctx[-m:]
                    cols[f"covariate_{j}"] = np.concatenate([past, cfut[:H]])
                covs.append(cols or None)
            out = [None] * len(targets)
            empty = [i for i, x in enumerate(targets) if not np.isfinite(x).any()]
            for i, q in zip(empty, plain([targets[i] for i in empty], H) if empty else []):
                out[i] = q
            keep = [i for i in range(len(targets)) if out[i] is None]
            for lo in range(0, len(keep), batch):
                idx = keep[lo:lo + batch]
                q = forecaster.forecast([targets[i] for i in idx], H, future_covariates=[covs[i] for i in idx],
                                        covariate_mode=mode).quantiles
                for i, qi in zip(idx, q):
                    out[i] = qi
            yield from out

    def fn(contexts, H, futures=None):
        raise ValueError("covariate_model reads the windows of a task file, see `task`")
    fn.task = task
    return fn


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder holding data/<source>/*.parquet (revision REV)")
    ap.add_argument("--out", default=None, help="default: benchmarks/results/indeval, or "
                    "benchmarks/results/indeval_competitors/Hirondel (covariates, <mode>) with --covariate-mode")
    ap.add_argument("--sources", default="industrial", choices=["industrial", "all", "pronto"],
                    help="the five industrial sources, the nine of the paper (with MetroPT-3, ETT, Electricity, Solar), "
                         "or PRONTO alone (the only source with covariates)")
    ap.add_argument("--model", default="weights", help="local folder or Hugging Face repo id")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--covariate-mode", default="off", choices=["off", "gate", "always"],
                    help="read PRONTO's control channels as known-future covariates (see hirondel.covariates)")
    a = ap.parse_args()
    from benchmarks.adapter import hirondel_forecast_fn
    from hirondel import HirondelForecaster, __version__
    forecaster = HirondelForecaster.from_pretrained(a.model, device=a.device)
    sources = ("pronto",) if a.sources == "pronto" else SOURCES + (SHARED if a.sources == "all" else ())
    if a.covariate_mode == "off":
        name, out = "Hirondel", a.out or "benchmarks/results/indeval"
        fn = hirondel_forecast_fn(forecaster)
        model = lambda C, H, Y: list(fn(C, H))                       # noqa: E731
    else:
        name = f"Hirondel (covariates, {a.covariate_mode})"
        out = a.out or os.path.join("benchmarks", "results", "indeval_competitors", name)
        model = covariate_model(forecaster, a.covariate_mode)
    res = run(a.data, sources, {name: model}, out, log=lambda *x: print(*x, flush=True))
    if a.covariate_mode != "off":
        path = os.path.join(out, "headline.json")
        with open(path) as f:
            head = json.load(f)
        head.update({"model": name, "package_version": __version__, "univariate": True,
                     "covariates": f"PRONTO's control channels as known-future covariates, covariate_mode="
                                   f"{a.covariate_mode!r}; each target channel forecast alone"})
        with open(path, "w") as f:
            json.dump(head, f, indent=1)
    if a.sources == "pronto":
        m, c = res["sources"]["pronto"][name]
        print(f"IND-Eval (PRONTO): nMASE {m:.3f}  nCRPS {c:.3f}  ({res['seconds']:.0f} s)")
        return
    for label, src in (("5 industrial sources", SOURCES), ("9 sources", sources))[:1 + (a.sources == "all")]:
        m, c = aggregate(res["sources"], src, [name])[name]
        print(f"IND-Eval ({label}): nMASE {m:.3f}  nCRPS {c:.3f}  ({res['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
