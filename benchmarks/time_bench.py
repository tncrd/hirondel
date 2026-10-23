"""TIME (98 tasks) with the official loader and scorer (github.com/zqiao11/TIME, `timebench`).

`timebench` pins numpy 1.26, datasets 2.17 and gluonts 0.15, so it runs in its own environment (as TIME's own scripts
do), and the model in this package's environment. The two meet through files in a work folder:

    # 1. TIME environment: contexts of every task, cut by TIME's loader
    python -m benchmarks.time_bench dump --time-repo TIME --work work/time
    # 2. package environment: quantiles for every context
    python -m benchmarks.time_bench forecast --work work/time --model <folder or repo id> [--device cuda]
    # 3. TIME environment: TIME's saver writes the per-window metrics, then the headline
    python -m benchmarks.time_bench score --time-repo TIME --work work/time
    cd work/time && python ../../TIME/scripts/compute_local_leaderboard.py     # official table (3 decimals)

TIME_DATASET must point to the TIME data. Multivariate tasks are split into univariate series (`to_univariate=True`, the
benchmark's option for univariate models); all 98 tasks are scored. Headline: per task, MASE and CRPS averaged over
windows and series, divided by Seasonal Naive's (TIME's published results), geometric mean over the tasks, as in
`scripts/compute_local_leaderboard.py`; `headline` gives it unrounded.

This module imports neither torch nor the package at the top: stages 1 and 3 run without them.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_TASKS = 98
NAIVE = "seasonal_naive"


def pack(series: list) -> tuple[np.ndarray, np.ndarray]:
    flat = np.concatenate(series).astype(np.float32) if series else np.zeros(0, np.float32)
    return flat, np.cumsum([0] + [len(s) for s in series]).astype(np.int64)


def unpack(flat: np.ndarray, offsets: np.ndarray) -> list:
    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def _file(name: str, term: str) -> str:
    return f"{name.replace('/', '__')}__{term}"


# ---- TIME environment ------------------------------------------------------------------------------------------------
def _config(time_repo: str) -> dict:
    import yaml
    sys.path.insert(0, os.path.join(time_repo, "src"))
    with open(os.path.join(time_repo, "src", "timebench", "config", "datasets.yaml")) as f:
        return yaml.safe_load(f)


def tasks(cfg: dict) -> list[tuple[str, str]]:
    pairs = [(name, term) for name, c in cfg["datasets"].items() for term in ("short", "medium", "long")
             if isinstance(c, dict) and term in c]
    if len(pairs) != N_TASKS:
        raise ValueError(f"datasets.yaml gives {len(pairs)} tasks, expected {N_TASKS}")
    return pairs


def _dataset(name: str, term: str, cfg: dict):
    """TIME's loader with TIME's settings. `to_univariate` only when there is more than one variate: on a univariate
    dataset it collapses the target to a 0-d array and gluonts fails."""
    from timebench.evaluation.data import Dataset, get_dataset_settings
    st = get_dataset_settings(name, term, cfg)
    kw = dict(name=name, term=term, prediction_length=st["prediction_length"], test_length=st["test_length"])
    probe = Dataset(to_univariate=False, **kw)
    return probe if probe.target_dim <= 1 else Dataset(to_univariate=True, **kw)


def dump(time_repo: str, work: str) -> None:
    cfg = _config(time_repo)
    os.makedirs(os.path.join(work, "contexts"), exist_ok=True)
    for i, (name, term) in enumerate(tasks(cfg)):
        ds = _dataset(name, term, cfg)
        flat, offsets = pack([np.asarray(e["target"], np.float32) for e in ds.test_data.input])
        np.savez_compressed(os.path.join(work, "contexts", _file(name, term) + ".npz"), flat=flat, offsets=offsets,
                            prediction_length=ds.prediction_length, name=name, term=term)
        print(f"[{i + 1:3d}/{N_TASKS}] {name}/{term}: {len(offsets) - 1} series, H = {ds.prediction_length}", flush=True)


def score(time_repo: str, work: str, model_name: str) -> None:
    cfg = _config(time_repo)                                   # puts TIME's src on the path first
    from gluonts.time_feature import get_seasonality
    from timebench.evaluation.saver import save_window_predictions
    out = os.path.join(work, "output", "results", model_name)
    for i, (name, term) in enumerate(tasks(cfg)):
        q = np.load(os.path.join(work, "quantiles", model_name, _file(name, term) + ".npz"))["q"]   # [N, H, 9]
        ds = _dataset(name, term, cfg)
        # TIME's contract is [N, quantile, H]; the seasonality is the MASE denominator (TIME's default of 1 is wrong
        # for every sub-daily dataset; their Seasonal Naive results use get_seasonality(freq)).
        save_window_predictions(ds, np.ascontiguousarray(q.transpose(0, 2, 1)), ds_config=f"{name}/{term}",
                                output_base_dir=out, seasonality=get_seasonality(str(ds.freq)), quantile_levels=LEVELS)
        print(f"[{i + 1:3d}/{N_TASKS}] {name}/{term} scored", flush=True)


# ---- package environment ---------------------------------------------------------------------------------------------
def forecast(work: str, forecast_fn, model_name: str, batch_size: int = 256) -> None:
    out = os.path.join(work, "quantiles", model_name)
    os.makedirs(out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(work, "contexts", "*.npz")))
    if len(paths) != N_TASKS:
        raise ValueError(f"{len(paths)} context files in {work}, expected {N_TASKS}: run the dump stage first")
    for i, path in enumerate(paths):
        z = np.load(path, allow_pickle=False)
        series, pl = unpack(z["flat"], z["offsets"]), int(z["prediction_length"])
        q = np.concatenate([np.asarray(forecast_fn(series[j:j + batch_size], pl), np.float32)
                            for j in range(0, len(series), batch_size)])
        if q.shape != (len(series), pl, len(LEVELS)) or not np.isfinite(q).all():
            raise ValueError(f"{os.path.basename(path)}: bad forecast, shape {q.shape}")
        np.savez_compressed(os.path.join(out, os.path.basename(path)), q=q)
        print(f"[{i + 1:3d}/{N_TASKS}] {os.path.basename(path)[:-4]}: {q.shape}", flush=True)


# ---- headline (either environment) -----------------------------------------------------------------------------------
def _task_means(model_root: str) -> dict:
    out = {}
    for path in glob.glob(os.path.join(model_root, "*", "*", "*", "metrics.npz")):
        dataset, freq, horizon = path.split(os.sep)[-4:-1]
        m = np.load(path)
        with np.errstate(all="ignore"):
            out[(f"{dataset}/{freq}", horizon)] = (float(np.nanmean(m["MASE"])), float(np.nanmean(m["CRPS"])))
    return out


def headline(results_root: str, model_name: str, naive_root: str) -> dict:
    """Unrounded TIME headline: per task mean / Seasonal Naive mean, geometric mean over the tasks with a baseline."""
    ours, naive = _task_means(os.path.join(results_root, model_name)), _task_means(os.path.join(naive_root, NAIVE))
    common = sorted(set(ours) & set(naive))
    out = {"n_tasks": len(common)}
    for j, metric in enumerate(("MASE", "CRPS")):
        r = [ours[k][j] / naive[k][j] for k in common if naive[k][j] and np.isfinite(ours[k][j] / naive[k][j])]
        out[metric] = float(np.exp(np.mean(np.log(r))))
    return out


def write_results(results_root: str, model_name: str, naive_root: str, out_dir: str) -> dict:
    """Per-task means and normalised values (`tasks.csv`) and the headline (`headline.json`)."""
    import json
    ours, naive = _task_means(os.path.join(results_root, model_name)), _task_means(os.path.join(naive_root, NAIVE))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tasks.csv"), "w") as f:
        f.write("task,MASE,CRPS,MASE_norm,CRPS_norm\n")
        for k in sorted(set(ours) & set(naive)):
            (m, c), (nm, nc) = ours[k], naive[k]
            f.write(f"{k[0]}/{k[1]},{m!r},{c!r},{m / nm!r},{c / nc!r}\n")
    h = headline(results_root, model_name, naive_root)
    path = os.path.join(out_dir, "headline.json")
    saved = {}
    if os.path.exists(path):                   # provenance written by the forecast stage (competitors) is kept
        with open(path) as f:
            saved = json.load(f)
    with open(path, "w") as f:
        json.dump({**saved, **h}, f, indent=1)
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["dump", "forecast", "score"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--time-repo", help="clone of github.com/zqiao11/TIME (dump, score)")
    ap.add_argument("--model", default="weights", help="forecast: local folder or Hugging Face repo id")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="hirondel")
    ap.add_argument("--naive", help="score: folder holding seasonal_naive/ (TIME-Output results); default: download")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "time"))
    a = ap.parse_args()
    if a.stage == "dump":
        dump(a.time_repo, a.work)
    elif a.stage == "forecast":
        from benchmarks.adapter import hirondel_forecast_fn
        from hirondel import HirondelForecaster
        forecast(a.work, hirondel_forecast_fn(HirondelForecaster.from_pretrained(a.model, device=a.device)), a.name)
    else:
        score(a.time_repo, a.work, a.name)
        naive = a.naive
        if naive is None:
            from huggingface_hub import snapshot_download
            naive = os.path.join(snapshot_download("Real-TSF/TIME-Output", repo_type="dataset",
                                                   allow_patterns=[f"results/{NAIVE}/**"]), "results")
        h = write_results(os.path.join(a.work, "output", "results"), a.name, naive, a.out)
        print(f"TIME ({h['n_tasks']} tasks): MASE {h['MASE']:.4f}  CRPS {h['CRPS']:.4f}")


if __name__ == "__main__":
    main()
