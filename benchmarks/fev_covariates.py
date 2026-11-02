"""fev-bench, the 42 tasks with covariates: Hirondel without and with its known-future covariates (`hirondel.covariates`).

Same harness as `benchmarks/fev_bench.py` (official `fev.Task` windows and `evaluation_summary`), three runs through
`HirondelForecaster.forecast`: "off" (the covariates are not read, the published univariate forecast), "always" and
"gate" (`covariate_mode`). The known-future columns (`known_dynamic_columns`) of each window are passed over context and
horizon; the 12 tasks with past-only covariates get the univariate forecast in every run. Skill scores as the fev-bench
site computes them, restricted to these 42 tasks, with the published models.

    python -m benchmarks.fev_covariates --fev-bench fev/benchmarks/fev_bench --out benchmarks/results/fev_covariates \
        [--device cuda]

Resumable: finished tasks are kept in <out>/tasks.jsonl.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np

from benchmarks import fev_bench

MODES = ("off", "always", "gate")
BATCH = 256


def name(mode: str) -> str:
    return "Hirondel" if mode == "off" else f"Hirondel (covariates, {mode})"


def forecast(forecaster, series, covariates, H, mode):
    """[n, H, 9]; contexts with no observed value go through the harness adapter (`hirondel_forecast_fn`)."""
    from benchmarks.adapter import hirondel_forecast_fn
    out = np.empty((len(series), H, len(fev_bench.LEVELS)), np.float32)
    empty = np.array([not np.isfinite(s).any() for s in series])
    if empty.any():
        out[empty] = hirondel_forecast_fn(forecaster)([s for s, e in zip(series, empty) if e], H)
    keep = np.flatnonzero(~empty)
    for lo in range(0, len(keep), BATCH):
        idx = keep[lo:lo + BATCH]
        s, c = [series[i] for i in idx], [covariates[i] for i in idx]
        out[idx] = (forecaster.forecast(s, H) if mode == "off" else
                    forecaster.forecast(s, H, future_covariates=c, covariate_mode=mode)).quantiles
    return out


def predict_task(forecaster, task):
    """{mode: one predictions dataset per window}; each target column forecast as its own series."""
    import datasets
    import fev
    preds = {m: [] for m in MODES}
    for window in task.iter_windows(num_proc=1):
        past, fut = window.get_input_data()
        series, covariates = [], []
        for row_p, row_f in zip(past, fut):                 # item, then target column: the order fev expects
            known = {c: np.concatenate([np.asarray(row_p[c], dtype=object), np.asarray(row_f[c], dtype=object)])
                     for c in task.known_dynamic_columns}
            for tc in task.target_columns:
                series.append(np.asarray(row_p[tc], dtype=np.float32))
                covariates.append(known or None)
        for m in MODES:
            q = forecast(forecaster, series, covariates, task.horizon, m)
            if not np.isfinite(q).all():
                raise ValueError(f"{task.task_name} ({m}): non-finite forecast")
            ds = datasets.Dataset.from_dict(fev_bench.prediction_columns(q))
            preds[m].append(fev.combine_univariate_predictions_to_multivariate(ds, target_columns=task.target_columns))
    return preds


def covariate_tasks(fev_bench_dir: str) -> list:
    import fev
    bench = fev.Benchmark.from_list(fev_bench.tasks(fev_bench_dir))
    return [t for t in bench.tasks if t.known_dynamic_columns or t.past_dynamic_columns]


def run(forecaster, fev_bench_dir: str, out_dir: str, log=print, only=None) -> dict:
    """{mode: [evaluation summary per task]}, resumed from <out_dir>/tasks.jsonl."""
    tasks = [t for t in covariate_tasks(fev_bench_dir) if only is None or t.task_name in only]
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "tasks.jsonl")
    done = {}
    if os.path.exists(cache):
        with open(cache) as f:
            for line in f:
                r = json.loads(line)
                done[r["task"]] = r["summaries"]
    for i, task in enumerate(tasks):
        if task.task_name in done:
            continue
        t0 = time.perf_counter()
        preds = predict_task(forecaster, task)
        s = {}
        for m in MODES:
            s[m] = task.evaluation_summary(preds[m], model_name=name(m), inference_time_s=time.perf_counter() - t0)
            s[m]["trained_on_this_dataset"] = False
        done[task.task_name] = s
        with open(cache, "a") as f:
            f.write(json.dumps({"task": task.task_name, "summaries": s}, default=str) + "\n")
        log(f"[{i + 1:2d}/{len(tasks)}] {task.task_name:38s} "
            + " / ".join(f"{s[m]['test_error']:.4f}" for m in MODES) + f"  ({time.perf_counter() - t0:.0f} s)")
        del preds
        gc.collect()
    return {m: [done[t.task_name][m] for t in tasks] for m in MODES}


def published(fev_bench_dir: str):
    """Every published result, the models the site hides included (as `showcase.build.fev_skills`: a skill score does
    not depend on which other models are listed)."""
    import glob
    import pandas as pd
    return pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(fev_bench_dir, "results", "*.csv")))],
                     ignore_index=True)


def leaderboard(summaries: dict, fev_bench_dir: str, metric: str = "SQL", tasks=None):
    """The site's leaderboard (skill score against Seasonal Naive) over `tasks` (default: those of `summaries`):
    published models and our runs."""
    import fev
    import pandas as pd
    names = tasks or [s["task_name"] for s in next(iter(summaries.values()))]
    off = published(fev_bench_dir)
    df = pd.concat([off[off["task_name"].isin(names)]] + [pd.DataFrame(v) for v in summaries.values()],
                   ignore_index=True)
    return fev.analysis.leaderboard(summaries=df, metric_column=metric, missing_strategy="impute",
                                    baseline_model=fev_bench.BASELINE,
                                    leakage_imputation_model=fev_bench.LEAKAGE_IMPUTATION,
                                    normalize_time_per_n_forecasts=100)


def headline(summaries: dict, fev_bench_dir: str, out_dir: str) -> dict:
    known = [t.task_name for t in covariate_tasks(fev_bench_dir) if t.known_dynamic_columns]
    from hirondel import __version__
    from hirondel.covariates import GATE_MARGIN
    lb = leaderboard(summaries, fev_bench_dir)
    lb.to_csv(os.path.join(out_dir, "leaderboard_SQL.csv"))
    err = {m: np.array([s["test_error"] for s in summaries[m]]) for m in MODES}
    tasks = [s["task_name"] for s in summaries["off"]]
    out = {"package_version": __version__, "gate_margin": GATE_MARGIN, "n_tasks": len(tasks),
           "known_future_tasks": known,
           "skill_SQL": {name(m): float(lb.loc[name(m), "skill_score"]) for m in MODES},
           "per_task_test_error": {t: {m: float(err[m][i]) for m in MODES} for i, t in enumerate(tasks)}}
    full = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "fev", "summaries.json")
    if os.path.exists(full):                       # all 100 tasks: the univariate run, covariate tasks replaced
        with open(full) as f:
            base = json.load(f)
        base = base if isinstance(base, list) else base["summaries"]
        by_task = {t: i for i, t in enumerate(tasks)}
        runs = {name(m): [{**(summaries[m][by_task[s["task_name"]]] if s["task_name"] in by_task else s),
                           "model_name": name(m)} for s in base] for m in ("off", "gate")}
        out["all_tasks"] = {"n_tasks": len(base)}
        for metric in ("SQL", "MASE"):
            lb100 = leaderboard(runs, fev_bench_dir, metric, tasks=[s["task_name"] for s in base])
            out["all_tasks"][f"skill_{metric}"] = {k: float(lb100.loc[k, "skill_score"]) for k in runs}
            out["all_tasks"][f"skill_{metric}"]["Chronos-2"] = float(lb100.loc["Chronos-2", "skill_score"])
    for m in ("always", "gate"):
        r = err[m] / err["off"]
        out[f"{m}_vs_off"] = {"gmean": float(np.exp(np.log(r).mean())), "worst": float(r.max()),
                              "worst_task": tasks[int(r.argmax())], "best": float(r.min()),
                              "best_task": tasks[int(r.argmin())], "n_worse": int((r > 1.001).sum()),
                              "n_better": int((r < 0.999).sum())}
    with open(os.path.join(out_dir, "headline.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fev-bench", required=True, help="benchmarks/fev_bench folder of the fev repository (ceb25ae)")
    ap.add_argument("--out", default=os.path.join("benchmarks", "results", "fev_covariates"))
    ap.add_argument("--model", default="weights", help="local folder or Hugging Face repo id")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None, help="comma-separated task names (a check; no headline)")
    a = ap.parse_args()
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from hirondel import HirondelForecaster
    forecaster = HirondelForecaster.from_pretrained(a.model, device=a.device)
    only = set(a.only.split(",")) if a.only else None
    summaries = run(forecaster, a.fev_bench, a.out, log=lambda *x: print(*x, flush=True), only=only)
    if only:
        return
    h = headline(summaries, a.fev_bench, a.out)
    print(json.dumps({k: v for k, v in h.items() if k != "per_task_test_error"}, indent=1))


if __name__ == "__main__":
    main()
