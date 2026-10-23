"""fev-bench (100 tasks) with the official harness: `fev.Task` windows and `evaluation_summary`, and the leaderboard as the
fev-bench site computes it (`fev.analysis.leaderboard`: skill score against Seasonal Naive, missing results imputed,
results on a training dataset replaced by Chronos-Bolt's; models shown by default: pretrained, task-specific and
statistical, without the hidden ones of `models.yaml`).

Univariate and covariate-blind: every target column is forecast as its own series (`as_univariate=True`) and covariate
columns are not read. All 100 tasks are scored. Our training data contains none of the fev-bench datasets (`DATA.md`),
so no task is declared `trained_on_this_dataset`.

    git clone https://github.com/autogluon/fev && git -C fev checkout ceb25ae
    pip install ".[bench]" "fev==0.9.0"
    python -m benchmarks.fev_bench --fev-bench fev/benchmarks/fev_bench --out benchmarks/results/fev [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
N_TASKS = 100
BASELINE = "Seasonal Naive"
LEAKAGE_IMPUTATION = "Chronos-Bolt"
VISIBLE_TYPES = ("pretrained", "task-specific", "statistical")


def tasks(fev_bench_dir: str) -> list[dict]:
    """The official task definitions (`tasks.yaml` of the fev repository), or an error if the list has drifted."""
    import yaml
    with open(os.path.join(fev_bench_dir, "tasks.yaml")) as f:
        out = yaml.safe_load(f)["tasks"]
    if len(out) != N_TASKS:
        raise ValueError(f"tasks.yaml has {len(out)} tasks, expected {N_TASKS}")
    return out


def official_summaries(fev_bench_dir: str):
    """Published results of the models the site shows by default (DataFrame)."""
    import pandas as pd
    import yaml
    with open(os.path.join(fev_bench_dir, "models.yaml")) as f:
        meta = yaml.safe_load(f)
    keep = [k for k, v in meta.items() if v.get("model_type") in VISIBLE_TYPES and not v.get("hidden")]
    paths = [os.path.join(fev_bench_dir, "results", f"{k}.csv") for k in keep]
    return pd.concat([pd.read_csv(p) for p in paths if os.path.exists(p)], ignore_index=True)


def prediction_columns(q: np.ndarray) -> dict:
    """[n_series, horizon, 9] -> the columns fev expects: "predictions" (median) and one per quantile level."""
    cols = {"predictions": q[..., LEVELS.index(0.5)].tolist()}
    for j, level in enumerate(LEVELS):
        cols[str(level)] = q[..., j].tolist()
    return cols


def predict_task(task, forecast_fn, batch_size: int = 256) -> list:
    """One predictions dataset per evaluation window; each target column forecast as its own series."""
    import datasets
    import fev
    out = []
    for window in task.iter_windows(num_proc=1):
        past = fev.convert_input_data(window, adapter="pandas", as_univariate=True)[0]
        series = [g["target"].to_numpy(np.float32) for _, g in past.groupby("id", sort=False)]
        q = np.concatenate([np.asarray(forecast_fn(series[i:i + batch_size], task.horizon))
                            for i in range(0, len(series), batch_size)])
        if q.shape != (len(series), task.horizon, len(LEVELS)) or not np.isfinite(q).all():
            raise ValueError(f"{task.task_name}: bad forecast, shape {q.shape}")
        preds = datasets.Dataset.from_dict(prediction_columns(q))
        out.append(fev.combine_univariate_predictions_to_multivariate(preds, target_columns=task.target_columns))
    return out


def evaluate(forecast_fn, model_name: str, fev_bench_dir: str, only=None) -> list[dict]:
    import fev
    bench = fev.Benchmark.from_list(tasks(fev_bench_dir))
    summaries = []
    for i, task in enumerate(bench.tasks):
        if only is not None and task.task_name not in only:
            continue
        t0 = time.perf_counter()
        preds = predict_task(task, forecast_fn)
        s = task.evaluation_summary(preds, model_name=model_name, inference_time_s=time.perf_counter() - t0)
        s["trained_on_this_dataset"] = False
        summaries.append(s)
        print(f"[{i + 1:3d}/{N_TASKS}] {s['task_name']:40s} SQL {s['SQL']:.4f}  MASE {s['MASE']:.4f}  "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return summaries


def leaderboard(summaries: list[dict], model_name: str, fev_bench_dir: str, metric: str):
    """The site's leaderboard with this model added; sorted by win rate as on the site."""
    import fev
    import pandas as pd
    ours = pd.DataFrame(summaries)
    ours["model_name"] = model_name
    lb = fev.analysis.leaderboard(summaries=pd.concat([official_summaries(fev_bench_dir), ours], ignore_index=True),
                                  metric_column=metric, missing_strategy="impute", baseline_model=BASELINE,
                                  leakage_imputation_model=LEAKAGE_IMPUTATION, normalize_time_per_n_forecasts=100)
    return lb.sort_values("win_rate", ascending=False)


def headline(summaries: list[dict], model_name: str, fev_bench_dir: str, out_dir: str | None = None) -> dict:
    """Skill score, win rate and rank for SQL and MASE; with `out_dir`, also writes `leaderboard_<metric>.csv`."""
    out = {}
    for metric in ("SQL", "MASE"):
        lb = leaderboard(summaries, model_name, fev_bench_dir, metric)
        if out_dir is not None:
            lb.to_csv(os.path.join(out_dir, f"leaderboard_{metric}.csv"))
        out[metric] = {"skill_score": 100 * float(lb.loc[model_name, "skill_score"]),
                       "win_rate": 100 * float(lb.loc[model_name, "win_rate"]),
                       "rank": list(lb.index).index(model_name) + 1, "n_models": len(lb)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fev-bench", required=True, help="benchmarks/fev_bench of a clone of github.com/autogluon/fev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="weights", help="local folder or Hugging Face repo id")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="nom")
    a = ap.parse_args()

    from benchmarks.adapter import nom_forecast_fn
    from nom import NomForecaster
    fn = nom_forecast_fn(NomForecaster.from_pretrained(a.model, device=a.device))
    summaries = evaluate(fn, a.name, a.fev_bench)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summaries.json"), "w") as f:
        json.dump(summaries, f, indent=1, default=str)
    h = headline(summaries, a.name, a.fev_bench, a.out)
    with open(os.path.join(a.out, "headline.json"), "w") as f:
        json.dump(h, f, indent=1)
    for metric, v in h.items():
        print(f"fev-bench {metric}: skill {v['skill_score']:.2f}  win rate {v['win_rate']:.1f}  "
              f"rank {v['rank']}/{v['n_models']}")


if __name__ == "__main__":
    main()
