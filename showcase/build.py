"""Regenerate the showcase from the results: data tables with sources, the accuracy / size figures, the banner, README.md
and MODEL_CARD.md.

    python -m showcase.build --gift-eval gift-eval --fev-bench fev/benchmarks/fev_bench \
        --time-output TIME-Output/results --time-naive TIME-Output/results [--model weights]

Inputs: clones of the leaderboards' public results (gift-eval, fev, Real-TSF/TIME-Output), this repository's benchmark
results (benchmarks/results/) and the model list (showcase/models.yaml). The output is deterministic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "benchmarks", "results")
DATE = "2026-10-22"
SOURCES = {
    "GIFT-Eval": "github.com/SalesforceAIResearch/gift-eval @ 6fdb10d, results/{key}/all_results.csv",
    "fev-bench": "github.com/autogluon/fev @ f9e2c7d, benchmarks/fev_bench/results",
    "TIME": "huggingface.co/datasets/Real-TSF/TIME-Output @ 4c93dea, results/{key}",
}
OURS_SOURCES = {"GIFT-Eval": "benchmarks/results/gift/all_results.csv", "fev-bench": "benchmarks/results/fev/summaries.json",
                "TIME": "benchmarks/results/time/tasks.csv"}
# (benchmark, metric shown, axis label)
PANELS = [("GIFT-Eval", "MASE", "MASE"), ("TIME", "MASE", "MASE"), ("fev-bench", "MASE", "MASE (1 - skill)"),
          ("GIFT-Eval", "CRPS", "CRPS (WQL)"), ("TIME", "CRPS", "CRPS (WQL)"), ("fev-bench", "SQL", "SQL (1 - skill)")]


# ---- small pure helpers (tested) -------------------------------------------------------------------------------------
def pareto_front(points):
    """(name, size, error) -> the points no other point beats on both size and error, sorted by size."""
    front, best = [], math.inf
    for p in sorted(points, key=lambda p: (p[1], p[2])):
        if p[2] < best:
            front.append(p)
            best = p[2]
    return front


def front_steps(front, x_max):
    """Staircase through a Pareto front: the best error reachable at each size, up to x_max."""
    xs, ys = [], []
    for i, (_, x, y) in enumerate(front):
        if i:
            xs.append(x)
            ys.append(ys[-1])
        xs.append(x)
        ys.append(y)
    xs.append(x_max)
    ys.append(ys[-1])
    return xs, ys


def fev_error(skill_percent: float) -> float:
    """fev-bench skill score (percent) -> the geometric-mean relative error it summarises (lower is better)."""
    return 1 - skill_percent / 100


def fill(template: str, values: dict) -> str:
    import re
    return re.sub(r"<!-- gen:(\w+) -->", lambda m: values[m.group(1)], template)


def load_models(path: str = os.path.join(HERE, "models.yaml")) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ---- official numbers ------------------------------------------------------------------------------------------------
def gift_scores(gift_eval: str, key: str) -> dict:
    from benchmarks.gift import headline
    h = headline(os.path.join(gift_eval, "results", key, "all_results.csv"),
                 os.path.join(gift_eval, "results", "seasonal_naive", "all_results.csv"))
    return {"MASE": h["MASE"], "CRPS": h["CRPS"]}


def fev_skills(fev_bench: str) -> dict:
    """{model_name: {"MASE": skill %, "SQL": skill %}} for every published model, hidden ones included (the skill score
    does not depend on which other models are shown)."""
    import glob
    import fev
    import pandas as pd
    from benchmarks.fev_bench import BASELINE, LEAKAGE_IMPUTATION
    summaries = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(fev_bench, "results", "*.csv")))],
                          ignore_index=True)
    out = {}
    for metric in ("MASE", "SQL"):
        lb = fev.analysis.leaderboard(summaries=summaries, metric_column=metric, missing_strategy="impute",
                                      baseline_model=BASELINE, leakage_imputation_model=LEAKAGE_IMPUTATION,
                                      normalize_time_per_n_forecasts=100)
        for name, row in lb.iterrows():
            out.setdefault(name, {})[metric] = 100 * float(row["skill_score"])
    return out


def time_scores(time_output: str, time_naive: str, key: str) -> dict:
    from benchmarks.time_bench import headline
    h = headline(time_output, key, time_naive)
    if h["n_tasks"] != 98:
        raise ValueError(f"TIME results of {key}: {h['n_tasks']} tasks, expected 98")
    return {"MASE": h["MASE"], "CRPS": h["CRPS"]}


def ours_scores() -> dict:
    from benchmarks.gift import headline
    g = headline(os.path.join(RESULTS, "gift", "all_results.csv"),
                 os.path.join(os.environ.get("GIFT_NAIVE", ""), "all_results.csv")) if os.environ.get("GIFT_NAIVE") else None
    with open(os.path.join(RESULTS, "fev", "headline.json")) as f:
        fv = json.load(f)
    with open(os.path.join(RESULTS, "time", "headline.json")) as f:
        tm = json.load(f)
    return {"GIFT-Eval": g, "fev-bench": {"MASE": fv["MASE"]["skill_score"], "SQL": fv["SQL"]["skill_score"],
                                          "rank_MASE": fv["MASE"]["rank"], "rank_SQL": fv["SQL"]["rank"],
                                          "n_models": fv["SQL"]["n_models"]},
            "TIME": {"MASE": tm["MASE"], "CRPS": tm["CRPS"]}}


def collect(gift_eval: str, fev_bench: str, time_output: str, time_naive: str) -> list[dict]:
    """Rows {model, params, license, permissive, benchmark, metric, value, source, date}; fev values are skill scores (%)."""
    lst = load_models()
    os.environ["GIFT_NAIVE"] = os.path.join(gift_eval, "results", "seasonal_naive")
    rows = []
    ours = ours_scores()
    o = lst["ours"]
    for bench, vals in ours.items():
        for metric in ("MASE", "CRPS", "SQL"):
            if metric in vals:
                rows.append({"model": o["name"], "params": o["params"], "license": o["license"], "permissive": True,
                             "benchmark": bench, "metric": metric, "value": vals[metric],
                             "source": f"this repository, {OURS_SOURCES[bench]}", "date": DATE})
    skills = fev_skills(fev_bench)
    for m in lst["models"]:
        base = {"model": m["name"], "params": m["params"], "license": m["license"], "permissive": m["permissive"],
                "date": DATE}
        if m["gift"]:
            for metric, v in gift_scores(gift_eval, m["gift"]).items():
                rows.append({**base, "benchmark": "GIFT-Eval", "metric": metric, "value": v,
                             "source": SOURCES["GIFT-Eval"].format(key=m["gift"])})
        if m["fev"]:
            for metric in ("MASE", "SQL"):
                rows.append({**base, "benchmark": "fev-bench", "metric": metric, "value": skills[m["fev"]][metric],
                             "source": SOURCES["fev-bench"]})
        if m["time"]:
            for metric, v in time_scores(time_output, time_naive, m["time"]).items():
                rows.append({**base, "benchmark": "TIME", "metric": metric, "value": v,
                             "source": SOURCES["TIME"].format(key=m["time"])})
    return rows


def same_protocol() -> list[dict]:
    """GIFT-Eval, forced univariate, computed by us: our model and benchmarks/competitors.yaml."""
    rows = []
    from benchmarks.gift import headline  # noqa: F401  (ours comes from the official table)
    for d in sorted(os.listdir(os.path.join(RESULTS, "gift_competitors"))):
        p = os.path.join(RESULTS, "gift_competitors", d, "headline.json")
        if os.path.exists(p):
            with open(p) as f:
                h = json.load(f)
            rows.append({"model": h["model"], "MASE": h["MASE"], "CRPS": h["CRPS"], "package": h["package"],
                         "package_version": h["package_version"], "revision": h["revision"][:7], "date": h["date"]})
    return rows


def univariate_rows(official: list[dict], computed: dict, natively_univariate: set, ours: str,
                    meta: dict | None = None) -> list[dict]:
    """The univariate setting (each series forecast alone, no covariates): our numbers; a competitor's forced-univariate
    run computed by us when it exists; otherwise its official number only if the model is univariate by design."""
    out = []
    for r in official:
        key = (r["model"], r["benchmark"], r["metric"])
        if r["model"] == ours:
            out.append(dict(r))
        elif key in computed:
            v, src = computed[key]
            out.append({**r, "value": v, "source": src})
        elif r["model"] in natively_univariate:
            out.append({**r, "source": r["source"] + " (univariate by design)"})
    covered = {(r["model"], r["benchmark"], r["metric"]) for r in official}
    for (model, bench, metric), (v, src) in sorted(computed.items()):
        if (model, bench, metric) not in covered and model in (meta or {}):
            m = meta[model]
            out.append({"model": model, "params": m["params"], "license": m["license"], "permissive": m["permissive"],
                        "benchmark": bench, "metric": metric, "value": v, "source": src, "date": DATE})
    return out


def computed_univariate() -> dict:
    """{(model, benchmark, metric): (value, source)} from benchmarks/results/{gift,fev,time}_competitors."""
    out = {}
    for bench, folder in (("GIFT-Eval", "gift_competitors"), ("fev-bench", "fev_competitors"), ("TIME", "time_competitors")):
        root = os.path.join(RESULTS, folder)
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d, "headline.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                h = json.load(f)
            src = f"computed by us, forced univariate (benchmarks/results/{folder}/{d})"
            metrics = ("MASE", "SQL") if bench == "fev-bench" else ("MASE", "CRPS")
            for m in metrics:
                if m in h and isinstance(h[m], (int, float)):
                    out[(d, bench, m)] = (float(h[m]), src)
    return out


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gift-eval", required=True)
    ap.add_argument("--fev-bench", required=True)
    ap.add_argument("--time-output", required=True, help="results/ folder of Real-TSF/TIME-Output")
    ap.add_argument("--time-naive", required=True, help="folder holding seasonal_naive/ of Real-TSF/TIME-Output")
    ap.add_argument("--model", default=os.path.join(ROOT, "weights"))
    ap.add_argument("--data-only", action="store_true")
    a = ap.parse_args()
    rows = collect(a.gift_eval, a.fev_bench, a.time_output, a.time_naive)
    write_csv(os.path.join(HERE, "data", "official.csv"), rows)
    lst = load_models()
    meta = {m["name"]: m for m in lst["models"]}
    from benchmarks.competitors import load_list
    native = set(load_list(os.path.join(ROOT, "benchmarks", "competitors.yaml"))["natively_univariate"])
    uni = univariate_rows(rows, computed_univariate(), native, lst["ours"]["name"], meta)
    write_csv(os.path.join(HERE, "data", "univariate.csv"), uni)
    write_csv(os.path.join(HERE, "data", "same_protocol.csv"), same_protocol())
    print(f"{len(rows)} official values -> showcase/data/official.csv")
    if a.data_only:
        return
    from showcase import render
    render.all(rows, same_protocol(), ours_scores(), load_models(), a.model, uni)


if __name__ == "__main__":
    main()
