"""GIFT-Eval (97 tasks) with the official harness: `gift_eval.data.Dataset` + `gluonts.model.evaluate_model`, the metric
list, dataset naming and output columns of the official notebooks (github.com/SalesforceAIResearch/gift-eval,
`notebooks/*.ipynb`), and the leaderboard headline: geometric mean over the 97 tasks of MASE[0.5] and
mean_weighted_sum_quantile_loss (CRPS), each divided by Seasonal Naive's value on the same task (`results/seasonal_naive/`, `results/Seasonal_Naive/` in recent clones).

    git clone https://github.com/SalesforceAIResearch/gift-eval && pip install -e gift-eval ".[bench]"
    export GIFT_EVAL=<folder of the Salesforce/GiftEval dataset>
    python -m benchmarks.gift --gift-eval gift-eval --out results/gift [--model Org/Nom-7M] [--device cuda]

Writes `<out>/all_results.csv` (leaderboard format, one row per task) and prints the headline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

# Verbatim from the official notebooks (`SHORT_DATASETS`, `MED_LONG_DATASETS`): 55 short + 21 medium + 21 long = 97.
SHORT_DATASETS = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D "
    "electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W "
    "saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H "
    "kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D "
    "ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T "
    "bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T "
    "bizitobs_l2c/H"
).split()
MED_LONG_DATASETS = (
    "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T "
    "bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
).split()
PRETTY_NAMES = {"saugeenday": "saugeen", "temperature_rain_with_missing": "temperature_rain",
                "kdd_cup_2018_with_missing": "kdd_cup_2018", "car_parts_with_missing": "car_parts"}

MASE_COL, CRPS_COL = "eval_metrics/MASE[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss"
METRIC_NAMES = ("MSE[mean]", "MSE[0.5]", "MAE[0.5]", "MASE[0.5]", "MAPE[0.5]", "sMAPE[0.5]", "MSIS", "RMSE[mean]",
                "NRMSE[mean]", "ND[0.5]", "mean_weighted_sum_quantile_loss")
COLUMNS = ["dataset", "model"] + [f"eval_metrics/{m}" for m in METRIC_NAMES] + ["domain", "num_variates"]


def tasks() -> list[tuple[str, str]]:
    """The 97 (dataset, term) pairs of the leaderboard."""
    return [(d, "short") for d in SHORT_DATASETS] + [(d, t) for t in ("medium", "long") for d in MED_LONG_DATASETS]


def task_key(name: str, term: str, properties: dict) -> str:
    """Leaderboard key "<pretty name>/<freq>/<term>", as the official notebooks write it."""
    base, _, freq = name.partition("/")
    key = PRETTY_NAMES.get(base.lower(), base.lower())
    return f"{key}/{freq or properties[key]['frequency']}/{term}"


def official_metrics():
    from gluonts.ev.metrics import MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE, MeanWeightedSumQuantileLoss
    return [MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(), MSIS(), RMSE(), NRMSE(),
            ND(), MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])]


def done_tasks(out_csv: str) -> set:
    """Tasks already scored in `out_csv` (empty if the file does not exist)."""
    if not os.path.exists(out_csv):
        return set()
    with open(out_csv) as f:
        return {r["dataset"] for r in csv.DictReader(f)}


def evaluate(make_predictor, model_name: str, properties: dict, out_csv: str, only=None,
             resume: bool = False) -> list[dict]:
    """Run every task. `make_predictor(prediction_length)` returns a GluonTS predictor (e.g. a `GluonTSAdapter`).
    Rows are appended to `out_csv` as they come, so an interrupted run keeps what it scored; with `resume`, the tasks
    already in `out_csv` are skipped and new rows are appended."""
    from gift_eval.data import Dataset
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality
    metrics = official_metrics()
    rows = []
    skip = done_tasks(out_csv) if resume else set()
    with open(out_csv, "a" if skip else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not skip:
            writer.writeheader()
        for name, term in tasks():
            if only is not None and name not in only:
                continue
            key = task_key(name, term, properties)
            if key in skip:
                continue
            to_univariate = Dataset(name=name, term=term, to_univariate=False).target_dim != 1
            dataset = Dataset(name=name, term=term, to_univariate=to_univariate)
            t0 = time.time()
            res = evaluate_model(make_predictor(dataset.prediction_length), test_data=dataset.test_data,
                                 metrics=metrics, batch_size=1024, axis=None, mask_invalid_label=True,
                                 allow_nan_forecast=False, seasonality=get_seasonality(dataset.freq))
            prop = properties[key.split("/")[0]]
            row = {"dataset": key, "model": model_name, "domain": prop["domain"], "num_variates": prop["num_variates"],
                   **{f"eval_metrics/{m}": float(res[m].iloc[0]) for m in METRIC_NAMES}}
            writer.writerow(row)
            f.flush()
            rows.append(row)
            print(f"{key:40s} MASE {row[MASE_COL]:.4f}  CRPS {row[CRPS_COL]:.4f}  ({time.time() - t0:.0f} s)", flush=True)
    return rows


def _read(path: str) -> dict:
    with open(path) as f:
        return {r["dataset"]: (float(r[MASE_COL]), float(r[CRPS_COL])) for r in csv.DictReader(f)}


def naive_csv(gift_eval: str) -> str:
    """Seasonal Naive's results in a gift-eval clone: results/seasonal_naive, renamed results/Seasonal_Naive in 2026-10."""
    for name in ("seasonal_naive", "Seasonal_Naive"):
        p = os.path.join(gift_eval, "results", name, "all_results.csv")
        if os.path.exists(os.path.dirname(p)):
            return p
    raise FileNotFoundError(f"no Seasonal Naive results under {gift_eval}/results")


def headline(results_csv: str, naive_csv: str, n_tasks: int = 97) -> dict:
    """Leaderboard headline: geometric mean over the tasks of (model / Seasonal Naive), for MASE and CRPS."""
    ours, naive = _read(results_csv), _read(naive_csv)
    common = sorted(set(ours) & set(naive))
    if len(common) != n_tasks:
        raise ValueError(f"results cover {len(common)} of {n_tasks} tasks")
    out = {"n_tasks": len(common)}
    for j, name in enumerate(("MASE", "CRPS")):
        out[name] = math.exp(sum(math.log(ours[k][j] / naive[k][j]) for k in common) / len(common))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gift-eval", required=True, help="clone of github.com/SalesforceAIResearch/gift-eval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="weights", help="local folder or Hugging Face repo id")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--no-flip", action="store_true")
    a = ap.parse_args()

    from benchmarks.adapter import GluonTSAdapter, nom_forecast_fn
    from nom import NomForecaster
    with open(os.path.join(a.gift_eval, "notebooks", "dataset_properties.json")) as f:
        properties = json.load(f)
    forecaster = NomForecaster.from_pretrained(a.model, device=a.device, batch_size=a.batch_size)
    fn = nom_forecast_fn(forecaster, flip=False if a.no_flip else None)
    os.makedirs(a.out, exist_ok=True)
    out_csv = os.path.join(a.out, "all_results.csv")
    evaluate(lambda pl: GluonTSAdapter(fn, pl, batch_size=a.batch_size), "nom", properties, out_csv)
    h = headline(out_csv, naive_csv(a.gift_eval))
    print(f"\nGIFT-Eval ({h['n_tasks']} tasks): MASE {h['MASE']:.4f}  CRPS {h['CRPS']:.4f}")


if __name__ == "__main__":
    main()
