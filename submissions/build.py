"""Leaderboard submission bundles from this repository's results. Nothing is sent: each bundle holds the files and the
steps of one leaderboard's official procedure (read on 2026-10-22).

    python -m submissions.build --display Nom --hf-url https://huggingface.co/<user>/<repo> \
        --code-url https://github.com/<user>/<repo> --org "<name>" --git-ref v1.0 [--time-results <folder>]

- GIFT-Eval: pull request to github.com/SalesforceAIResearch/gift-eval adding results/<model>/{all_results.csv,
  config.json} (gift-eval README, "Submission").
- fev-bench: pull request to github.com/autogluon/fev adding models/<name>/{model.py, requirements.txt}, the results CSV
  benchmarks/fev_bench/results/<name>.csv and an entry in benchmarks/fev_bench/models.yaml (fev models/README.md,
  "Submit results"); the maintainers reproduce the results before listing them.
- TIME: pull request on huggingface.co/datasets/Real-TSF/TIME-Output uploading output/results/<model>/ (TIME README,
  "Submit Results to TIME Leaderboard").
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "benchmarks", "results")


def _package() -> str:
    import nom as pkg                                       # the installed package (renamed with tools/rename.py)
    return pkg.__name__


def gift(out_root: str, display: str, hf_url: str, code_url: str, org: str, git_ref: str) -> str:
    out = os.path.join(out_root, "gift-eval", "results", display)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(RESULTS, "gift", "all_results.csv")) as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(out, "all_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "model": display})
    config = {"model": display, "model_type": "pretrained", "model_dtype": "float32", "model_link": hf_url,
              "code_link": f"{code_url}/blob/{git_ref}/benchmarks/gift.py", "org": org, "testdata_leakage": "No",
              "replication_code_available": "Yes"}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    return out


WRAPPER = '''import datasets
import numpy as np

import fev


class {cls}Model(fev.ForecastingModel):
    """{display} from {code_url} (weights: {hf_url}).

    Univariate and covariate-blind: every target column is forecast as its own series, covariate columns are not read.
    Default inference procedure of the package (flip average, zero clipping of non-negative series). A context with no observed value, which the package's
    `forecast()` refuses, is run through the model on its empty context (as in the package's benchmark adapter).
    """

    model_name = "{name}"
    trained_on_datasets = []

    def __init__(self, model_path: str = "{hf_id}", device: str = "auto", batch_size: int = 256):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self._forecaster = None

    def _load(self):
        import torch
        from {package} import {cls}Forecaster

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._forecaster is None:
            self._forecaster = {cls}Forecaster.from_pretrained(self.model_path, device=self.device)
        return self._forecaster

    def _forecast(self, series, horizon):
        f = self._forecaster
        empty = np.array([not np.isfinite(s).any() for s in series], bool)
        out = np.empty((len(series), horizon, 9), np.float32)
        if (~empty).any():
            out[~empty] = f.forecast([s for s, e in zip(series, empty) if not e], horizon).quantiles
        if empty.any():
            out[empty] = f._forecast_batch([s for s, e in zip(series, empty) if e], horizon, f.cfg.flip)
        return out

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        self._load()
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        predictions_per_window = []
        for window in task.iter_windows():
            past = fev.convert_input_data(window, adapter="pandas", as_univariate=True)[0]
            series = [g["target"].to_numpy(np.float32) for _, g in past.groupby("id", sort=False)]
            with self._record_inference_time():
                q = np.concatenate([self._forecast(series[i:i + self.batch_size], task.horizon)
                                    for i in range(0, len(series), self.batch_size)])
            idx = [levels.index(round(float(x), 2)) for x in task.quantile_levels]
            preds = {{"predictions": q[..., levels.index(0.5)].tolist()}}
            for j, level in zip(idx, task.quantile_levels):
                preds[str(level)] = q[..., j].tolist()
            predictions_per_window.append(fev.utils.combine_univariate_predictions_to_multivariate(
                datasets.Dataset.from_dict(preds), target_columns=task.target_columns))
        return predictions_per_window
'''


def fev(out_root: str, display: str, hf_url: str, code_url: str, org: str, git_ref: str) -> str:
    package = _package()
    name = package.lower()
    cls = display.replace("-", "").replace(" ", "")
    out = os.path.join(out_root, "fev")
    model_dir = os.path.join(out, "models", name)
    res_dir = os.path.join(out, "benchmarks", "fev_bench", "results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    hf_id = hf_url.split("huggingface.co/")[-1]
    with open(os.path.join(model_dir, "model.py"), "w") as f:
        f.write(WRAPPER.format(cls=cls, display=display, code_url=code_url, hf_url=hf_url, name=name, hf_id=hf_id,
                               package=package))
    with open(os.path.join(model_dir, "requirements.txt"), "w") as f:
        f.write(f"{package} @ git+{code_url}.git@{git_ref}\n")
    with open(os.path.join(RESULTS, "fev", "summaries.json")) as f:
        summaries = json.load(f)
    fields = list(summaries[0]) + ["model_class", "model_kwargs"]
    with open(os.path.join(res_dir, f"{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            w.writerow({**{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in s.items()},
                        "model_name": display, "model_class": name, "model_kwargs": "{}"})
    with open(os.path.join(out, "models_yaml_entry.yaml"), "w") as f:
        f.write(f"{name}:\n  organization: {org}\n  url: {hf_url}\n  model_type: pretrained\n  zero_shot: true\n"
                f"  commercial_use: true\n")
    return out


def time(out_root: str, display: str, hf_url: str, code_url: str, org: str, git_ref: str,
         results_root: str | None) -> str:
    out = os.path.join(out_root, "time")
    os.makedirs(out, exist_ok=True)
    if results_root:
        shutil.copytree(results_root, os.path.join(out, "results", display), dirs_exist_ok=True)
    with open(os.path.join(out, "SUBMIT.md"), "w") as f:
        f.write(f"""# TIME submission ({display})

`results/{display}/` holds TIME's own per-window outputs for the 98 tasks (written by `timebench`'s
`save_window_predictions`, see `benchmarks/time_bench.py`). TIME README, "Submit Results to TIME Leaderboard":

```python
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="results/{display}", path_in_repo="results/{display}",
                      repo_id="Real-TSF/TIME-Output", repo_type="dataset",
                      commit_message="Submit evaluation results for {display}", create_pr=True)
```

The results are included in the leaderboard after review. Replication code: {code_url}/blob/{git_ref}/benchmarks/time_bench.py
""")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--display", required=True)
    ap.add_argument("--hf-url", required=True)
    ap.add_argument("--code-url", required=True)
    ap.add_argument("--org", required=True)
    ap.add_argument("--git-ref", default="v1.0")
    ap.add_argument("--time-results", help="output/results/<name> folder written by `benchmarks.time_bench score`")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args()
    kw = dict(display=a.display, hf_url=a.hf_url, code_url=a.code_url, org=a.org, git_ref=a.git_ref)
    for p in (gift(a.out, **kw), fev(a.out, **kw), time(a.out, results_root=a.time_results, **kw)):
        print("->", p)


if __name__ == "__main__":
    main()
