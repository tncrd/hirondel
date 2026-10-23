"""Leaderboard submission bundles, generated from this repository's results (nothing is sent anywhere)."""
import csv
import json
import os

import pytest

from submissions import build

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ARGS = dict(display="Hirondel", hf_url="https://huggingface.co/Tncrd/Hirondel-7M", code_url="https://github.com/tncrd/hirondel",
            org="The Hirondel Authors", git_ref="v1.0.0")


def test_gift_bundle(tmp_path):
    out = build.gift(str(tmp_path), **ARGS)
    rows = list(csv.DictReader(open(os.path.join(out, "all_results.csv"))))
    assert len(rows) == 97 and {r["model"] for r in rows} == {"Hirondel"}
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["model"] == "Hirondel" and cfg["testdata_leakage"] == "No" and cfg["model_dtype"] == "float32"
    assert cfg["model_type"] in ("pretrained", "zero-shot") and cfg["model_link"] == ARGS["hf_url"]


def test_fev_bundle(tmp_path):
    pytest.importorskip("yaml")
    out = build.fev(str(tmp_path), **ARGS)
    rows = list(csv.DictReader(open(os.path.join(out, "benchmarks", "fev_bench", "results", "hirondel.csv"))))
    assert len(rows) == 100 and {r["model_name"] for r in rows} == {"Hirondel"}
    assert all(r["trained_on_this_dataset"] in ("False", "false") for r in rows)
    req = open(os.path.join(out, "models", "hirondel", "requirements.txt")).read()
    assert "git+https://github.com/tncrd/hirondel" in req and "@v1.0.0" in req
    src = open(os.path.join(out, "models", "hirondel", "model.py")).read()
    compile(src, "model.py", "exec")
    assert 'model_name = "hirondel"' in src and "trained_on_datasets = []" in src
    entry = open(os.path.join(out, "models_yaml_entry.yaml")).read()
    assert "commercial_use: true" in entry and "zero_shot: true" in entry


def test_time_bundle_instructions(tmp_path):
    out = build.time(str(tmp_path), results_root=None, **ARGS)
    text = open(os.path.join(out, "SUBMIT.md")).read()
    assert "Real-TSF/TIME-Output" in text and "create_pr=True" in text
