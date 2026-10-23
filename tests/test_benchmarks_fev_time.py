"""fev-bench and TIME plumbing on synthetic files: task lists, official summaries, prediction layout, the TIME bridge
and its headline. No benchmark data is read; the optional dependencies skip the tests that need them."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import fev_bench, time_bench

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


# ---- fev-bench -------------------------------------------------------------------------------------------------------
def _fev_dir(tmp_path, n_tasks=100):
    yaml = pytest.importorskip("yaml")
    d = tmp_path / "fev_bench"
    (d / "results").mkdir(parents=True)
    with open(d / "tasks.yaml", "w") as f:
        yaml.safe_dump({"tasks": [{"dataset_config": f"t{i}", "horizon": 4} for i in range(n_tasks)]}, f)
    with open(d / "models.yaml", "w") as f:
        yaml.safe_dump({"a": {"model_type": "pretrained"}, "b": {"model_type": "closed-api"},
                        "c": {"model_type": "pretrained", "hidden": True}, "d": {"model_type": "statistical"}}, f)
    for k in "abcd":
        with open(d / "results" / f"{k}.csv", "w") as f:
            f.write("model_name,task_name,SQL\n" + f"{k},t0,1.0\n")
    return str(d)


def test_fev_tasks_are_the_official_100(tmp_path):
    assert len(fev_bench.tasks(_fev_dir(tmp_path))) == 100


def test_fev_tasks_refuse_a_drifted_list(tmp_path):
    with pytest.raises(ValueError, match="99 tasks, expected 100"):
        fev_bench.tasks(_fev_dir(tmp_path, 99))


def test_fev_official_summaries_are_the_models_shown_by_default(tmp_path):
    pytest.importorskip("pandas")
    s = fev_bench.official_summaries(_fev_dir(tmp_path))
    assert sorted(s["model_name"].unique()) == ["a", "d"]                 # no closed-api, no hidden model


def test_fev_prediction_columns():
    q = np.arange(2 * 3 * 9, dtype=np.float32).reshape(2, 3, 9)
    cols = fev_bench.prediction_columns(q)
    assert list(cols) == ["predictions"] + [str(x) for x in fev_bench.LEVELS]
    assert cols["predictions"] == q[..., 4].tolist() and cols["0.9"] == q[..., 8].tolist()


# ---- TIME ------------------------------------------------------------------------------------------------------------
def test_time_bridge_does_not_import_torch():
    """The dump and score stages run in TIME's own environment, which has no torch."""
    code = "import sys; import benchmarks.time_bench; assert 'torch' not in sys.modules and 'hirondel' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_time_pack_roundtrip():
    series = [np.arange(5, dtype=np.float32), np.array([], np.float32), np.array([np.nan, 1.0], np.float32)]
    flat, offsets = time_bench.pack(series)
    back = time_bench.unpack(flat, offsets)
    assert len(back) == 3 and all(np.array_equal(a, b, equal_nan=True) for a, b in zip(series, back))


def _metrics(root, model, dataset_id, horizon, mase, crps):
    d = os.path.join(root, model, dataset_id, horizon)
    os.makedirs(d)
    np.savez(os.path.join(d, "metrics.npz"), MASE=np.asarray(mase, float), CRPS=np.asarray(crps, float))


def test_time_headline_known_answer(tmp_path):
    """Per task: mean over windows / series (NaN ignored), divided by Seasonal Naive's; geometric mean over tasks."""
    ours, naive = str(tmp_path / "ours"), str(tmp_path / "naive")
    _metrics(ours, "m", "A/H", "short", [1.0, 3.0, np.nan], [0.5, 0.5])
    _metrics(naive, "seasonal_naive", "A/H", "short", [4.0], [1.0])
    _metrics(ours, "m", "B/D", "long", [2.0], [0.2])
    _metrics(naive, "seasonal_naive", "B/D", "long", [1.0], [0.8])
    _metrics(naive, "seasonal_naive", "C/D", "short", [1.0], [1.0])           # no result of ours: not counted
    h = time_bench.headline(ours, "m", naive)
    assert h["n_tasks"] == 2
    assert abs(h["MASE"] - math.sqrt(0.5 * 2.0)) < 1e-12
    assert abs(h["CRPS"] - math.sqrt(0.5 * 0.25)) < 1e-12


def test_time_results_files(tmp_path):
    ours, naive, out = str(tmp_path / "ours"), str(tmp_path / "naive"), str(tmp_path / "out")
    _metrics(ours, "m", "A/H", "short", [1.0, 3.0], [0.5, 0.5])
    _metrics(naive, "seasonal_naive", "A/H", "short", [4.0], [1.0])
    h = time_bench.write_results(ours, "m", naive, out)
    rows = open(os.path.join(out, "tasks.csv")).read().splitlines()
    assert rows[0] == "task,MASE,CRPS,MASE_norm,CRPS_norm" and rows[1].startswith("A/H/short,2.0,0.5,0.5,0.5")
    assert json.load(open(os.path.join(out, "headline.json"))) == h and h["n_tasks"] == 1


def test_gift_naive_folder_in_old_and_new_clones(tmp_path):
    """gift-eval renamed results/seasonal_naive to results/Seasonal_Naive (2026-10); both clones are read."""
    from benchmarks import gift
    for name in ("seasonal_naive", "Seasonal_Naive"):
        root = tmp_path / name / "gift-eval"
        (root / "results" / name).mkdir(parents=True)
        assert gift.naive_csv(str(root)) == str(root / "results" / name / "all_results.csv")


def test_time_score_keeps_the_run_provenance(tmp_path):
    """The forecast stage of a competitor writes its provenance (package, revision, date) to headline.json; the score
    stage adds the metrics to it instead of overwriting it."""
    ours, naive, out = str(tmp_path / "ours"), str(tmp_path / "naive"), str(tmp_path / "out")
    _metrics(ours, "m", "A/H", "short", [1.0, 3.0], [0.5, 0.5])
    _metrics(naive, "seasonal_naive", "A/H", "short", [4.0], [1.0])
    os.makedirs(out)
    json.dump({"model": "m", "revision": "abc", "MASE": "stale"}, open(os.path.join(out, "headline.json"), "w"))
    h = time_bench.write_results(ours, "m", naive, out)
    saved = json.load(open(os.path.join(out, "headline.json")))
    assert saved["revision"] == "abc" and saved["model"] == "m"
    assert saved["MASE"] == h["MASE"] and saved["n_tasks"] == 1
