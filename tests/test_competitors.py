"""The competitor list and the adapters' plumbing (no model is downloaded here)."""
import os

import numpy as np
import pytest

from benchmarks import competitors

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def test_list_is_the_one_written_before_scoring():
    pytest.importorskip("yaml")
    c = competitors.load_list(os.path.join(ROOT, "benchmarks", "competitors.yaml"))
    assert [m["name"] for m in c["competitors"]] == ["Timestral-5M", "Timestral-28M", "Chronos-2", "TiRex",
                                                     "TimesFM-2.5", "Toto-2.0-22m"]
    for m in c["competitors"]:
        assert {"repo_id", "revision", "card_statement", "license", "package", "source"} <= set(m)
        assert len(m["revision"]) == 40
    assert set(competitors.LOADERS) == {m["name"] for m in c["competitors"]}


def test_gift_eval_override_for_tirex():
    pytest.importorskip("yaml")
    c = competitors.load_list(os.path.join(ROOT, "benchmarks", "competitors.yaml"))
    m = competitors.entry(c, "TiRex", benchmark="gift")
    assert m["repo_id"] == "NX-AI/TiRex-1.1-gifteval" and m["revision"].startswith("01b1ebc")
    assert competitors.entry(c, "TiRex", benchmark="fev")["repo_id"] == "NX-AI/TiRex"
    assert competitors.entry(c, "Chronos-2", benchmark="gift")["repo_id"] == "amazon/chronos-2"


def test_left_pad():
    x, observed = competitors.left_pad([np.array([1.0, 2.0], np.float32), np.array([3.0, np.nan, 4.0], np.float32)])
    np.testing.assert_array_equal(x, np.array([[np.nan, 1, 2], [3, np.nan, 4]], np.float32))
    np.testing.assert_array_equal(observed, [[False, True, True], [True, False, True]])


def test_level_indices():
    levels = [0.01, 0.05] + [round(0.05 * k, 2) for k in range(2, 19)] + [0.95, 0.99]
    idx = competitors.level_indices(levels)
    assert [levels[i] for i in idx] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    with pytest.raises(ValueError, match="0.1"):
        competitors.level_indices([0.2, 0.5, 0.8])


def test_empty_contexts_become_a_single_zero_for_packages_that_refuse_them():
    a, b = np.array([1.0, np.nan], np.float32), np.full(5, np.nan, np.float32)
    out = competitors.no_empty([a, b])
    assert out[0] is a and out[1].tolist() == [0.0]


def test_tirex_resampling_variant_is_a_second_line():
    assert competitors.result_name("TiRex", None) == "TiRex"
    assert competitors.result_name("TiRex", "resample") == "TiRex (resampling)"
    with pytest.raises(ValueError, match="no variant"):
        competitors.result_name("Chronos-2", "resample")


def test_result_folders_per_benchmark():
    assert competitors.out_dir("gift", "TiRex").endswith("results/gift_competitors/TiRex")
    assert competitors.out_dir("fev", "Chronos-2").endswith("results/fev_competitors/Chronos-2")
    assert competitors.out_dir("time", "Timestral-5M").endswith("results/time_competitors/Timestral-5M")


def test_univariate_runs_on_fev_and_time_are_listed():
    pytest.importorskip("yaml")
    c = competitors.load_list(os.path.join(ROOT, "benchmarks", "competitors.yaml"))
    assert c["fev_time_univariate_runs"] == ["Chronos-2", "Toto-2.0-22m", "Timestral-5M", "Timestral-28M"]
    assert c["natively_univariate"] == ["TiRex", "TimesFM-2.5"]
