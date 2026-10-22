"""Showcase generation: Pareto front, rendering rules, traceability of every number. No leaderboard data is read here."""
import os
import re

import pytest

from showcase import build

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def test_pareto_front_known_answer():
    pts = [("a", 5, 0.80), ("b", 10, 0.70), ("c", 20, 0.75), ("d", 50, 0.60), ("e", 50, 0.65)]
    assert [p[0] for p in build.pareto_front(pts)] == ["a", "b", "d"]


def test_front_steps_are_a_staircase():
    xs, ys = build.front_steps([("a", 5, 0.8), ("b", 10, 0.7)], x_max=100)
    assert xs == [5, 10, 10, 100] and ys == [0.8, 0.8, 0.7, 0.7]


def test_fev_error_is_one_minus_skill():
    assert build.fev_error(32.8) == pytest.approx(0.672)


def test_fill_template_replaces_every_marker():
    out = build.fill("a <!-- gen:x --> b <!-- gen:y -->", {"x": "1", "y": "2"})
    assert out == "a 1 b 2"
    with pytest.raises(KeyError, match="z"):
        build.fill("<!-- gen:z -->", {})


def test_showcase_list_is_well_formed():
    pytest.importorskip("yaml")
    lst = build.load_models()
    names = [m["name"] for m in lst["models"]]
    assert len(names) == len(set(names))
    for m in lst["models"]:
        assert m["params"] > 0 and m["size_source"].startswith("https://") and isinstance(m["permissive"], bool)
        assert any(m[k] for k in ("gift", "fev", "time"))


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "showcase", "data", "official.csv")),
                    reason="showcase not generated yet")
def test_every_number_in_the_readme_table_is_traced():
    """The results table of the README only shows values present in showcase/data/official.csv."""
    import csv
    with open(os.path.join(ROOT, "showcase", "data", "official.csv")) as f:
        rows = list(csv.DictReader(f))
    assert all(r["source"] and r["date"] for r in rows)
    traced = {f"{float(r['value']):.4f}" for r in rows} | {f"{float(r['value']):.2f}" for r in rows}
    readme = open(os.path.join(ROOT, "README.md")).read()
    table = readme.split("<!-- results:start -->")[1].split("<!-- results:end -->")[0]
    for num in re.findall(r"\b\d+\.\d{2,4}\b", table):
        assert num in traced, num


def test_no_internal_names_in_generated_text():
    for path in ("README.md", "MODEL_CARD.md"):
        p = os.path.join(ROOT, path)
        if os.path.exists(p):
            text = open(p).read()
            assert not re.search(r"v11|_s0\b|tsfm|docs/research|registre", text), path
