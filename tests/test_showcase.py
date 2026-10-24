"""Showcase generation: Pareto front, rendering rules, traceability of every number. No leaderboard data is read here."""
import os
import re

import pytest

from showcase import build, render

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


def test_univariate_rows_prefer_our_runs_and_keep_only_univariate_official():
    official = [
        {"model": "Hirondel", "params": 1, "license": "a", "permissive": True, "benchmark": "TIME", "metric": "MASE",
         "value": 0.65, "source": "this repository", "date": "d"},
        {"model": "X", "params": 2, "license": "a", "permissive": True, "benchmark": "TIME", "metric": "MASE",
         "value": 0.60, "source": "official", "date": "d"},      # joint on the leaderboard, our run replaces it
        {"model": "U", "params": 3, "license": "a", "permissive": True, "benchmark": "TIME", "metric": "MASE",
         "value": 0.70, "source": "official", "date": "d"},      # univariate by design: official kept
        {"model": "J", "params": 4, "license": "a", "permissive": True, "benchmark": "TIME", "metric": "MASE",
         "value": 0.50, "source": "official", "date": "d"},      # joint, no run of ours: left out
    ]
    computed = {("X", "TIME", "MASE"): (0.66, "computed by us")}
    rows = build.univariate_rows(official, computed, natively_univariate={"U"}, ours="Hirondel")
    got = {(r["model"], r["value"]) for r in rows}
    assert got == {("Hirondel", 0.65), ("X", 0.66), ("U", 0.70)}


def test_univariate_rows_include_our_runs_without_an_official_entry():
    meta = {"T": {"params": 5, "license": "b", "permissive": False}}
    rows = build.univariate_rows([], {("T", "GIFT-Eval", "MASE"): (0.71, "computed by us")}, set(), ours="Hirondel",
                                 meta=meta)
    assert rows == [{"model": "T", "params": 5, "license": "b", "permissive": False, "benchmark": "GIFT-Eval",
                     "metric": "MASE", "value": 0.71, "source": "computed by us", "date": build.DATE}]


def test_indeval_table_from_published_numbers_and_ours():
    pub = render.load_indeval()
    src9 = pub["sources"] + pub["shared_sources"]
    ours = {"per_source": {s: [0.9, 0.5] for s in src9}, "nMASE_nCRPS": {}}
    other = {**pub["computed"][0], "per_source": {s: [0.95, 0.55] for s in src9}}
    rows = render.indeval_rows(pub, ours, "Hirondel", [other])
    chronos = next(r for r in rows if r["model"] == "Chronos-2")
    assert chronos["mean"] == pytest.approx((0.822, 0.583), abs=5e-4)               # geometric mean of the five sources
    assert chronos["mean9"] == pytest.approx((0.755, 0.539), abs=5e-4)              # the paper's aggregate, nine sources
    assert [r["model"] for r in rows][:2] == ["Hirondel", "Timestral-5M"]
    assert next(r for r in rows if r["model"] == "Hirondel")["mean"] == pytest.approx((0.9, 0.5))
    table = render.indeval_table(rows, pub)
    assert "| 3W (oil wells) |" in table and "**0.473**" in table                               # TimesFM-3 best nCRPS on PRONTO
    assert "| **Geometric mean, 9 sources (the paper's aggregate)** |" in table and "Timestral-5M (5.2M)" in table


def test_indeval_univariate_table_lists_every_model_of_our_runs_and_leaves_the_first_table_alone():
    pub = render.load_indeval()
    src9 = pub["sources"] + pub["shared_sources"]
    ours = {"params": 7_462_049, "per_source": {s: [0.9, 0.5] for s in src9}}
    runs = [{**u, "per_source": {s: [0.8 + 0.01 * i, 0.6] for s in src9}} for i, u in enumerate(pub["univariate"])]
    assert [u["name"] for u in pub["univariate"]] == ["Timestral-5M", "Timestral-28M", "Chronos-2", "TiRex",
                                                      "TimesFM-2.5", "Toto-2.0-22m", "Toto-2.0-4m"]
    rows = render.indeval_univariate_rows(pub, ours, "Hirondel", runs)
    assert [r["params"] for r in rows] == sorted(r["params"] for r in rows)           # by size
    table = render.indeval_univariate_table(rows, pub)
    assert table.count("\n") == len(rows) + 1                                        # header, rule, one line per model
    assert "| **Hirondel (ours)** | 7.5M |" in table and "| TiRex | 35M |" in table
    assert "**0.800** / 0.600" in table and "0.900 / **0.500**" in table           # best of each metric in bold
    # the first table (our runs and the paper's published numbers) is built as before, without these runs
    first = render.indeval_rows(pub, ours, "Hirondel", [])
    assert [r["model"] for r in first] == ["Hirondel", "Chronos-2", "TiRex-2", "TimesFM-3"]


def test_indeval_bold_marks_every_model_tied_on_the_shown_value():
    pub = render.load_indeval()
    src9 = pub["sources"] + pub["shared_sources"]
    ours = {"params": 7_462_049, "per_source": {s: [0.7538, 0.5] for s in src9}}
    runs = [{**pub["univariate"][0], "per_source": {s: [0.7541, 0.6] for s in src9}}]
    table = render.indeval_table(render.indeval_univariate_rows(pub, ours, "Hirondel", runs), pub)
    assert table.count("**0.754**") == 2 * (len(src9) + 2)                   # both models, 9 sources and 2 means


def test_indeval_one_figure_with_both_settings_and_links_between_them(tmp_path):
    from PIL import Image
    pub = render.load_indeval()
    src9 = pub["sources"] + pub["shared_sources"]
    ours = {"params": 7_462_049, "per_source": {s: [0.9, 0.5] for s in src9}}
    computed = [{**c, "per_source": {s: [0.8, 0.5] for s in src9}} for c in pub["computed"]]
    runs = [{**u, "per_source": {s: [0.85, 0.55] for s in src9}} for u in pub["univariate"]]
    rows = render.indeval_figure_rows(render.indeval_rows(pub, ours, "Hirondel", computed),
                                      render.indeval_univariate_rows(pub, ours, "Hirondel", runs))
    names = [(r["model"], r["covariates"]) for r in rows]
    assert len(names) == len(set(names)) == 1 + 2 + 3 + 6       # ours, Timestral-5M x2, the paper's 3, 6 more runs
    assert sorted(render.indeval_links(rows)) == ["Chronos-2", "Timestral-5M"]      # same model, both settings
    render.indeval_figure(rows, str(tmp_path / "f.png"))
    w, h = Image.open(tmp_path / "f.png").size
    assert h > 0.75 * w                                       # two rows of panels: 5 sources, then 9 sources


def test_indeval_panels_do_not_shadow_the_pareto_panels():
    assert render.PANELS is build.PANELS and render.INDEVAL_PANELS != build.PANELS


def test_model_card_indeval_numbers_match_the_results():
    import json
    res = os.path.join(ROOT, "benchmarks", "results")
    with open(os.path.join(res, "indeval", "headline.json")) as f:
        ours = json.load(f)["nMASE_nCRPS_industrial"]["Hirondel"]
    with open(os.path.join(res, "indeval_competitors", "Chronos-2", "headline.json")) as f:
        alone = json.load(f)["nMASE_nCRPS_industrial"]["Chronos-2"][0]
    pub = render.load_indeval()
    joint = render._gmean([next(m for m in pub["models"] if m["name"] == "Chronos-2")["nmase"][s] for s in pub["sources"]])
    with open(os.path.join(render.HERE, "MODEL_CARD.template.md")) as f:
        card = f.read()
    assert f"{ours[0]:.3f} / {ours[1]:.3f}" in card and f"Chronos-2 from {alone:.3f} to {joint:.3f}" in card
