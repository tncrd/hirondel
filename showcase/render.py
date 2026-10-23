"""Figures, banner, README.md and MODEL_CARD.md from the collected rows (see `showcase.build`)."""
from __future__ import annotations

import collections
import os

from showcase.build import HERE, PANELS, ROOT, fev_error, fill, front_steps, pareto_front

ACCENT = "#0E7C7B"         # our model
ACCENT_FILL = "#CFE8E6"    # improvement area
INK = "#2B2B2B"
GREY = "#9A9A9A"
DARK = "#4A4A4A"
BG = "#FBFAF7"
GRID = "#E6E3DC"
FIG = os.path.join(HERE, "figures")


def _table(rows):
    t = collections.defaultdict(dict)
    meta = {}
    for r in rows:
        t[r["model"]][(r["benchmark"], r["metric"])] = float(r["value"])
        meta[r["model"]] = r
    return t, meta


def _err(bench, metric, v):
    return fev_error(v) if bench == "fev-bench" else v


def _style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.facecolor": BG, "figure.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": DARK, "ytick.color": DARK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.facecolor": BG,
    })


# (dx, dy, horizontal alignment) in points, to keep labels apart; default (5, 4, "left")
LABEL_OFFSETS = {"t0-gamma": (-6, -12, "right"), "TimesFM-3": (6, 4, "left"), "Toto-2.0-22m": (5, -12, "left"),
                 "TiRex": (6, 6, "left"), "Timestral-5M": (-6, 6, "right"), "Timestral-209M": (6, -12, "left"),
                 "Moirai-2.0-small": (6, 4, "left")}
LABEL_OFFSETS_BY_BENCH = {"GIFT-Eval": {"Toto-2.0-22m": (5, 4, "left"), "TiRex": (6, -12, "left"),
                                        "FlowState-r1.1": (-6, 5, "right")},
                          "TIME": {"t0-gamma": (6, -12, "left"), "Timestral-209M": (-6, 5, "right"),
                                   "Timestral-5M": (-6, 6, "right")},
                          "fev-bench": {"t0-gamma": (6, -12, "left"), "Timestral-209M": (-6, 5, "right"),
                                        "TiRex": (8, 2, "left"), "Toto-2.0-22m": (-6, -12, "right")}}

LABEL_OFFSETS_BY_PANEL = {("GIFT-Eval", "MASE"): {"FlowState-r1.1": (5, 4, "left")},
                          ("fev-bench", "MASE"): {"Moirai-2.0-small": (6, -12, "left")}}


OFFICIAL_TITLE = "Official leaderboards (some models use covariates or forecast the variables of a series jointly)"
OFFICIAL_NOTE = ("Official leaderboard results, read on 2026-10-23 (fev-bench: 1 - skill score). Pareto fronts over models "
                 "with a permissive license; the others are shown hollow. Sizes from the official model cards. "
                 "Sources: showcase/data/official.csv.")
UNIVARIATE_TITLE = "Univariate setting: every model forecasts each series alone, without covariates"
UNIVARIATE_NOTE = ("Our runs of each model in forced univariate mode with its public package (benchmarks/competitors.yaml); "
                   "TiRex, TimesFM-2.5, FlowState-r1.1 and Moirai-2.0, univariate by design, from their official results. "
                   "t0-gamma and TimesFM-3 use covariates and were not run in this setting: see the official leaderboards "
                   "figure. Pareto fronts over models with a permissive license; the others hollow. "
                   "Sources: showcase/data/univariate.csv.")


def pareto_figure(rows, models, ours_name, path, title=OFFICIAL_TITLE, note=OFFICIAL_NOTE):
    """Accuracy against size, one panel per benchmark and metric. The Pareto fronts are drawn over the models with a
    permissive license (the claim of the head sentence); models under a non-commercial or restricted license stay on
    the figure as hollow markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    _style(plt)
    t, meta = _table(rows)
    family = {m["name"]: m["family"] for m in models["models"]}
    permissive = {m["name"]: m["permissive"] for m in models["models"]}
    permissive[ours_name] = True
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.4))
    x_max = 1.5e9
    for ax, (bench, metric, label) in zip(axes.flat, PANELS):
        pts = [(m, float(meta[m]["params"]), _err(bench, metric, t[m][(bench, metric)]))
               for m in t if (bench, metric) in t[m]]
        others = [p for p in pts if p[0] != ours_name]
        ours = [p for p in pts if p[0] == ours_name]
        open_pts = [p for p in others if permissive[p[0]]]
        public_front = pareto_front(open_pts)
        full_front = pareto_front(open_pts + ours)
        on_front = {p[0] for p in public_front}
        fam = collections.defaultdict(list)
        for p in others:
            fam[family.get(p[0], p[0])].append(p)
        for ps in fam.values():
            if len(ps) > 1:
                ps = sorted(ps, key=lambda p: p[1])
                ax.plot([p[1] for p in ps], [p[2] for p in ps], color=GREY, lw=0.8, zorder=1, alpha=0.6)
        xs0, ys0 = front_steps(public_front, x_max)
        xs1, ys1 = front_steps(full_front, x_max)
        ax.plot(xs0, ys0, color=DARK, lw=1.0, ls=(0, (4, 3)), zorder=2)
        if ours and ours[0][0] in {p[0] for p in full_front}:
            grid = np.geomspace(xs1[0], x_max, 3000)
            f1 = np.array([min(p[2] for p in full_front if p[1] <= g) for g in grid])
            f0 = np.array([min([p[2] for p in public_front if p[1] <= g] or [np.inf]) for g in grid])
            # shaded only where a public model already exists at that size: an improvement over something
            ax.fill_between(grid, f1, np.where(np.isfinite(f0), f0, f1), where=np.isfinite(f0) & (f1 < f0 - 1e-12),
                            color=ACCENT_FILL, step="post", zorder=0, lw=0)
            ax.plot(xs1, ys1, color=ACCENT, lw=1.7, zorder=3)
        span = max(p[2] for p in pts) - min(p[2] for p in pts) or 1.0

        def near_ours(x, y):
            return bool(ours) and abs(np.log10(x / ours[0][1])) < 0.35 and abs(y - ours[0][2]) < 0.12 * span

        below = any(near_ours(x, y) and y < ours[0][2] for _, x, y in others)
        for name, x, y in others:
            front = name in on_front
            if permissive[name]:
                ax.scatter([x], [y], s=28 if front else 16, color=DARK if front else GREY, zorder=4, edgecolors="none")
            else:
                ax.scatter([x], [y], s=22, facecolors=BG, edgecolors=GREY, linewidths=1.1, zorder=4)
            dx, dy, ha = (LABEL_OFFSETS_BY_PANEL.get((bench, metric), {}).get(name)
                          or LABEL_OFFSETS_BY_BENCH.get(bench, {}).get(name) or LABEL_OFFSETS.get(name, (5, 4, "left")))
            if near_ours(x, y):                   # a neighbour of ours: its label on the side away from ours
                dy = -12 if y < ours[0][2] else 6
            ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, ha=ha,
                        color=INK if front else "#7A7A7A", style="normal" if permissive[name] else "italic", zorder=5)
        for name, x, y in ours:
            ax.scatter([x], [y], s=120, color=ACCENT, zorder=6, edgecolors="white", linewidths=1.4)
            # below the point, unless a neighbour sits just below it: then above
            ax.annotate(f"{name} (ours)", (x, y), xytext=(-9, 5 if below else -4), textcoords="offset points",
                        fontsize=10.5, fontweight="bold", color=ACCENT, ha="right", va="bottom" if below else "top",
                        zorder=7)
        ax.margins(y=0.09)
        ax.set_xscale("log")
        ax.set_xlim(1.2e6, x_max)
        ax.set_xticks([1e7, 3e7, 1e8, 3e8, 1e9])
        ax.set_xticklabels(["10M", "30M", "100M", "300M", "1B"])
        ax.set_title(f"{bench} · {label}", loc="left", fontsize=12, fontweight="bold", color=INK)
        ax.text(1.0, 1.02, "lower is better", transform=ax.transAxes, ha="right", fontsize=8.5, color=GREY)
    for ax in axes[1]:
        ax.set_xlabel("Parameters (log scale)")
    handles = [
        Line2D([], [], marker="o", ls="", color=ACCENT, markersize=9, label="Ours (Apache-2.0)"),
        Line2D([], [], marker="o", ls="", color=DARK, markersize=6, label="Pareto-optimal, permissive license"),
        Line2D([], [], marker="o", ls="", color=GREY, markersize=5, label="Other, permissive license"),
        Line2D([], [], marker="o", ls="", markerfacecolor=BG, markeredgecolor=GREY, markersize=6,
               label="Non-commercial or restricted license"),
        Line2D([], [], color=GREY, lw=0.8, label="Same family"),
        Line2D([], [], color=DARK, lw=1.0, ls=(0, (4, 3)), label="Front without ours"),
        Line2D([], [], color=ACCENT, lw=1.7, label="Front with ours"),
        Patch(color=ACCENT_FILL, label="Improvement"),
    ]
    fig.suptitle(title, x=0.012, y=0.997, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.legend(handles=handles, loc="upper center", ncol=8, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    import textwrap
    lines = textwrap.wrap(note, 190)
    fig.text(0.5, 0.004, "\n".join(lines), ha="center", fontsize=8.5, color=GREY, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.012 + 0.017 * len(lines), 1, 0.93))     # room for every line of the note
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160, metadata={"Software": None})
    plt.close(fig)


def same_protocol_figure(sp, sizes, ours_name, path, permissive=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric in zip(axes, ("MASE", "CRPS")):
        for r in sp:
            x = sizes[r["model"]]
            mine = r["model"] == ours_name
            if mine:
                ax.scatter([x], [r[metric]], s=120, color=ACCENT, zorder=4, edgecolors="white", linewidths=1.4)
            elif (permissive or {}).get(r["model"].split(" (")[0], True):
                ax.scatter([x], [r[metric]], s=28, color=DARK, zorder=4, edgecolors="none")
            else:
                ax.scatter([x], [r[metric]], s=26, facecolors=BG, edgecolors=DARK, linewidths=1.1, zorder=4)
            ax.annotate(f"{r['model']} (ours)" if mine else r["model"], (x, r[metric]),
                        xytext=(7, -13) if mine else (5, -11) if r["model"] == "TiRex" else (5, 4),
                        textcoords="offset points",                      # TiRex below: Toto-2.0-22m sits beside it
                        fontsize=10 if mine else 8, fontweight="bold" if mine else "normal",
                        color=ACCENT if mine else INK)
        ax.set_xscale("log")
        ax.set_xlim(3e6, 6e8)
        ax.set_xticks([1e7, 3e7, 1e8, 3e8])
        ax.set_xticklabels(["10M", "30M", "100M", "300M"])
        ax.set_title(f"GIFT-Eval · {metric}" + (" (WQL)" if metric == "CRPS" else ""), loc="left", fontsize=12,
                     fontweight="bold", color=INK)
        ax.text(1.0, 1.02, "lower is better", transform=ax.transAxes, ha="right", fontsize=8.5, color=GREY)
        ax.set_xlabel("Parameters (log scale)")
    fig.text(0.5, 0.01, "Same protocol, computed by us: forced univariate (each series alone, no covariates), public "
             "packages, settings of each model's GIFT-Eval notebook.\nNot the leaderboards' official numbers. "
             "Hollow: non-commercial or restricted license.", ha="center", fontsize=8.5, color=GREY, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=160, metadata={"Software": None})
    plt.close(fig)


def banner(model_path, title, tagline, path):
    """A real forecast of the released model (CPU, one thread) on a seasonal series, with its quantile fan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from hirondel import HirondelForecaster
    _style(plt)
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    t = np.arange(600)
    y = (10 + 3 * np.sin(2 * np.pi * t / 48) + 1.2 * np.sin(2 * np.pi * t / 168) + 0.004 * t
         + 0.5 * rng.normal(size=t.size)).astype(np.float32)
    ctx, H = y[:504], 96
    f = HirondelForecaster.from_pretrained(model_path).forecast(ctx, H)
    q = f.quantiles[0]
    fig = plt.figure(figsize=(12, 3.2))
    ax = fig.add_axes([0.50, 0.08, 0.48, 0.84])
    xs, xf = np.arange(504 - 216, 504), np.arange(504, 504 + H)
    ax.plot(xs, ctx[-216:], color=DARK, lw=1.2)
    ax.plot(xf, y[504:504 + H], color=GREY, lw=1.0, ls=(0, (2, 2)))
    for lo, hi, a in ((0, 8, 0.18), (1, 7, 0.28), (2, 6, 0.40)):
        ax.fill_between(xf, q[:, lo], q[:, hi], color=ACCENT, alpha=a, lw=0)
    ax.plot(xf, q[:, 4], color=ACCENT, lw=1.8)
    ax.axis("off")
    fig.text(0.04, 0.56, title, fontsize=44, fontweight="bold", color=INK)
    fig.text(0.04, 0.30, tagline, fontsize=12.5, color=DARK, linespacing=1.5)
    fig.text(0.04, 0.14, "Apache-2.0 · zero-shot · CPU-friendly", fontsize=11, color=ACCENT)
    fig.savefig(path, dpi=160, metadata={"Software": None})
    plt.close(fig)


def _f(v, n=4):
    return f"{v:.{n}f}"


def univariate_table(uni, models, name):
    """Univariate setting, all three benchmarks: one row per model, sorted by size."""
    t, meta = _table(uni)
    lic = {m["name"]: m["license"] for m in models["models"]}
    lic[name] = models["ours"]["license"]
    cols = [("GIFT-Eval", "MASE"), ("GIFT-Eval", "CRPS"), ("TIME", "MASE"), ("TIME", "CRPS"), ("fev-bench", "SQL")]
    lines = ["| Model | Parameters | License | GIFT-Eval MASE | GIFT-Eval CRPS | TIME MASE | TIME CRPS | fev-bench SQL skill |",
             "|---|---|---|---|---|---|---|---|"]
    for m in sorted(t, key=lambda m: float(meta[m]["params"])):
        cells = [(_f(t[m][c], 2) if c[0] == "fev-bench" else _f(t[m][c])) if c in t[m] else "" for c in cols]
        label = f"**{m} (ours)**" if m == name else m
        lines.append(f"| {label} | {float(meta[m]['params']) / 1e6:.1f}M | {lic.get(m, '')} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def readme_values(rows, sp, ours, models, sentence):
    t, meta = _table(rows)
    name = models["ours"]["name"]
    g, tm, fv = ours["GIFT-Eval"], ours["TIME"], ours["fev-bench"]
    results = "\n".join([
        "<!-- results:start -->",
        "| Benchmark | MASE | CRPS (fev-bench: SQL) |",
        "|---|---|---|",
        f"| GIFT-Eval (97 tasks, normalised by Seasonal Naive; lower is better) | {_f(g['MASE'])} | {_f(g['CRPS'])} |",
        f"| fev-bench (100 tasks, skill score in %; higher is better) | {_f(fv['MASE'], 2)} | {_f(fv['SQL'], 2)} |",
        f"| TIME (98 tasks, normalised by Seasonal Naive; lower is better) | {_f(tm['MASE'])} | {_f(tm['CRPS'])} |",
        "<!-- results:end -->",
        "",
        f"These numbers are computed by us and not yet listed on the leaderboards. Added to the fev-bench leaderboard, "
        f"they would rank {fv['rank_SQL']} of {fv['n_models']} (SQL) and {fv['rank_MASE']} of {fv['n_models']} (MASE) "
        f"among the models the site shows by default (2026-10-23).",
    ])
    sizes = {m["name"]: m["params"] for m in models["models"]}
    sizes[name] = models["ours"]["params"]
    lines = ["| Model | Parameters | License | GIFT-Eval MASE | GIFT-Eval CRPS |", "|---|---|---|---|---|"]
    lic = {m["name"]: m["license"] for m in models["models"]}
    lic[name] = models["ours"]["license"]
    for r in sorted(sp, key=lambda r: r["MASE"]):
        n = r["model"].split(" (")[0]
        p = sizes.get(n)
        lines.append(f"| {'**' + r['model'] + ' (ours)**' if r['model'] == name else r['model']} | "
                     f"{p / 1e6:.1f}M | {lic.get(n, '')} | {_f(r['MASE'])} | {_f(r['CRPS'])} |" if p else
                     f"| {r['model']} | | | {_f(r['MASE'])} | {_f(r['CRPS'])} |")
    return {"sentence": sentence, "results": results, "same_protocol": "\n".join(lines)}


def all(rows, sp, ours, models, model_path, uni=None):
    name = models["ours"]["name"]
    os.makedirs(FIG, exist_ok=True)
    pareto_figure(rows, models, name, os.path.join(FIG, "pareto.png"))
    if uni:
        pareto_figure(uni, models, name, os.path.join(FIG, "univariate.png"), UNIVARIATE_TITLE, UNIVARIATE_NOTE)
    sizes = {m["name"]: m["params"] for m in models["models"]}
    sizes[name] = models["ours"]["params"]
    sp_all = sp + [{"model": name, "MASE": ours["GIFT-Eval"]["MASE"], "CRPS": ours["GIFT-Eval"]["CRPS"]}]
    for r in sp_all:
        sizes.setdefault(r["model"], sizes.get(r["model"].split(" (")[0]))
    permissive = {m["name"]: m["permissive"] for m in models["models"]}
    same_protocol_figure(sp_all, sizes, name, os.path.join(FIG, "same_protocol.png"), permissive)
    banner(model_path, name, "A 7M-parameter model for zero-shot\nprobabilistic time-series forecasting",
           os.path.join(FIG, "banner.png"))
    with open(os.path.join(HERE, "sentence.txt")) as f:
        sentence = f.read().strip()
    values = readme_values(rows, sp_all, ours, models, sentence)
    values["univariate"] = univariate_table(uni or [], models, name)
    values["indeval"] = indeval_section(name, models["ours"]["params"])
    for template, out in (("README.template.md", "README.md"), ("MODEL_CARD.template.md", "MODEL_CARD.md")):
        with open(os.path.join(HERE, template)) as f:
            text = fill(f.read(), {**values, **card_values(ours)})
        with open(os.path.join(ROOT, out), "w") as f:
            f.write(text)


def card_values(ours):
    g, tm, fv = ours["GIFT-Eval"], ours["TIME"], ours["fev-bench"]
    return {"gift_mase": _f(g["MASE"]), "gift_crps": _f(g["CRPS"]), "time_mase": _f(tm["MASE"]),
            "time_crps": _f(tm["CRPS"]), "fev_mase": _f(fv["MASE"], 2), "fev_sql": _f(fv["SQL"], 2)}


# ---- IND-Eval -------------------------------------------------------------------------------------------------------
def load_indeval(path: str = os.path.join(HERE, "indeval.yaml")) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _gmean(x):
    import math
    return math.exp(sum(math.log(v) for v in x) / len(x))


def indeval_rows(pub, ours, name):
    """One row per model: per-source (nMASE, nCRPS) and their geometric mean over the five sources. Ours first."""
    src = pub["sources"]
    rows = [{"model": name, "params": ours.get("params"), "permissive": True, "ours": True, "covariates": False,
             "per_source": {s: tuple(ours["per_source"][s]) for s in src}}]
    for m in pub["models"]:
        rows.append({"model": m["name"], "params": m["params"], "permissive": m["permissive"], "ours": False,
                     "covariates": m.get("covariates", False),
                     "per_source": {s: (m["nmase"][s], m["ncrps"][s]) for s in src}})
    for r in rows:
        r["mean"] = (_gmean([r["per_source"][s][0] for s in src]), _gmean([r["per_source"][s][1] for s in src]))
    return rows


def _size(p):
    return f"{p / 1e6:.0f}M" if p >= 1e7 else f"{p / 1e6:.1f}M"


def _label(r):
    size = _size(r["params"]) if r["params"] else ""
    return f"{r['model']} ({size}{', ' if size and r['ours'] else ''}{'ours' if r['ours'] else ''})"


def indeval_table(rows, pub):
    """Markdown table, nMASE / nCRPS per source and their mean; the best value of each metric in bold."""
    src = pub["sources"]
    head = ["Source"] + [_label(r) for r in rows]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def cells(get):
        best = [min(get(r)[k] for r in rows) for k in (0, 1)]
        return [" / ".join((f"**{get(r)[k]:.3f}**" if abs(get(r)[k] - best[k]) < 1e-12 else f"{get(r)[k]:.3f}")
                           for k in (0, 1)) for r in rows]
    for s in src:
        lines.append(f"| {pub['source_names'][s]} | " + " | ".join(cells(lambda r: r["per_source"][s])) + " |")
    lines.append("| **Geometric mean, 5 sources** | " + " | ".join(cells(lambda r: r["mean"])) + " |")
    return "\n".join(lines)


INDEVAL_NOTE = ("IND-Eval, five industrial sources (NEATS, 3W, PRONTO, Mesa del Sol, tablet compression); geometric mean of "
                "the sources. {name}: our run, each channel forecast alone, no covariates. Others: numbers published in "
                "the IND-Eval paper (arXiv 2610.16652), joint forecasting with PRONTO's covariates.")


def indeval_figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    _style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, k, metric in zip(axes, (0, 1), ("nMASE", "nCRPS")):
        for r in rows:
            x, y = r["params"], r["mean"][k]
            marker = "X" if r["covariates"] else "o"             # cross: reads covariates; circle: univariate
            if r["ours"]:
                ax.scatter([x], [y], s=120, marker=marker, color=ACCENT, zorder=4, edgecolors="white", linewidths=1.4)
            elif r["permissive"]:
                ax.scatter([x], [y], s=60, marker=marker, color=DARK, zorder=4, edgecolors="none")
            else:
                ax.scatter([x], [y], s=60, marker=marker, facecolors=BG, edgecolors=DARK, linewidths=1.1, zorder=4)
            ax.annotate(f"{r['model']} (ours)" if r["ours"] else r["model"], (x, y),
                        xytext=(9, -4) if r["ours"] else (6, 4), textcoords="offset points", va="top" if r["ours"] else "baseline",
                        fontsize=10 if r["ours"] else 8, fontweight="bold" if r["ours"] else "normal",
                        color=ACCENT if r["ours"] else INK)
        ax.margins(y=0.25)
        ax.set_xscale("log")
        ax.set_xlim(3e6, 6e8)
        ax.set_xticks([1e7, 3e7, 1e8, 3e8])
        ax.set_xticklabels(["10M", "30M", "100M", "300M"])
        ax.set_title(f"IND-Eval · {metric}", loc="left", fontsize=12, fontweight="bold", color=INK)
        if k == 1:                                        # nCRPS panel: its upper left corner is empty
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([], [], ls="", marker="o", color=DARK, label="univariate, no covariates"),
                               Line2D([], [], ls="", marker="X", color=DARK, markersize=8,
                                      label="joint, with covariates (PRONTO)"),
                               Line2D([], [], ls="", marker="X", markerfacecolor=BG, markeredgecolor=DARK, markersize=8,
                                      label="hollow: non-commercial license")],
                      loc="upper left", fontsize=8, frameon=False)
        ax.text(1.0, 1.02, "lower is better", transform=ax.transAxes, ha="right", fontsize=8.5, color=GREY)
        ax.set_xlabel("Parameters (log scale)")
    note = INDEVAL_NOTE.format(name=next(r["model"] for r in rows if r["ours"]))
    lines = textwrap.wrap(note, 150)
    fig.text(0.5, 0.01, "\n".join(lines), ha="center", fontsize=8.5, color=GREY, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.03 + 0.035 * len(lines), 1, 1))
    fig.savefig(path, dpi=160, metadata={"Software": None})
    plt.close(fig)


def indeval_section(name, params, results=os.path.join(ROOT, "benchmarks", "results", "indeval", "headline.json")):
    """Figure + table from our results and the published numbers; empty when IND-Eval was not run."""
    if not os.path.exists(results):
        return ""
    import json
    with open(results) as f:
        hd = json.load(f)
    pub = load_indeval()
    rows = indeval_rows(pub, {"params": params, "per_source": {s: hd["per_source"][s][name] for s in pub["sources"]}},
                        name)
    indeval_figure(rows, os.path.join(FIG, "indeval.png"))
    return indeval_table(rows, pub)
