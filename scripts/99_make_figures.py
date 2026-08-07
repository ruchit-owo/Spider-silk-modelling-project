"""Build every figure from the saved result files.

Reads only from results/. Runs on CPU and needs no model, so figures can be
regenerated without repeating a multi-hour run.

Figures produced:

    fig1_table1_reproduction.png  paper's reported R2 against our pool
                                  maximum, mean and median, per property set
    fig2_bestofn.png              expected best-of-N curves
    fig3_r2_distribution.png      full R2 distribution per set, with the
                                  paper's value marked
    fig4_ablation.png             prediction shift per ablation, against the
                                  measured noise floor
    fig5_baseline.png             baseline R2, in-sample vs cross-validated
    fig6_nonmasp.png              mean prediction by spidroin family

Any figure whose input is missing is skipped with a message rather than
failing, so a partial run still produces what it can.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

PAPER_C = "#2f6fa8"
OURS_C = "#c4552b"
GREY = "#8a8a8a"


def _load(name: str):
    p = config.RESULTS / f"{name}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)["result"]


def fig_table1() -> None:
    res = _load("02_table1_reproduction")
    if res is None:
        print("skip fig1: results/02_table1_reproduction.json missing")
        return
    s = [x for x in res["summaries"] if "r2_max" in x]
    if not s:
        print("skip fig1: no scored sets")
        return
    names = [x["set"] for x in s]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(x - 0.26, [v["paper_r2"] for v in s], width=0.24,
           color=PAPER_C, label="paper (max over unstated N)")
    ax.bar(x, [v["r2_max"] for v in s], width=0.24,
           color=OURS_C, label=f"ours, max over N={s[0]['n_scored']}")
    ax.bar(x + 0.26, [v["r2_median"] for v in s], width=0.24,
           color=GREY, label="ours, median single sample")

    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, names)
    ax.set_xlabel("property set")
    ax.set_ylabel(r"$R^2$ within the 8-vector")
    ax.set_title("Table 1 reproduction: the maximum is a statistic about the search")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(config.FIGURES / "fig1_table1_reproduction.png")
    plt.close(fig)
    print("wrote fig1_table1_reproduction.png")


def fig_bestofn() -> None:
    p = config.RESULTS / "bestofn_curves.csv"
    if not p.exists():
        print("skip fig2: results/bestofn_curves.csv missing")
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    cmap = plt.get_cmap("tab10")
    for i, (name, grp) in enumerate(df.groupby("set")):
        grp = grp.sort_values("n")
        ax.plot(grp["n"], grp["mean"], marker="o", ms=3, color=cmap(i % 10), label=name)
        ax.fill_between(grp["n"], grp["p05"], grp["p95"], color=cmap(i % 10), alpha=0.12)
        paper = config.ALL_R2_PAPER.get(name)
        if paper is not None:
            ax.axhline(paper, color=cmap(i % 10), ls=":", lw=0.8)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("N candidates sampled, then best kept")
    ax.set_ylabel(r"expected best $R^2$")
    ax.set_title("Reported performance grows with the sampling budget alone\n"
                 "(dotted lines: the paper's reported value for each set)",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(config.FIGURES / "fig2_bestofn.png")
    plt.close(fig)
    print("wrote fig2_bestofn.png")


def fig_r2_distribution() -> None:
    pools = sorted((config.RESULTS / "pools").glob("pool_*.csv"))
    if not pools:
        print("skip fig3: no pools")
        return
    pools = [p for p in pools if p.stem.replace("pool_", "") in config.ALL_SETS]
    n = len(pools)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow),
                             squeeze=False)
    for ax, p in zip(axes.ravel(), pools):
        name = p.stem.replace("pool_", "")
        df = pd.read_csv(p)
        r2 = df[df["forward_parsed"] & df["novel_exact"]]["r2"].to_numpy(dtype=float)
        r2 = r2[np.isfinite(r2)]
        if r2.size == 0:
            ax.set_visible(False)
            continue
        # Clip the display range: the left tail runs to large negative values
        # and would compress everything of interest into one bin.
        lo = float(np.percentile(r2, 2))
        ax.hist(np.clip(r2, lo, 1.0), bins=40, color=GREY, edgecolor="none")
        ax.axvline(config.ALL_R2_PAPER[name], color=PAPER_C, lw=1.4,
                   label="paper")
        ax.axvline(r2.max(), color=OURS_C, lw=1.4, ls="--", label="our max")
        ax.set_title(f"{name}  (n={r2.size})", fontsize=8)
        ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle(r"Distribution of within-vector $R^2$ over the candidate pool",
                 fontsize=10)
    fig.savefig(config.FIGURES / "fig3_r2_distribution.png")
    plt.close(fig)
    print("wrote fig3_r2_distribution.png")


def fig_ablation() -> None:
    res = _load("11_ablation_sequence_content")
    if res is None:
        print("skip fig4: results/11_ablation_sequence_content.json missing")
        return
    s = pd.DataFrame(res["summary"]).sort_values("mean_abs_delta")
    floor = res["noise_floor"]["pooled_mean_abs_dev"]

    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(s) + 1.6))
    colours = [OURS_C if "length_control" not in n else GREY for n in s["ablation"]]
    ax.barh(s["ablation"], s["mean_abs_delta"], color=colours)
    ax.axvline(floor, color="k", ls="--", lw=1.0,
               label=f"repeat noise floor ({floor:.4f})")
    ax.set_xlabel("mean |change| in the 8-vector prediction")
    ax.set_title("What the forward task reads from the sequence\n"
                 "(grey: length-matched controls)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(config.FIGURES / "fig4_ablation.png")
    plt.close(fig)
    print("wrote fig4_ablation.png")


def fig_baseline() -> None:
    p = config.RESULTS / "baseline_summary.csv"
    if not p.exists():
        print("skip fig5: results/baseline_summary.csv missing")
        return
    df = pd.read_csv(p)
    labels = df["feature_set"] + "\n" + df["model"]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(x - 0.19, df["mean_r2_in_sample_mechanical"], width=0.36,
           color=OURS_C, label="in-sample (fit and scored on all 1,033)")
    ax.bar(x + 0.19, df["mean_r2_cv_mechanical"], width=0.36,
           color=GREY, label="5-fold cross-validated")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel(r"mean $R^2$ over the 4 mechanical properties")
    ax.set_title("Composition baselines memorise but do not generalise", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(config.FIGURES / "fig5_baseline.png")
    plt.close(fig)
    print("wrote fig5_baseline.png")


def fig_nonmasp() -> None:
    res = _load("13_extension_nonmasp")
    if res is None:
        print("skip fig6: results/13_extension_nonmasp.json missing")
        return
    df = pd.DataFrame(res["mean_prediction_by_family"])
    mech = [config.PROPERTY_NAMES[i] for i in config.MECHANICAL_IDX]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    w = 0.8 / len(mech)
    for i, m in enumerate(mech):
        ax.bar(x + (i - (len(mech) - 1) / 2) * w, df[m], width=w, label=m)
    ax.set_xticks(x, df["family"])
    ax.set_ylabel("mean predicted value (normalised)")
    ax.set_title("Forward-task output by spidroin family\n"
                 "(the model was fine-tuned on MaSp only)", fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=4)
    fig.savefig(config.FIGURES / "fig6_nonmasp.png")
    plt.close(fig)
    print("wrote fig6_nonmasp.png")


def main() -> int:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    for fn in (fig_table1, fig_bestofn, fig_r2_distribution,
               fig_ablation, fig_baseline, fig_nonmasp):
        try:
            fn()
        except Exception as exc:  # a broken figure must not block the others
            print(f"{fn.__name__} failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
