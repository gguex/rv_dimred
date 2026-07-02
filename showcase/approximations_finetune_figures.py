"""
approximations_finetune_figures.py  —  heatmaps of the hollow-RV sweep
======================================================================

Reads results/indices/approximations_finetune/approximations_finetune_tidy.csv and
builds, per (dataset × method), a panel of heatmaps over the joint (hp × lambda)
grid:
  * Procrustes (framework vs reference; lower = better recovery),
  * kNN overlap (framework vs reference; higher = better),
  * trustworthiness — framework and reference (shared colour scale),
  * ARI — framework and reference (labelled datasets; shared colour scale).

Reference quality depends only on the neighbour hyperparameter (not lambda), so its
heatmaps show constant rows — directly comparable to the framework panel beside it.
Figures go to results/figures/approximations_finetune/.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.benchmark_common import RESULTS_DIR, indices_dir

FAMILY = "approximations_finetune"


def load_rows() -> list[dict[str, str]]:
    path = indices_dir(FAMILY) / "approximations_finetune_tidy.csv"
    with path.open() as f:
        return list(csv.DictReader(f))


def fig_dir():
    d = RESULTS_DIR / "figures" / FAMILY
    d.mkdir(parents=True, exist_ok=True)
    return d


def heatmap(ax, M, hp_vals, lam_vals, hp_name, title, lower_better, vlim):
    cmap = "viridis_r" if lower_better else "viridis"
    vmin, vmax = vlim
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(lam_vals)))
    ax.set_xticklabels([f"{x:g}" for x in lam_vals])
    ax.set_yticks(range(len(hp_vals)))
    ax.set_yticklabels([f"{x:g}" for x in hp_vals])
    ax.set_xlabel("lambda")
    ax.set_ylabel(hp_name)
    for i in range(len(hp_vals)):
        for j in range(len(lam_vals)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        color="w", fontsize=7)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def build(dataset: str, mkey: str, method: str, hp_name: str, rows: list[dict]) -> None:
    hp_vals = sorted({int(r["hp_value"]) for r in rows})
    lam_vals = sorted({float(r["lambda"]) for r in rows if r["variant"] == "framework"})
    hi, li = {h: i for i, h in enumerate(hp_vals)}, {l: j for j, l in enumerate(lam_vals)}

    # fw[index][hp,lam]  and  ref[index][hp]  (ref is lambda-independent)
    fw = defaultdict(lambda: np.full((len(hp_vals), len(lam_vals)), np.nan))
    ref = defaultdict(lambda: np.full((len(hp_vals), len(lam_vals)), np.nan))
    for r in rows:
        hp, idx, val = int(r["hp_value"]), r["index_name"], float(r["value"])
        if r["variant"] == "framework":
            fw[idx][hi[hp], li[float(r["lambda"])]] = val
        else:  # reference: broadcast across all lambda columns
            ref[idx][hi[hp], :] = val

    has_ari = "ari" in fw and not np.isnan(fw["ari"]).all()

    def lim(*mats):
        vals = np.concatenate([m[~np.isnan(m)] for m in mats]) if mats else np.array([0.0])
        return (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)

    # panel list: (matrix, title, lower_better, vlim)
    t_lim = lim(fw["trustworthiness"], ref["trustworthiness"])
    panels = [
        (fw["procrustes"], "Procrustes (fw vs ref) — lower better", True,
         lim(fw["procrustes"])),
        (fw["knn_overlap"], "kNN overlap (fw vs ref) — higher better", False,
         lim(fw["knn_overlap"])),
        (fw["trustworthiness"], "trustworthiness — framework", False, t_lim),
        (ref["trustworthiness"], "trustworthiness — reference", False, t_lim),
    ]
    if has_ari:
        a_lim = lim(fw["ari"], ref["ari"])
        panels += [
            (fw["ari"], "ARI — framework", False, a_lim),
            (ref["ari"], "ARI — reference", False, a_lim),
        ]

    ncol = 2
    nrow = (len(panels) + 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (M, title, lb, vl) in zip(axes, panels):
        heatmap(ax, M, hp_vals, lam_vals, hp_name, title, lb, vl)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{dataset} — {method}: hollow-RV sweep ({hp_name} × lambda)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = fig_dir() / f"heatmap_{dataset}_{mkey}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> None:
    rows = load_rows()
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    meta: dict[tuple[str, str], tuple[str, str]] = {}
    for r in rows:
        key = (r["dataset"], r["method_key"])
        groups[key].append(r)
        meta[key] = (r["method"], r["hp_name"])
    for (dataset, mkey), grp in groups.items():
        method, hp_name = meta[(dataset, mkey)]
        build(dataset, mkey, method, hp_name, grp)


if __name__ == "__main__":
    main()
