"""
tether_figure.py  —  art. §6.5  Figure 1: the diagonal tether
=============================================================

Composes Figure 1 from the artefacts of tether_run.py:

  top    triptych of the three MNIST embeddings (full-RV, hollow-RV, t-SNE),
         coloured by digit, each autoscaled (equal aspect) with its RMS spread
         in the title -- the tether tightens full-RV and loosens toward t-SNE.
  bottom energy split along the optimization: the structural energy ||K̊_Y||^2
         collapses (~7x) while the degree floor sum_i r_i^2 stays essentially
         pinned -- the mechanism of Prop. 5 / the justification Lemma.

Writes results/04_tether/tether_figure.{png,pdf}.
"""

# ruff: noqa: E402, I001  (imports follow the sys.path bootstrap)
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (for `src`)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.benchmark_common import exp_coords_dir, exp_indices_dir  # noqa: E402

EXP = "04_tether"
PANELS = [
    ("full_rv", "full-RV"),
    ("hollow_rv", "hollow-RV"),
    ("tsne", "t-SNE"),
]


def load_trajectory() -> dict[str, dict[str, np.ndarray]]:
    """config -> {iter, e_diag, e_hollow} as arrays."""
    rows = list(csv.DictReader((exp_indices_dir(EXP) / "tether_trajectory.csv").open()))
    out: dict[str, dict[str, np.ndarray]] = {}
    for cfg in ("full_rv", "hollow_rv"):
        r = [x for x in rows if x["config"] == cfg]
        out[cfg] = {
            "iter": np.array([int(x["iter"]) for x in r]),
            "e_diag": np.array([float(x["e_diag"]) for x in r]),
            "e_hollow": np.array([float(x["e_hollow"]) for x in r]),
        }
    return out


def load_finals() -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader((exp_indices_dir(EXP) / "tether_final.csv").open()))
    return {r["config"]: {"spread": float(r["spread"]),
                          "frac_diag": float(r["frac_diag"])} for r in rows}


def main() -> None:
    cdir = exp_coords_dir(EXP)
    labels = np.load(cdir / "labels.npy")
    finals = load_finals()
    traj = load_trajectory()

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.28, wspace=0.12)

    # ── top: triptych ────────────────────────────────────────────────────────
    for col, (key, name) in enumerate(PANELS):
        Y = np.load(cdir / f"{key}.npy")
        ax = fig.add_subplot(gs[0, col])
        ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10", s=7,
                   alpha=0.85, linewidths=0)
        f = finals[key]
        ax.set_title(
            f"{name}\nspread {f['spread']:.1f} · frac-diag {f['frac_diag']:.2f}",
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # ── bottom: energy split along the optimization ─────────────────────────
    ax = fig.add_subplot(gs[1, :])
    colors = {"full_rv": "#1f77b4", "hollow_rv": "#d62728"}
    names = {"full_rv": "full-RV", "hollow_rv": "hollow-RV"}
    for cfg in ("full_rv", "hollow_rv"):
        t = traj[cfg]
        ax.semilogy(t["iter"], t["e_hollow"], color=colors[cfg], lw=2,
                    label=rf"$\|\mathring{{K}}_Y\|^2$ ({names[cfg]})")
    # the degree floor (near-identical for both objectives): plot one, band the other
    t = traj["full_rv"]
    ax.semilogy(t["iter"], t["e_diag"], color="black", lw=1.6, ls="--",
                label=r"$\sum_i r_i^2$ (degree floor)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy (log scale)")
    ax.set_title(r"structural energy $\|\mathring{K}_Y\|^2$ collapses; "
                 r"degree floor $\sum_i r_i^2$ stays pinned", fontsize=11)
    ax.legend(loc="center right", fontsize=9, framealpha=0.9)
    ax.margins(x=0.01)

    out = exp_coords_dir(EXP).parent / "tether_figure"
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}.png and {out}.pdf")


if __name__ == "__main__":
    main()
