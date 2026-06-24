"""
hybrids_fulldata_figures.py  —  §5.3.3  full-data illustration figures
======================================================================

Builds, from results/coordinates/hybrids/{dataset}__supervised_b{β}__full.npy
(+ run_meta_full.csv for the RV annotations), a 1×|β| panel row per labelled
dataset showing the WHOLE point cloud (coloured by label) evolving along the
supervised dial β = 0 (t-SNE) → 1 (class-centroid). Illustration only.
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import indices as ix
from src.benchmark_common import (
    SEED,
    SUPERVISED_BETAS,
    TRUST_K,
    coord_path,
    coords_dir,
    fig_path,
)
from src.datasets import load_all

SECTION = "5.3.3"
FAMILY = "hybrids"
META_NAME = "run_meta_full.csv"


def load_rv() -> dict[tuple[str, str], float]:
    rv: dict[tuple[str, str], float] = {}
    with (coords_dir(FAMILY) / META_NAME).open() as f:
        for r in csv.DictReader(f):
            rv[(r["dataset"], r["key"])] = float(r["rv_final"])
    return rv


def main() -> None:
    datasets = load_all(random_state=SEED)
    rv = load_rv()
    for ds in datasets.values():
        if not ds.has_labels:
            continue
        n = len(SUPERVISED_BETAS)
        fig, axes = plt.subplots(1, n, figsize=(n * 3.4, 3.6))
        for ax, beta in zip(axes, SUPERVISED_BETAS):
            key = f"supervised_b{beta}"
            Y = np.load(coord_path(FAMILY, ds.name, key, "full"))
            ax.scatter(
                Y[:, 0], Y[:, 1], c=ds.color, cmap=ds.cmap, s=6, alpha=0.8, linewidths=0
            )
            t = ix.trustworthiness(ds.X, Y, k=TRUST_K)
            a = ix.ari(Y, ds.labels)
            end = " (t-SNE)" if beta == 0.0 else (" (class)" if beta == 1.0 else "")
            ax.set_title(
                f"β={beta}{end}\nRV={rv[(ds.name, key)]:.3f} "
                f"Trust={t:.3f} ARI={a:.3f}",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"{ds.name} — supervised interpolation on all data (β = t-SNE → class)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(
            fig_path(SECTION, ds.name, "supervised", "fulldata"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        print(f"  {ds.name}: full-data sweep written")


if __name__ == "__main__":
    main()
