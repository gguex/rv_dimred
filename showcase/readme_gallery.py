"""
readme_gallery.py  —  the spectral-zoo gallery for the repository README
========================================================================

A single grid (3 datasets x 6 spectral methods) of the closed-form RV embeddings,
each Procrustes-aligned to its reference-library orientation and colored by the
dataset's ground truth. The point of the picture: the whole spectral zoo is one
operation --- the projection of a different input kernel onto the same cone.

Writes results/figures/spectral_gallery.png.
"""

# ruff: noqa: E402, I001  (imports follow the sys.path bootstrap)
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (for `src`)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import indices as ix  # noqa: E402
from src.benchmark_common import (  # noqa: E402
    SEED,
    SPECTRAL_METHODS,
    exp_coord_path,
)
from src.datasets import load_all  # noqa: E402

EXP = "01_spectral"
DATASETS = ("mnist", "singlecell", "swissroll")
OUT = (
    Path(__file__).resolve().parents[1]
    / "results" / "figures" / "spectral_gallery.png"
)


def main() -> None:
    datasets = load_all(random_state=SEED)
    methods = list(SPECTRAL_METHODS)
    nrow, ncol = len(DATASETS), len(methods)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow))

    for r, name in enumerate(DATASETS):
        ds = datasets[name]
        for c, m in enumerate(methods):
            ax = axes[r, c]
            ref = np.load(exp_coord_path(EXP, ds.name, m.key, "reference"))
            fw = np.load(exp_coord_path(EXP, ds.name, m.key, "framework"))
            _, fw_aligned, _ = ix.align_procrustes(ref, fw)  # canonical orientation
            ax.scatter(
                fw_aligned[:, 0], fw_aligned[:, 1],
                c=ds.color, cmap=ds.cmap, s=4, alpha=0.85, linewidths=0,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(m.name, fontsize=11)
            if c == 0:
                ax.set_ylabel(ds.name, fontsize=11)

    fig.suptitle(
        "The spectral zoo, one operation: closed-form RV projection of six input "
        "kernels onto the same cone",
        fontsize=13,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
