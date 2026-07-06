"""
dial_scatter_figure.py  —  art. §7.5  scatter grid of the supervised dial
=========================================================================

An illustrative companion to Figure 2: for each dataset (row) and dial value beta
(column), the train embedding coloured by class. Reading left to right, the classes
contract and separate as beta rises, collapsing toward points at beta = 1 (the class
kernel removes within-class variance). Writes
results/05_supervised_dial/dial_scatter.{png,pdf}.
"""

# ruff: noqa: E402, I001  (imports follow the sys.path bootstrap)
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (for `src`)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.benchmark_common import (  # noqa: E402
    SUPERVISED_BETAS,
    exp_coord_path,
    exp_coords_dir,
)

EXP = "05_supervised_dial"
DATASETS = ("mnist", "singlecell")


def main() -> None:
    n_col = len(SUPERVISED_BETAS)
    fig, axes = plt.subplots(2, n_col, figsize=(4 * n_col, 8))
    for row, ds in enumerate(DATASETS):
        labels = np.load(exp_coord_path(EXP, ds, "labels", "train"))
        _, codes = np.unique(labels, return_inverse=True)
        cmap = "tab10" if len(np.unique(codes)) <= 10 else "tab20"
        for col, beta in enumerate(SUPERVISED_BETAS):
            ax = axes[row, col]
            Y = np.load(exp_coord_path(EXP, ds, f"dial_b{beta}", "train"))
            ax.scatter(Y[:, 0], Y[:, 1], c=codes, cmap=cmap, s=6,
                       alpha=0.8, linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(rf"$\beta = {beta}$", fontsize=13)
            if col == 0:
                ax.set_ylabel(ds, fontsize=13)
    fig.suptitle(
        "Supervised dial: t-SNE (β=0) → class kernel (β=1), train embeddings",
        fontsize=14,
    )
    fig.tight_layout()
    out = exp_coords_dir(EXP).parent / "dial_scatter"
    fig.savefig(f"{out}.png", dpi=170, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}.png and {out}.pdf")


if __name__ == "__main__":
    main()
