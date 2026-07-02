"""
hybrids_figures.py  —  §5.3.3  figures from saved coordinates
=============================================================

Rebuilds, from results/coordinates/hybrids/*.npy, for each labelled dataset:
  * sweep    — a 1×|β| panel row of the supervised interpolation: train points
               (grey) with the projected test points coloured by their held-out
               label, so out-of-sample class organisation is visible,
  * quality  — trustworthiness and ARI vs β, on train AND test.

The train/test split is re-derived deterministically (same partition as the run).
Metrics are recomputed from the coordinates so the script is self-contained.
"""

from __future__ import annotations

from typing import Any

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
    fig_path,
    supervised_split,
)
from src.datasets import Dataset, load_all

SECTION = "5.3.3"
FAMILY = "hybrids"


def sweep_panels(
    ds: Dataset, X_te: np.ndarray, y_te: np.ndarray
) -> None:
    n = len(SUPERVISED_BETAS)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.4, 3.6))
    for ax, beta in zip(axes, SUPERVISED_BETAS):
        key = f"supervised_b{beta}"
        Y_tr = np.load(coord_path(FAMILY, ds.name, key, "train"))
        Y_te = np.load(coord_path(FAMILY, ds.name, key, "test"))
        ax.scatter(Y_tr[:, 0], Y_tr[:, 1], c="0.8", s=4, alpha=0.5, linewidths=0)
        ax.scatter(
            Y_te[:, 0], Y_te[:, 1], c=y_te, cmap=ds.cmap, s=10, alpha=0.9, linewidths=0
        )
        end = " (t-SNE)" if beta == 0.0 else (" (class)" if beta == 1.0 else "")
        ax.set_title(f"β={beta}{end}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{ds.name} — supervised t-SNE: train (grey) + projected test (coloured)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(
        fig_path(SECTION, ds.name, "supervised", "sweep"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def quality_curves(
    ds: Dataset, X_tr: np.ndarray, X_te: np.ndarray,
    y_tr: np.ndarray, y_te: np.ndarray,
) -> None:
    metrics: dict[str, list[float]] = {
        "trust_tr": [], "trust_te": [], "ari_tr": [], "ari_te": []
    }
    for beta in SUPERVISED_BETAS:
        key = f"supervised_b{beta}"
        Y_tr = np.load(coord_path(FAMILY, ds.name, key, "train"))
        Y_te = np.load(coord_path(FAMILY, ds.name, key, "test"))
        metrics["trust_tr"].append(ix.trustworthiness(X_tr, Y_tr, k=TRUST_K))
        metrics["trust_te"].append(ix.trustworthiness(X_te, Y_te, k=TRUST_K))
        metrics["ari_tr"].append(ix.ari(Y_tr, y_tr))
        metrics["ari_te"].append(ix.ari(Y_te, y_te))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot: Any = axes[0]
    plot.plot(SUPERVISED_BETAS, metrics["trust_tr"], marker="o", label="train")
    plot.plot(SUPERVISED_BETAS, metrics["trust_te"], marker="s", label="test")
    plot.set_xlabel("β (supervision weight)")
    plot.set_ylabel("Trustworthiness")
    plot.set_title("Trustworthiness vs β")
    plot.legend(fontsize=9)
    plot = axes[1]
    plot.plot(SUPERVISED_BETAS, metrics["ari_tr"], marker="o", label="train")
    plot.plot(SUPERVISED_BETAS, metrics["ari_te"], marker="s", label="test")
    plot.set_xlabel("β (supervision weight)")
    plot.set_ylabel("ARI")
    plot.set_title("ARI vs β  (test = generalisation)")
    plot.legend(fontsize=9)
    fig.suptitle(f"{ds.name} — supervised interpolation: quality vs β", fontsize=11)
    fig.tight_layout()
    fig.savefig(
        fig_path(SECTION, ds.name, "supervised", "quality"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    datasets = load_all(random_state=SEED)
    for ds in datasets.values():
        if not ds.has_labels:
            continue
        X_tr, X_te, y_tr, y_te = supervised_split(ds.X, ds.labels)
        sweep_panels(ds, X_te, y_te)
        quality_curves(ds, X_tr, X_te, y_tr, y_te)
        print(f"  {ds.name}: sweep panels + quality curves written")


if __name__ == "__main__":
    main()
