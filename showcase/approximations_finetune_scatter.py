"""
approximations_finetune_scatter.py  —  embedding montages of the hollow-RV sweep
================================================================================

Per (dataset × method), a montage of the saved embeddings: rows = the neighbour
hyperparameter (perplexity / n_neighbors), columns = the softening lambda, plus a
final column for the library reference. Every framework panel is Procrustes-aligned
to its row's reference, so orientation/scale are comparable across the whole grid
and against the reference column. Points are coloured by the dataset's labels /
manifold parameter.

Reads results/coordinates/approximations_finetune/*.npy; writes
results/figures/approximations_finetune/montage_{dataset}_{method}.png.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import indices as ix
from src.benchmark_common import RESULTS_DIR, SEED, coord_path
from src.datasets import Dataset, load_all
from scripts.approximations_finetune_run import (
    FAMILY,
    LAMBDAS,
    METHODS,
    fw_key,
    ref_key,
)


def fig_dir():
    d = RESULTS_DIR / "figures" / FAMILY
    d.mkdir(parents=True, exist_ok=True)
    return d


def panel(ax, ds: Dataset, Y: np.ndarray) -> None:
    ax.scatter(Y[:, 0], Y[:, 1], c=ds.color, cmap=ds.cmap, s=4, alpha=0.8, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])


def montage(ds: Dataset, name: str, mkey: str, hp_name: str, hp_vals: list) -> None:
    ncol = len(LAMBDAS) + 1  # lambda columns + reference
    nrow = len(hp_vals)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.3 * nrow))
    axes = np.atleast_2d(axes)

    for i, hp in enumerate(hp_vals):
        ref = np.load(coord_path(FAMILY, ds.name, ref_key(mkey, hp), "reference"))
        ref_std = None
        for j, lam in enumerate(LAMBDAS):
            fw = np.load(coord_path(FAMILY, ds.name, fw_key(mkey, hp, lam), "framework"))
            ref_std, fw_al, d = ix.align_procrustes(ref, fw)
            panel(axes[i, j], ds, fw_al)
            axes[i, j].set_title(f"Δ={d:.3f}", fontsize=8)
            if i == 0:
                axes[i, j].set_title(f"λ={lam:g}\nΔ={d:.3f}", fontsize=8)
        panel(axes[i, ncol - 1], ds, ref_std)
        if i == 0:
            axes[i, ncol - 1].set_title("reference", fontsize=9)
        axes[i, 0].set_ylabel(f"{hp_name}={hp}", fontsize=9)

    fig.suptitle(
        f"{ds.name} — {name}: hollow-RV embeddings ({hp_name} × λ), "
        f"framework Procrustes-aligned to reference (Δ = disparity)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = fig_dir() / f"montage_{ds.name}_{mkey}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> None:
    datasets = load_all(random_state=SEED)
    for ds in datasets.values():
        for name, mkey, hp_name, hp_vals, *_ in METHODS:
            montage(ds, name, mkey, hp_name, hp_vals)


if __name__ == "__main__":
    main()
