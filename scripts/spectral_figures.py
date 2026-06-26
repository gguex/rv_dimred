"""
spectral_figures.py  —  figures from saved spectral coordinates
===============================================================

Builds, from results/coordinates/spectral/*.npy, per (dataset × method) two
Procrustes-aligned scatters in separate files (framework_linear, reference) for
side-by-side placement. Both panels share the same Procrustes frame so the
closed-form spectral embedding and the library reference are directly comparable.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import indices as ix
from src.benchmark_common import (
    SEED,
    SPECTRAL_METHODS,
    coord_path,
    fig_path,
)
from src.datasets import Dataset, load_all

FAMILY = "spectral"


def scatter(
    ds: Dataset, method_key: str, variant: str, Y: np.ndarray, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        Y[:, 0], Y[:, 1], c=ds.color, cmap=ds.cmap, s=10, alpha=0.85, linewidths=0
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(
        fig_path("", ds.name, method_key, variant), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    datasets = load_all(random_state=SEED)

    for ds in datasets.values():
        for m in SPECTRAL_METHODS:
            ref = np.load(coord_path(FAMILY, ds.name, m.key, "reference"))
            fw = np.load(coord_path(FAMILY, ds.name, m.key, "framework_linear"))
            d = ix.procrustes_disparity(fw, ref)
            ref_std, fw_aligned, _ = ix.align_procrustes(ref, fw)
            scatter(
                ds,
                m.key,
                "linear",
                fw_aligned,
                f"{ds.name} — {m.name} (linear)\nProcrustes={d:.4f}",
            )
            scatter(
                ds, m.key, "reference", ref_std, f"{ds.name} — {m.name} (reference)"
            )
        print(f"  {ds.name}: scatters written")


if __name__ == "__main__":
    main()
