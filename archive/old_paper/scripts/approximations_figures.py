"""
approximations_figures.py  —  §5.3.2  figures from saved coordinates
====================================================================

For each (dataset × method ∈ {t-SNE, UMAP}) builds two separate, Procrustes-aligned
scatter files (framework / reference) from results/coordinates/approximations/, for
side-by-side placement in LaTeX.
"""

from __future__ import annotations

import numpy as np

from src import indices as ix
from src.benchmark_common import SEED, coord_path, save_pair_figures
from src.datasets import load_all

SECTION = "5.3.2"
FAMILY = "approximations"
METHODS = [("t-SNE", "tsne"), ("UMAP", "umap")]


def main() -> None:
    datasets = load_all(random_state=SEED)
    for ds in datasets.values():
        for name, key in METHODS:
            fw = np.load(coord_path(FAMILY, ds.name, key, "framework"))
            ref = np.load(coord_path(FAMILY, ds.name, key, "reference"))
            disparity = ix.procrustes_disparity(fw, ref)
            save_pair_figures(
                SECTION,
                ds,
                key,
                name,
                fw,
                ref,
                fw_note=f"Procrustes={disparity:.4f}",
            )
        print(f"  {ds.name}: framework/reference scatters written")


if __name__ == "__main__":
    main()
