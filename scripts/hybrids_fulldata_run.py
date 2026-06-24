"""
hybrids_fulldata_run.py  —  §5.3.3  full-data illustration coordinates
======================================================================

ILLUSTRATION companion to hybrids_run.py (no train/test split). Runs the same
supervised interpolation (class-kernel ↔ t-SNE) but on ALL points of each
labelled dataset, so the β sweep can be shown as a single clean evolution of the
whole cloud. This is purely for visualisation — the honest, evaluable results
live in the train/test pipeline (hybrids_run.py).

    K_in(β)  = β·K_class + (1-β)·K_gaussianAffinity(γ=SOFTENING)
    K_out(β) = β·linear(Y) + (1-β)·StudentT(Y, ν=1)

Saves coordinates to results/coordinates/hybrids/{dataset}__supervised_b{β}__full.npy
(+ run_meta_full.csv with β and RV). Figures are built by hybrids_fulldata_figures.py.
"""

from __future__ import annotations

import csv
import time

import numpy as np

from src.benchmark_common import (
    PERPLEXITY,
    SEED,
    SOFTENING,
    SUPERVISED_BETAS,
    coord_path,
    coords_dir,
    get_device,
    normalize_kernel,
    pca_init,
    rv_embed,
    supervised_output_kernel,
    to_tensor,
)
from src.datasets import load_all
from src.rv_kernels import (
    compute_class_kernel_torch,
    compute_gaussian_affinity_kernel_torch,
    default_weights,
)

SECTION = "5.3.3"
FAMILY = "hybrids"
META_FIELDS = ["dataset", "key", "beta", "rv_final"]
META_NAME = "run_meta_full.csv"


def main() -> None:
    device = get_device()
    print(f"device: {device}  (full-data illustration)\n")
    datasets = load_all(random_state=SEED)
    meta_rows: list[dict[str, object]] = []

    for ds in datasets.values():
        if not ds.has_labels:
            continue
        print(f"§{SECTION}  dataset = {ds.name}  (n={ds.n})")
        X_t = to_tensor(ds.X, device)
        w = default_weights(ds.n, device)
        init = pca_init(ds.X)

        k_class = normalize_kernel(
            compute_class_kernel_torch(
                X_t, param={"labels": ds.labels}, weights=w, device=device
            )
        )
        k_gauss = normalize_kernel(
            compute_gaussian_affinity_kernel_torch(
                X_t,
                param={"perplexity": PERPLEXITY, "gamma": SOFTENING},
                weights=w,
                device=device,
            )
        )

        t0 = time.time()
        for beta in SUPERVISED_BETAS:
            key = f"supervised_b{beta}"
            k_in = beta * k_class + (1.0 - beta) * k_gauss
            Y, rv = rv_embed(
                k_in, init, device, supervised_output_kernel, beta, weights=w
            )
            np.save(coord_path(FAMILY, ds.name, key, "full"), Y.astype(np.float32))
            meta_rows.append(
                {"dataset": ds.name, "key": key, "beta": beta, "rv_final": rv}
            )
        print(f"  {len(SUPERVISED_BETAS)} β values  ({time.time() - t0:.1f}s)")

    meta = coords_dir(FAMILY) / META_NAME
    with meta.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"\nSaved full-data coordinates + {META_NAME} → {coords_dir(FAMILY)}")


if __name__ == "__main__":
    main()
