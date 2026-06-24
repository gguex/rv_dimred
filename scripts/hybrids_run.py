"""
hybrids_run.py  —  §5.3.3  compute & save hybrid embeddings
===========================================================

Two interpolations with no reference implementation — they exist only because the
framework lets you add input kernels and swap the output tail:

  (1) Global-Local × Tail
        input  = α·K_linear + (1-α)·K_adaptiveGaussian   (each unit-Frobenius)
        output = Student-t(ν),  ν ∈ {∞ (Gaussian), 1, 0.5}
        → full α×ν grid per dataset.

  (2) Unsupervised-Supervised   (labelled datasets only)
        input  = α·K_adaptiveGaussian + (1-α)·K_class
        output = Student-t(ν=1)
        → α-sweep.

Only coordinates are saved, to results/coordinates/hybrids/ (+ run_meta.csv with
the sweep parameters and RV). Indices and figures are built from these by
hybrids_indices.py / hybrids_figures.py.
"""

from __future__ import annotations

import csv
import time

import numpy as np

from src.benchmark_common import (
    HYBRID_ALPHAS,
    HYBRID_NUS,
    PERPLEXITY,
    SEED,
    coord_path,
    coords_dir,
    get_device,
    meta_path,
    normalize_kernel,
    pca_init,
    rv_embed,
    to_tensor,
)
from src.datasets import load_all
from src.rv_kernels import (
    compute_class_kernel_torch,
    compute_gaussian_affinity_kernel_torch,
    compute_linear_kernel_torch,
    default_weights,
)

SECTION = "5.3.3"
FAMILY = "hybrids"
META_FIELDS = ["dataset", "family", "key", "alpha", "nu", "rv_final"]


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")
    datasets = load_all(random_state=SEED)
    meta_rows: list[dict[str, object]] = []

    for ds in datasets.values():
        print("=" * 64)
        print(f"§{SECTION}  dataset = {ds.name}  (n={ds.n})")
        print("=" * 64)
        X_t = to_tensor(ds.X, device)
        w = default_weights(ds.n, device)
        init = pca_init(ds.X)

        # ---- (1) Global-Local × Tail ----
        t0 = time.time()
        K_lin = normalize_kernel(
            compute_linear_kernel_torch(X_t, weights=w, device=device)
        )
        K_gauss = normalize_kernel(
            compute_gaussian_affinity_kernel_torch(
                X_t, param={"perplexity": PERPLEXITY}, weights=w, device=device
            )
        )
        for alpha in HYBRID_ALPHAS:
            K_in = alpha * K_lin + (1.0 - alpha) * K_gauss
            for nk, out_kernel, nu in HYBRID_NUS:
                key = f"globallocal_a{alpha}_nu{nk}"
                Y, rv = rv_embed(K_in, init, device, out_kernel, nu, weights=w)
                np.save(
                    coord_path(FAMILY, ds.name, key, "embedding"),
                    Y.astype(np.float32),
                )
                meta_rows.append(
                    {
                        "dataset": ds.name,
                        "family": "globallocal",
                        "key": key,
                        "alpha": alpha,
                        "nu": nk,
                        "rv_final": rv,
                    }
                )
        print(f"  Global-Local × Tail grid done ({time.time() - t0:.1f}s)")

        # ---- (2) Unsupervised-Supervised (labelled only) ----
        if ds.has_labels:
            t0 = time.time()
            K_class = normalize_kernel(
                compute_class_kernel_torch(
                    X_t, param={"labels": ds.labels}, weights=w, device=device
                )
            )
            for alpha in HYBRID_ALPHAS:
                K_in = alpha * K_gauss + (1.0 - alpha) * K_class
                key = f"unsupsup_a{alpha}"
                Y, rv = rv_embed(K_in, init, device, "student_t", 1.0, weights=w)
                np.save(
                    coord_path(FAMILY, ds.name, key, "embedding"),
                    Y.astype(np.float32),
                )
                meta_rows.append(
                    {
                        "dataset": ds.name,
                        "family": "unsupsup",
                        "key": key,
                        "alpha": alpha,
                        "nu": "1.0",
                        "rv_final": rv,
                    }
                )
            print(f"  Unsup-Sup α-sweep done ({time.time() - t0:.1f}s)")

    with meta_path(FAMILY).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"\nSaved coordinates + run_meta.csv → {coords_dir(FAMILY)}")


if __name__ == "__main__":
    main()
