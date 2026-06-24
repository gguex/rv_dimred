"""
hybrids_run.py  —  §5.3.3  compute & save supervised-interpolation embeddings
=============================================================================

A single interpretable dial β interpolates an unsupervised t-SNE into a fully
class-supervised embedding — something the framework enables by blending kernels,
with no library equivalent:

    K_in(β)  = β·K_class + (1-β)·K_gaussianAffinity(γ=SOFTENING)   (each unit-‖·‖)
    K_out(β) = β·linear(Y) + (1-β)·StudentT(Y, ν=1)                (each unit-‖·‖)

    β = 0 → t-SNE (unsupervised)
    β = 1 → class-centroid kernel (fully supervised; NOT LDA — it collapses each
            class to a point, so the useful regime is intermediate β)

Honest train/test protocol (labelled datasets only): K_class is built from TRAIN
labels; the embedding is optimised on train; TEST points are projected
out-of-sample from their features alone (no test labels) and saved too, so the
class-separation claim can be evaluated on held-out points.

Saves train + projected-test coordinates to results/coordinates/hybrids/ (+
run_meta.csv with β and RV). Indices and figures are built from these by
hybrids_indices.py / hybrids_figures.py.
"""

from __future__ import annotations

import csv
import time

import numpy as np
import torch

from src.benchmark_common import (
    PERPLEXITY,
    SEED,
    SOFTENING,
    SUPERVISED_BETAS,
    coord_path,
    coords_dir,
    get_device,
    meta_path,
    normalize_kernel,
    pca_init,
    project_out_of_sample,
    rv_embed,
    supervised_split,
    to_tensor,
)
from src.datasets import load_all
from src.rv_kernels import (
    compute_class_kernel_torch,
    compute_gaussian_affinity_kernel_torch,
    compute_linear_kernel_torch,
    compute_student_t_kernel_torch,
    default_weights,
)

SECTION = "5.3.3"
FAMILY = "hybrids"
META_FIELDS = ["dataset", "key", "beta", "rv_final"]


def blended_output(
    coords: torch.Tensor,
    param: object = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """K_out(β) = β·linear(Y) + (1-β)·StudentT(Y, ν=1), each unit-Frobenius."""
    beta = float(param)  # type: ignore[arg-type]
    k_lin = compute_linear_kernel_torch(coords, weights=weights, device=device)
    k_t = compute_student_t_kernel_torch(
        coords, param=1.0, weights=weights, device=device
    )
    return beta * normalize_kernel(k_lin) + (1.0 - beta) * normalize_kernel(k_t)


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")
    datasets = load_all(random_state=SEED)
    meta_rows: list[dict[str, object]] = []

    for ds in datasets.values():
        if not ds.has_labels:
            continue  # supervised interpolation needs labels
        print("=" * 64)
        print(f"§{SECTION}  dataset = {ds.name}  (n={ds.n})")
        print("=" * 64)

        X_tr, X_te, y_tr, _ = supervised_split(ds.X, ds.labels)
        X_tr_t = to_tensor(X_tr, device)
        w = default_weights(len(X_tr), device)
        init = pca_init(X_tr)

        k_class = normalize_kernel(
            compute_class_kernel_torch(
                X_tr_t, param={"labels": y_tr}, weights=w, device=device
            )
        )
        k_gauss = normalize_kernel(
            compute_gaussian_affinity_kernel_torch(
                X_tr_t,
                param={"perplexity": PERPLEXITY, "gamma": SOFTENING},
                weights=w,
                device=device,
            )
        )

        t0 = time.time()
        for beta in SUPERVISED_BETAS:
            key = f"supervised_b{beta}"
            k_in = beta * k_class + (1.0 - beta) * k_gauss
            Y_tr, rv = rv_embed(k_in, init, device, blended_output, beta, weights=w)
            Y_te = project_out_of_sample(X_tr, Y_tr, X_te)
            np.save(coord_path(FAMILY, ds.name, key, "train"), Y_tr.astype(np.float32))
            np.save(coord_path(FAMILY, ds.name, key, "test"), Y_te)
            meta_rows.append(
                {"dataset": ds.name, "key": key, "beta": beta, "rv_final": rv}
            )
        print(
            f"  train n={len(X_tr)}  test n={len(X_te)}  "
            f"{len(SUPERVISED_BETAS)} β values  ({time.time() - t0:.1f}s)"
        )

    with meta_path(FAMILY).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"\nSaved coordinates + run_meta.csv → {coords_dir(FAMILY)}")


if __name__ == "__main__":
    main()
