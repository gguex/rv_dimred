"""
supervised_dial_run.py  —  art. §7.5  the supervised dial (double kernel dial)
==============================================================================

A single dial beta interpolates an unsupervised t-SNE into a fully class-supervised
embedding by moving the INPUT *and* the OUTPUT kernel together — something the RV
framework enables by blending kernels, with no library counterpart:

    K_in(beta)  = (1 - beta) K_ag  + beta K_Z         (each unit-Frobenius)
    K_out(beta) = (1 - beta) StudentT(Y, nu=1) + beta linear(Y)   (each unit-Frob.)
    objective   = HOLLOW RV  (both diagonals zeroed; art. §7.4)

    beta = 0  ->  t-SNE (unsupervised)
    beta = 1  ->  linear <-> class-centroid kernel = cMDS on class centroids, which
                  collapses each class to a point (fully supervised); the useful
                  regime is intermediate beta.

Dialing the output toward linear is what makes supervision pay: only a linear output
can realise the low-rank centroid structure of K_Z (a fixed Student-t output cannot).
K_ag is the adaptive-Gaussian t-SNE affinity softened at gamma = SOFTENING; K_Z is the
inter-class (label) kernel on the linear base. Honest train/test protocol (labelled
datasets only, mnist + singlecell): K_Z is built from TRAIN labels, the embedding is
optimized on train, and TEST points are projected out-of-sample from their features
alone (project_out_of_sample, no test labels), so the generalization of supervision
can be measured on held-out points.

Saves train + projected-test coordinates to results/05_supervised_dial/coordinates/
(+ run_meta.csv with beta and RV). Indices / figure built by the companion scripts.
"""

# ruff: noqa: E402, I001  (imports follow the sys.path bootstrap)
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (for `src`)

import numpy as np

from src.benchmark_common import (  # noqa: E402
    PERPLEXITY,
    SEED,
    SOFTENING,
    SUPERVISED_BETAS,
    exp_coord_path,
    exp_coords_dir,
    exp_meta_path,
    get_device,
    normalize_kernel,
    pca_init,
    project_out_of_sample,
    rv_embed,
    supervised_output_kernel,
    supervised_split,
    to_tensor,
)
from src.datasets import load_all  # noqa: E402
from src.rv_kernels import (  # noqa: E402
    compute_class_kernel_torch,
    default_weights,
    gaussian_affinity_base,
    soften_and_center,
)

EXP = "05_supervised_dial"
DATASETS = ("mnist", "singlecell")
META_FIELDS = ["dataset", "key", "beta", "rv_final"]


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")
    datasets = load_all(random_state=SEED)
    meta_rows: list[dict[str, object]] = []

    for name in DATASETS:
        ds = datasets[name]
        print("=" * 64)
        print(f"art. §7.5  dataset = {ds.name}  (n={ds.n})")
        print("=" * 64)

        X_tr, X_te, y_tr, y_te = supervised_split(ds.X, ds.labels)
        w = default_weights(len(X_tr), device)
        init = pca_init(X_tr)

        base = gaussian_affinity_base(X_tr, PERPLEXITY)
        k_gauss = normalize_kernel(soften_and_center(base, SOFTENING, w, device))
        # K_Z on the linear base: the low-rank class-centroid kernel that a linear
        # output can realise at beta = 1 (cf. the output dial below).
        k_class = normalize_kernel(
            compute_class_kernel_torch(
                to_tensor(X_tr, device),
                param={"labels": y_tr},
                weights=w,
                device=device,
            )
        )

        # save the split labels so indices/figure re-derive metrics on the same points
        np.save(exp_coord_path(EXP, ds.name, "labels", "train"), y_tr)
        np.save(exp_coord_path(EXP, ds.name, "labels", "test"), y_te)

        t0 = time.time()
        for beta in SUPERVISED_BETAS:
            key = f"dial_b{beta}"
            k_in = (1.0 - beta) * k_gauss + beta * k_class
            # output dials with the SAME beta (StudentT -> linear); hollow RV
            Y_tr, rv = rv_embed(
                k_in, init, device, supervised_output_kernel, beta,
                weights=w, hollow=True,
            )
            Y_te = project_out_of_sample(X_tr, Y_tr, X_te)
            np.save(exp_coord_path(EXP, ds.name, key, "train"), Y_tr.astype(np.float32))
            np.save(exp_coord_path(EXP, ds.name, key, "test"), Y_te)
            meta_rows.append(
                {"dataset": ds.name, "key": key, "beta": beta, "rv_final": rv}
            )
        print(
            f"  train n={len(X_tr)}  test n={len(X_te)}  "
            f"{len(SUPERVISED_BETAS)} beta values  ({time.time() - t0:.1f}s)"
        )

    with exp_meta_path(EXP).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"\nSaved coordinates + run_meta.csv → {exp_coords_dir(EXP)}")


if __name__ == "__main__":
    main()
