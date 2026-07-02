"""
approximations_run.py  —  §5.3.2  compute & save approximation embeddings
=========================================================================

t-SNE and UMAP across the 3 datasets. For every (dataset × method) it computes:
  * framework — adaptive-Gaussian / fuzzy input kernel (softened with gamma=
                SOFTENING, §5.3.2) + Student-t / UMAP output, RV-maximised from
                the shared full-scale PCA init,
  * reference — the library implementation (sklearn t-SNE / umap-learn), started
                from the shared PCA init (1e-4 convention) → deterministic.

Both sides use the per-dataset hyperparameter from APPROX_HYPERPARAMS (perplexity
for t-SNE, n_neighbors for UMAP), so the framework and its reference are matched.

Only coordinates are saved, to results/coordinates/approximations/ (+ run_meta.csv
for the framework RV and the hyperparameter). Indices and figures are built from
these by approximations_indices.py / approximations_figures.py.
"""

from __future__ import annotations

import csv
import time

import numpy as np
import umap
from sklearn.manifold import TSNE

from src.benchmark_common import (
    APPROX_HYPERPARAMS,
    SEED,
    SOFTENING,
    Q,
    coord_path,
    coords_dir,
    get_device,
    meta_path,
    pca_init,
    pca_init_sklearn,
    rv_embed,
    to_tensor,
)
from src.datasets import load_all
from src.rv_kernels import (
    compute_fuzzy_topological_kernel_torch,
    compute_gaussian_affinity_kernel_torch,
    default_weights,
)

SECTION = "5.3.2"
FAMILY = "approximations"
META_FIELDS = ["dataset", "method_key", "method", "hyperparam", "rv_final"]


def tsne_ref(X: np.ndarray, init: np.ndarray, perplexity: int) -> np.ndarray:
    return TSNE(
        n_components=Q, perplexity=perplexity, init=init, random_state=SEED
    ).fit_transform(X)


def umap_ref(X: np.ndarray, init: np.ndarray, n_neighbors: int) -> np.ndarray:
    return np.asarray(
        umap.UMAP(
            n_neighbors=n_neighbors, n_components=Q, init=init, random_state=SEED
        ).fit_transform(X)
    )


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")
    datasets = load_all(random_state=SEED)
    meta_rows: list[dict[str, object]] = []

    for ds in datasets.values():
        print("=" * 64)
        print(f"§{SECTION}  dataset = {ds.name}  (n={ds.n}, d={ds.X.shape[1]})")
        print("=" * 64)
        X_t = to_tensor(ds.X, device)
        w = default_weights(ds.n, device)
        init_fw = pca_init(ds.X)  # full-scale PCA — framework optimiser
        init_ref = pca_init_sklearn(ds.X)  # 1e-4 PCA — sklearn/UMAP convention

        hp = APPROX_HYPERPARAMS[ds.name]
        perp, k = hp["perplexity"], hp["n_neighbors"]

        K_gauss = compute_gaussian_affinity_kernel_torch(
            X_t,
            param={"perplexity": perp, "gamma": SOFTENING},
            weights=w,
            device=device,
        )
        K_fuzzy = compute_fuzzy_topological_kernel_torch(
            X_t,
            param={"k": k, "gamma": SOFTENING},
            weights=w,
            device=device,
        )

        configs = (
            ("t-SNE", "tsne", K_gauss, "student_t", 1.0, perp, tsne_ref),
            ("UMAP", "umap", K_fuzzy, "umap", None, k, umap_ref),
        )
        for name, key, K_in, out_kernel, out_param, hparam, ref_fn in configs:
            t0 = time.time()
            Y_fw, rv = rv_embed(K_in, init_fw, device, out_kernel, out_param, weights=w)
            np.save(
                coord_path(FAMILY, ds.name, key, "framework"), Y_fw.astype(np.float32)
            )
            Y_ref = np.asarray(ref_fn(ds.X, init_ref, hparam), dtype=np.float32)
            np.save(coord_path(FAMILY, ds.name, key, "reference"), Y_ref)
            meta_rows.append(
                {
                    "dataset": ds.name,
                    "method_key": key,
                    "method": name,
                    "hyperparam": hparam,
                    "rv_final": rv,
                }
            )
            print(f"  {name:<6} hp={hparam:<4} rv={rv:.4f}  ({time.time() - t0:.1f}s)")

    with meta_path(FAMILY).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"\nSaved coordinates + run_meta.csv → {coords_dir(FAMILY)}")


if __name__ == "__main__":
    main()
