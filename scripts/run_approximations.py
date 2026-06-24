"""
run_approximations.py  —  §5.3.2  Approximations (geometric, not bit-exact)
==========================================================================

t-SNE and UMAP across the 3 datasets. The framework (adaptive-Gaussian / fuzzy
input kernel + Student-t / UMAP output) is compared to the named library
implementation. Both the framework and the references use the **shared PCA
initialisation** (sklearn / UMAP convention for the references), so the
Procrustes comparison is legitimate and deterministic.

Outputs
  * indices appended to results/results_indices.csv (identity: Procrustes, kNN
    overlap; quality: trustworthiness, ARI for both framework and reference),
  * two separate scatter figures (framework / reference) per (method × dataset).

This driver first drops any old §5.3.2 rows, so re-running is idempotent and
leaves §5.3.1 / §5.3.3 untouched.
"""

from __future__ import annotations

import time

import numpy as np
import umap
from sklearn.manifold import TSNE

from src import indices as ix
from src.datasets import Dataset, load_all
from src.rv_kernels import (
    compute_fuzzy_topological_kernel_torch,
    compute_gaussian_affinity_kernel_torch,
    default_weights,
)
from src.section5_common import (
    K_NEIGHBORS,
    PERPLEXITY,
    SEED,
    TRUST_K,
    IndexLog,
    Q,
    drop_sections,
    get_device,
    pca_init,
    pca_init_sklearn,
    rv_embed,
    save_pair_figures,
    to_tensor,
)

SECTION = "5.3.2"


# ── Reference embeddings (PCA init only) ──────────────────────────────────────
def tsne_ref(X: np.ndarray, init: np.ndarray) -> np.ndarray:
    return TSNE(
        n_components=Q,
        perplexity=PERPLEXITY,
        init=init,
        random_state=SEED,
    ).fit_transform(X)


def umap_ref(X: np.ndarray, init: np.ndarray) -> np.ndarray:
    return np.asarray(
        umap.UMAP(
            n_neighbors=K_NEIGHBORS,
            n_components=Q,
            init=init,
            random_state=SEED,
        ).fit_transform(X)
    )


def log_quality(
    log: IndexLog, ds: Dataset, method: str, variant: str, Y: np.ndarray
) -> None:
    log.add(
        SECTION,
        ds.name,
        method,
        "trustworthiness",
        ix.trustworthiness(ds.X, Y, k=TRUST_K),
        variant=variant,
        k=TRUST_K,
    )
    if ds.has_labels:
        log.add(SECTION, ds.name, method, "ari", ix.ari(Y, ds.labels), variant=variant)


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")
    log = IndexLog()
    datasets = load_all(random_state=SEED)

    for ds in datasets.values():
        print("=" * 64)
        print(f"§{SECTION}  dataset = {ds.name}  (n={ds.n}, d={ds.X.shape[1]})")
        print("=" * 64)
        X_t = to_tensor(ds.X, device)
        w = default_weights(ds.n, device)
        init_fw = pca_init(ds.X)  # full-scale PCA — framework optimiser
        init_ref = pca_init_sklearn(ds.X)  # 1e-4 PCA — sklearn/UMAP convention

        # framework input kernels
        K_gauss = compute_gaussian_affinity_kernel_torch(
            X_t, param={"perplexity": PERPLEXITY}, weights=w, device=device
        )
        K_fuzzy = compute_fuzzy_topological_kernel_torch(
            X_t, param={"k": K_NEIGHBORS}, weights=w, device=device
        )

        # framework embeddings (PCA init, deterministic)
        t0 = time.time()
        Y_fw_tsne, rv_t = rv_embed(
            K_gauss, init_fw, device, "student_t", 1.0, weights=w
        )
        Y_fw_umap, rv_u = rv_embed(K_fuzzy, init_fw, device, "umap", weights=w)
        dt = time.time() - t0
        print(f"  framework t-SNE RV={rv_t:.4f}  UMAP RV={rv_u:.4f}  ({dt:.1f}s)")

        for label, ref_fn, Y_fw, rv_fw, key in (
            ("t-SNE", tsne_ref, Y_fw_tsne, rv_t, "tsne"),
            ("UMAP", umap_ref, Y_fw_umap, rv_u, "umap"),
        ):
            t0 = time.time()
            Y_ref = ref_fn(ds.X, init_ref)  # PCA-init reference (seed 0)

            disparity = ix.procrustes_disparity(Y_fw, Y_ref)
            knn = ix.knn_overlap(Y_fw, Y_ref, k=TRUST_K)
            log.add(SECTION, ds.name, label, "procrustes", disparity, k=TRUST_K)
            log.add(SECTION, ds.name, label, "knn_overlap", knn, k=TRUST_K)
            log.add(SECTION, ds.name, label, "rv_final", rv_fw, variant="framework")
            log_quality(log, ds, label, "framework", Y_fw)
            log_quality(log, ds, label, "reference", Y_ref)

            save_pair_figures(
                SECTION,
                ds,
                key,
                label,
                Y_fw,
                Y_ref,
                fw_note=f"Procrustes={disparity:.4f}  RV={rv_fw:.4f}",
            )
            print(
                f"  {label:<6} proc={disparity:.4f} knn={knn:.3f} "
                f"rv={rv_fw:.4f}  ({time.time() - t0:.1f}s)"
            )

    drop_sections({SECTION})  # idempotent re-run; keep §5.3.1 / §5.3.3 rows
    path = log.write(append=True)
    print(f"\nAppended indices → {path}")


if __name__ == "__main__":
    main()
