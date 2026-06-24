"""
test_hyperparams.py  —  ISOLATED experiment (safe to delete)
============================================================

Hyperparameter sweep for §5.3.2, on MNIST, with the softening fixed at γ=0.5.

Question: do the framework (RV-maximised) and the library reference get *closer*
at some hyperparameter setting?  For every value we run BOTH method and reference
with the SAME hyperparameter and report their agreement plus standalone quality:

  * t-SNE  — sweep perplexity (shared by the adaptive-Gaussian input kernel and
             sklearn TSNE; Student-t output ν=1 on the framework side).
  * UMAP   — sweep n_neighbors (shared by the fuzzy input kernel and umap-learn),
             then sweep min_dist (framework output a,b derived from min_dist via
             umap's find_ab_params, so both sides match).

Columns:
  Proc  = Procrustes disparity framework↔reference (lower = closer)
  kNN   = kNN overlap framework↔reference            (higher = closer)
  T_fw / T_rf = trustworthiness vs X
  A_fw / A_rf = ARI vs labels
  wb_fw / wb_rf = within/between scatter ratio (lower = tighter classes)
  RV    = final RV of the framework embedding
Nothing here touches production.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import umap
from sklearn.manifold import TSNE
from umap.umap_ import find_ab_params

from src import indices as ix
from src.benchmark_common import (
    K_NEIGHBORS,
    PERPLEXITY,
    SEED,
    SOFTENING,
    TRUST_K,
    Q,
    get_device,
    pca_init,
    pca_init_sklearn,
    rv_embed,
    to_tensor,
)
from src.datasets import load_mnist
from src.rv_kernels import (
    compute_fuzzy_topological_kernel_torch,
    compute_gaussian_affinity_kernel_torch,
    default_weights,
)

PERPLEXITIES = [5, 15, 30, 50, 100]
N_NEIGHBORS = [5, 15, 30, 50, 100]
MIN_DISTS = [0.0, 0.1, 0.25, 0.5, 0.8]
SPREAD = 1.0


def within_between(Y: np.ndarray, labels: np.ndarray) -> float:
    g = Y.mean(0)
    within, between = 0.0, 0.0
    for c in np.unique(labels):
        Yc = Y[labels == c]
        ctr = Yc.mean(0)
        within += ((Yc - ctr) ** 2).sum()
        between += len(Yc) * ((ctr - g) ** 2).sum()
    return float(within / between)


# ── framework embeddings (softened input kernel + RV ascent) ─────────────────

def fw_tsne(
    X_t: torch.Tensor,
    w: torch.Tensor,
    init: np.ndarray,
    device: str,
    perplexity: float,
) -> tuple[np.ndarray, float]:
    K = compute_gaussian_affinity_kernel_torch(
        X_t,
        param={"perplexity": perplexity, "gamma": SOFTENING},
        weights=w,
        device=device,
    )
    return rv_embed(K, init, device, "student_t", 1.0, weights=w)


def fw_umap(
    X_t: torch.Tensor,
    w: torch.Tensor,
    init: np.ndarray,
    device: str,
    n_neighbors: int,
    a: float,
    b: float,
) -> tuple[np.ndarray, float]:
    K = compute_fuzzy_topological_kernel_torch(
        X_t,
        param={"k": n_neighbors, "gamma": SOFTENING},
        weights=w,
        device=device,
    )
    return rv_embed(K, init, device, "umap", {"a": a, "b": b}, weights=w)


# ── library references ───────────────────────────────────────────────────────

def ref_tsne(X: np.ndarray, init: np.ndarray, perplexity: float) -> np.ndarray:
    return TSNE(
        n_components=Q, perplexity=perplexity, init=init, random_state=SEED
    ).fit_transform(X)


def ref_umap(
    X: np.ndarray, init: np.ndarray, n_neighbors: int, min_dist: float
) -> np.ndarray:
    return np.asarray(
        umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=Q,
            init=init,
            random_state=SEED,
        ).fit_transform(X)
    )


def row(
    hp_label: str,
    Y_fw: np.ndarray,
    Y_ref: np.ndarray,
    X: np.ndarray,
    labels: np.ndarray,
    rv: float,
) -> str:
    return (
        f"{hp_label:>8} "
        f"{ix.procrustes_disparity(Y_fw, Y_ref):>6.3f} "
        f"{ix.knn_overlap(Y_fw, Y_ref, k=TRUST_K):>6.3f} "
        f"{ix.trustworthiness(X, Y_fw, k=TRUST_K):>6.3f} "
        f"{ix.trustworthiness(X, Y_ref, k=TRUST_K):>6.3f} "
        f"{ix.ari(Y_fw, labels):>6.3f} "
        f"{ix.ari(Y_ref, labels):>6.3f} "
        f"{within_between(Y_fw, labels):>6.3f} "
        f"{within_between(Y_ref, labels):>6.3f} "
        f"{rv:>6.3f}"
    )


HDR = (
    f"{'hp':>8} {'Proc':>6} {'kNN':>6} {'T_fw':>6} {'T_rf':>6} "
    f"{'A_fw':>6} {'A_rf':>6} {'wb_fw':>6} {'wb_rf':>6} {'RV':>6}"
)


def main() -> None:
    device = get_device()
    print(f"device: {device}  (γ={SOFTENING}, MNIST)\n")
    ds = load_mnist(random_state=SEED)
    X_t = to_tensor(ds.X, device)
    w = default_weights(ds.n, device)
    init_fw = pca_init(ds.X)
    init_ref = pca_init_sklearn(ds.X)

    # ── t-SNE: perplexity sweep ───────────────────────────────────────────────
    print(f"=== t-SNE  perplexity sweep  (default={PERPLEXITY}) ===")
    print(HDR)
    print("-" * len(HDR))
    for perp in PERPLEXITIES:
        Y_fw, rv = fw_tsne(X_t, w, init_fw, device, perp)
        Y_ref = np.asarray(ref_tsne(ds.X, init_ref, perp), dtype=np.float32)
        print(row(f"p={perp}", Y_fw, Y_ref, ds.X, ds.labels, rv))
    print()

    # ── UMAP: n_neighbors sweep (min_dist=0.1) ────────────────────────────────
    a0, b0 = find_ab_params(SPREAD, 0.1)
    print(f"=== UMAP  n_neighbors sweep  (default={K_NEIGHBORS}, min_dist=0.1) ===")
    print(HDR)
    print("-" * len(HDR))
    for k in N_NEIGHBORS:
        Y_fw, rv = fw_umap(X_t, w, init_fw, device, k, a0, b0)
        Y_ref = ref_umap(ds.X, init_ref, k, 0.1)
        print(row(f"k={k}", Y_fw, Y_ref, ds.X, ds.labels, rv))
    print()

    # ── UMAP: min_dist sweep (n_neighbors=K_NEIGHBORS) ────────────────────────
    print(f"=== UMAP  min_dist sweep  (n_neighbors={K_NEIGHBORS}) ===")
    print(HDR)
    print("-" * len(HDR))
    for md in MIN_DISTS:
        a, b = find_ab_params(SPREAD, md)
        Y_fw, rv = fw_umap(X_t, w, init_fw, device, K_NEIGHBORS, a, b)
        Y_ref = ref_umap(ds.X, init_ref, K_NEIGHBORS, md)
        print(row(f"md={md}", Y_fw, Y_ref, ds.X, ds.labels, rv))
    print()


if __name__ == "__main__":
    main()
