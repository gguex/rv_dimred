"""
diagnose_recovery.py  —  diagnostics for §5.3.1 identity recovery
=================================================================

Read-only experiments (no production code changed) to localise why LLE /
Laplacian Eigenmaps / Diffusion Maps do not recover their references as cleanly
as PCA / Isomap / Kernel PCA.

  Test 1  per-axis scaling : Procrustes raw vs after z-scoring each axis. If the
          disparity collapses, the mismatch is the framework's √λ axis-scaling
          convention, not the kernel formula.
  Test 2  Laplacian "sklearn-style" kernel : binary kNN connectivity + normalised
          Laplacian (± degree weights) vs the heat/unnormalised production kernel.
  Test 3  LLE regularisation : sklearn's R = reg·trace(C) (no /k) vs production /k.
  Test 4  Diffusion exponent : framework gives axis ∝ exp(-t·μ/2); reference uses
          exp(-t·μ). Compare framework to both a full- and half-exponent reference.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import eigh as scipy_eigh
from scipy.linalg import pinvh
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from src import indices as ix
from src.datasets import load_all
from src.rv_kernels import (
    compute_diffusion_kernel_torch,
    compute_laplacian_kernel_torch,
    compute_lle_kernel_torch,
    default_weights,
    double_center,
)
from src.section5_common import (
    DIFFUSION_T,
    K_NEIGHBORS,
    diffusion_maps_ref,
    get_device,
    pca_init,
    rv_embed,
    to_tensor,
)


def zscore(Y: np.ndarray) -> np.ndarray:
    return (Y - Y.mean(0)) / (Y.std(0) + 1e-12)


def proc_pair(Y_ref: np.ndarray, Y_fw: np.ndarray) -> tuple[float, float]:
    """(raw Procrustes, per-axis-standardised Procrustes)."""
    raw = ix.procrustes_disparity(Y_ref, Y_fw)
    std = ix.procrustes_disparity(zscore(Y_ref), zscore(Y_fw))
    return raw, std


# ── Test 2 helpers: sklearn-style Laplacian kernels ───────────────────────────
def laplacian_kernel_sklearn_style(
    X: np.ndarray, k: int, device: str, degree_weights: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary symmetric kNN connectivity + normalised Laplacian, G = pinv(L_sym).
    Returns (centred kernel, weights) so rv_embed can use matching weights."""
    n = X.shape[0]
    A = kneighbors_graph(X, k, include_self=True)
    A = 0.5 * (A + A.T)
    A = np.asarray(A.todense())
    L_sym = csgraph_laplacian(A, normed=True)  # I - D^-1/2 A D^-1/2
    G = pinvh(L_sym)
    if degree_weights:
        d = A.sum(1)
        f = torch.as_tensor(d / d.sum(), dtype=torch.float32, device=device)
    else:
        f = default_weights(n, device)
    K = double_center(to_tensor(G, device), f, device)
    return K, f


# ── Test 3 helper: LLE kernel with sklearn's regularisation (no /k) ───────────
def lle_kernel_sklearn_reg(
    X: np.ndarray, k: int, device: str, reg: float = 1e-3
) -> torch.Tensor:
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nbrs.kneighbors(X)
    idx = idx[:, 1:]
    W = np.zeros((n, n))
    ones = np.ones(k)
    for i in range(n):
        Z = X[idx[i]] - X[i]
        C = Z @ Z.T
        trace = np.trace(C)
        R = reg * trace if trace > 0 else reg  # sklearn: no /k
        C.flat[:: k + 1] += R
        wi = np.linalg.solve(C, ones)
        wi /= wi.sum()
        W[i, idx[i]] = wi
    Iu = np.eye(n)
    Phi = (Iu - W).T @ (Iu - W)
    G = pinvh(Phi)
    w = default_weights(n, device)
    return double_center(to_tensor(G, device), w, device)


# ── Test 4 helper: diffusion reference with custom exponent ────────────────────
def diffusion_ref_exponent(
    X: np.ndarray, k: int, t: float, q: int, exponent: float
) -> np.ndarray:
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nbrs.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    sigma = float(np.median(dist[:, -1])) + 1e-12
    A = np.zeros((n, n))
    for i in range(n):
        A[i, idx[i]] = np.exp(-(dist[i] ** 2) / sigma**2)
    A = np.maximum(A, A.T)
    dinv = 1.0 / np.sqrt(A.sum(1) + 1e-12)
    L_sym = np.eye(n) - dinv[:, None] * A * dinv[None, :]
    vals, vecs = scipy_eigh(L_sym, subset_by_index=[0, q])
    return (vecs[:, 1 : q + 1] * np.exp(-exponent * t * vals[1 : q + 1])).astype(
        np.float32
    )


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")
    datasets = load_all()

    for ds in datasets.values():
        X, init = ds.X, pca_init(ds.X)
        w = default_weights(ds.n, device)
        X_t = to_tensor(X, device)
        print("=" * 70)
        print(f"dataset = {ds.name}  (n={ds.n})")
        print("=" * 70)

        # ---- Test 1: production kernels, raw vs per-axis-standardised ----
        print("[Test 1] per-axis scaling (raw → z-scored Procrustes)")
        K_lle = compute_lle_kernel_torch(
            X_t, param={"k": K_NEIGHBORS}, weights=w, device=device
        )
        Y = rv_embed(K_lle, init, device, "linear", weights=w)[0]
        ref = LocallyLinearEmbedding(
            n_neighbors=K_NEIGHBORS, n_components=2, random_state=0
        ).fit_transform(X)
        raw, std = proc_pair(ref, Y)
        print(f"  LLE             {raw:.4f} → {std:.4f}")

        K_lap = compute_laplacian_kernel_torch(
            X_t, param={"k": K_NEIGHBORS}, weights=w, device=device
        )
        Y = rv_embed(K_lap, init, device, "linear", weights=w)[0]
        ref = SpectralEmbedding(
            n_neighbors=K_NEIGHBORS, n_components=2, random_state=0
        ).fit_transform(X)
        raw, std = proc_pair(ref, Y)
        print(f"  Laplacian       {raw:.4f} → {std:.4f}")

        K_diff = compute_diffusion_kernel_torch(
            X_t, param={"k": K_NEIGHBORS, "t": DIFFUSION_T}, weights=w, device=device
        )
        Y = rv_embed(K_diff, init, device, "linear", weights=w)[0]
        ref = diffusion_maps_ref(X)
        raw, std = proc_pair(ref, Y)
        print(f"  Diffusion       {raw:.4f} → {std:.4f}")

        # ---- Test 2: Laplacian sklearn-style kernels ----
        print("[Test 2] Laplacian sklearn-style kernel vs SpectralEmbedding")
        ref = SpectralEmbedding(
            n_neighbors=K_NEIGHBORS, n_components=2, random_state=0
        ).fit_transform(X)
        for dw in (False, True):
            K, f = laplacian_kernel_sklearn_style(
                X, K_NEIGHBORS, device, degree_weights=dw
            )
            Y = rv_embed(K, init, device, "linear", weights=f)[0]
            raw, std = proc_pair(ref, Y)
            tag = "binary+norm+degW" if dw else "binary+norm+uniW"
            print(f"  {tag:20s} {raw:.4f} → {std:.4f}")

        # ---- Test 3: LLE reg alignment ----
        print("[Test 3] LLE reg = sklearn (no /k) vs SpectralEmbedding-LLE")
        ref = LocallyLinearEmbedding(
            n_neighbors=K_NEIGHBORS, n_components=2, random_state=0
        ).fit_transform(X)
        K = lle_kernel_sklearn_reg(X, K_NEIGHBORS, device)
        Y = rv_embed(K, init, device, "linear", weights=w)[0]
        raw, std = proc_pair(ref, Y)
        print(f"  LLE reg-aligned {raw:.4f} → {std:.4f}")

        # ---- Test 4: diffusion exponent ----
        print("[Test 4] framework diffusion vs full- (exp) and half- (exp/2) reference")
        Y = rv_embed(K_diff, init, device, "linear", weights=w)[0]
        ref_full = diffusion_ref_exponent(X, K_NEIGHBORS, DIFFUSION_T, 2, exponent=1.0)
        ref_half = diffusion_ref_exponent(X, K_NEIGHBORS, DIFFUSION_T, 2, exponent=0.5)
        rf, _ = proc_pair(ref_full, Y)
        rh, _ = proc_pair(ref_half, Y)
        print(f"  vs full-exponent ref  {rf:.4f}")
        print(f"  vs half-exponent ref  {rh:.4f}  (matches framework's √λ scaling)")

        # ---- Test 5: larger t for diffusion ----
        print("[Test 5] diffusion: larger t values (spectral gap effect)")
        for t_val in (1.0, 2.0, 5.0, 10.0, 20.0):
            K_d = compute_diffusion_kernel_torch(
                X_t, param={"k": K_NEIGHBORS, "t": t_val}, weights=w, device=device
            )
            Y_d = rv_embed(K_d, init, device, "linear", weights=w)[0]
            ref_d = diffusion_ref_exponent(X, K_NEIGHBORS, t_val, 2, exponent=1.0)
            raw, _ = proc_pair(ref_d, Y_d)
            print(f"  t={t_val:5.1f}  procrustes={raw:.4f}")
        print()


if __name__ == "__main__":
    main()
