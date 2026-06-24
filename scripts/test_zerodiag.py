"""
test_zerodiag.py  —  ISOLATED experiment (safe to delete)
=========================================================

Annealed-repulsion test for the §5.3.2 "framework too diffuse" gap, on MNIST.

Setup (fixed per the current question): input affinity softened with γ=0.5
(P → (P/max)^0.5), NORMAL output diagonal (q_ii = 1). After each Adam step on RV
we add an *annealed* repulsive nudge y_i += η_r(t)·r_i, r_i = Σ_j (1+s_ij)^{-1}
(y_i−y_j), with η_r(t) decaying linearly to 0 over the first 60% of iterations.
Because η_r→0, the limit point is a genuine RV critical point (we report ‖∇RV‖),
but the early repulsion biases basin selection toward more separated clusters
(graduated-optimization / early-exaggeration analog).

We sweep η_r0 (0 = pure-RV control) and report Procrustes/kNN vs the library
reference, trustworthiness, ARI, the within/between scatter ratio (w/b, lower =
tighter classes) and ‖∇RV‖. Nothing here touches production.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from src import indices as ix
from src.datasets import load_mnist
from src.rv_kernels import (
    compute_student_t_kernel_torch,
    compute_umap_kernel_torch,
    default_weights,
    double_center,
    pairwise_sq_dists,
    rv_coefficient,
    _perplexity_probabilities,
)
from src.benchmark_common import (
    K_NEIGHBORS,
    LR_RV,
    N_ITER_RV,
    PERPLEXITY,
    SEED,
    coord_path,
    get_device,
    pca_init,
    to_tensor,
)

GAMMA = 0.5
ETA_R0S = [0.0, 0.02, 0.05, 0.1]
ANNEAL_FRAC = 0.6  # repulsion is off after this fraction of iterations


# ── raw (uncentred, zero-diagonal) input affinities, then softened + centred ──
def raw_tsne_affinity(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    sq = (X**2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
    pcond = _perplexity_probabilities(d2, PERPLEXITY)
    return (pcond + pcond.T) / (2.0 * n)


def raw_umap_affinity(X: np.ndarray, k: int = K_NEIGHBORS) -> np.ndarray:
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nbrs.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    rho = dist[:, 0]
    target = np.log2(k)
    W = np.zeros((n, n))
    for i in range(n):
        d = np.maximum(dist[i] - rho[i], 0.0)
        sigma, lo, hi = 1.0, 0.0, np.inf
        for _ in range(64):
            psum = np.exp(-d / sigma).sum()
            if abs(psum - target) < 1e-5:
                break
            if psum > target:
                hi = sigma
                sigma = (lo + hi) / 2
            else:
                lo = sigma
                sigma = sigma * 2 if hi == np.inf else (lo + hi) / 2
        W[i, idx[i]] = np.exp(-d / sigma)
    return W + W.T - W * W.T


def softened_kernel(raw: np.ndarray, gamma: float, w, device) -> torch.Tensor:
    G = (raw / (raw.max() + 1e-12)) ** gamma
    return double_center(to_tensor(G, device), w, device)


def within_between(Y: np.ndarray, labels: np.ndarray) -> float:
    g = Y.mean(0)
    within, between = 0.0, 0.0
    for c in np.unique(labels):
        Yc = Y[labels == c]
        ctr = Yc.mean(0)
        within += ((Yc - ctr) ** 2).sum()
        between += len(Yc) * ((ctr - g) ** 2).sum()
    return float(within / between)


def rv_repulsion_anneal(K_X, init, device, out_fn, out_param, w, eta_r0):
    """Adam ascent on RV with an annealed repulsive nudge after each step."""
    Y = to_tensor(init, device).clone().requires_grad_(True)
    opt = torch.optim.Adam([Y], lr=LR_RV)
    T = max(1, int(ANNEAL_FRAC * N_ITER_RV))
    for t in range(N_ITER_RV):
        opt.zero_grad()
        rv = rv_coefficient(K_X, out_fn(Y, param=out_param, weights=w, device=device))
        (-rv).backward()
        opt.step()
        eta = eta_r0 * max(0.0, 1.0 - t / T)
        if eta > 0:
            with torch.no_grad():
                d2 = pairwise_sq_dists(Y)
                wrep = 1.0 / (1.0 + d2)
                wrep.fill_diagonal_(0.0)
                rep = wrep.sum(1, keepdim=True) * Y - wrep @ Y  # outward push
                rep = rep / (rep.norm() + 1e-12) * (Y.norm() + 1e-12)
                Y += eta * rep
    # final RV and its gradient norm (η_r is 0 at the end → should be a critical pt)
    Yf = Y.detach().clone().requires_grad_(True)
    rv = rv_coefficient(K_X, out_fn(Yf, param=out_param, weights=w, device=device))
    gnorm = float(torch.autograd.grad(rv, Yf)[0].norm())
    return Y.detach().cpu().numpy(), float(rv), gnorm


def main() -> None:
    device = get_device()
    print(f"device: {device}  (γ={GAMMA}, normal diagonal, anneal_frac={ANNEAL_FRAC})\n")
    ds = load_mnist(random_state=SEED)
    w = default_weights(ds.n, device)
    init = pca_init(ds.X)

    methods = [
        ("t-SNE", "tsne", raw_tsne_affinity, compute_student_t_kernel_torch, 1.0),
        ("UMAP", "umap", raw_umap_affinity, compute_umap_kernel_torch, None),
    ]
    hdr = (
        f"{'method':<7} {'eta_r0':>7} {'Proc':>7} {'kNN':>6} {'Trust':>7} "
        f"{'ARI':>6} {'w/b':>7} {'RV':>7} {'|gRV|':>8}"
    )
    for name, key, raw_fn, out_fn, out_param in methods:
        ref = np.load(coord_path("approximations", ds.name, key, "reference"))
        K_X = softened_kernel(raw_fn(ds.X), GAMMA, w, device)
        print(hdr)
        print("-" * len(hdr))
        print(
            f"{name:<7} {'ref':>7} {0.0:>7.4f} {1.0:>6.3f} "
            f"{ix.trustworthiness(ds.X, ref, k=K_NEIGHBORS):>7.4f} "
            f"{ix.ari(ref, ds.labels):>6.3f} {within_between(ref, ds.labels):>7.3f} "
            f"{'-':>7} {'-':>8}"
        )
        for eta_r0 in ETA_R0S:
            Y, rv, g = rv_repulsion_anneal(K_X, init, device, out_fn, out_param, w, eta_r0)
            print(
                f"{name:<7} {eta_r0:>7.2f} {ix.procrustes_disparity(Y, ref):>7.4f} "
                f"{ix.knn_overlap(Y, ref, k=K_NEIGHBORS):>6.3f} "
                f"{ix.trustworthiness(ds.X, Y, k=K_NEIGHBORS):>7.4f} "
                f"{ix.ari(Y, ds.labels):>6.3f} {within_between(Y, ds.labels):>7.3f} "
                f"{rv:>7.4f} {g:>8.2e}"
            )
        print()


if __name__ == "__main__":
    main()
