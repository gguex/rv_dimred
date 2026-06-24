"""
test_alignment.py  —  ISOLATED experiment (safe to delete)
==========================================================

Kernel-alignment objective with global output normalization, on MNIST (γ=0.5).

Motivation: RV(K_X, K_Y) is a Frobenius COSINE, hence scale-invariant in K_Y, so
the t-SNE/UMAP global partition function Z(Y) cancels and contributes no repulsion.
Here we drop the ‖K_Y‖ normalization and maximise the (scale-SENSITIVE) alignment

    A(Y) = tr(K_X · Q_norm) / ‖K_X‖ ,   Q_norm = G_offdiag(Y) / Z(Y) ,
    Z(Y) = Σ_{i≠j} G_ij ,   G = raw Student-t / UMAP output kernel (zero diagonal).

Its gradient is
    (1/Z) ∂tr(K_X G)/∂Y  −  [tr(K_X G)/Z²] ∂Z/∂Y
         └ attractive ┘            └ repulsive (the t-SNE-style Z term) ┘
i.e. exactly t-SNE's attraction/repulsion structure, now with a real repulsion.

K_X is the SAME softened, double-centred input kernel as production §5.3.2.
We compare the alignment embedding against the saved RV framework embedding and the
library reference (Procrustes/kNN vs reference, trustworthiness, ARI, within/between).
Nothing here touches production.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src import indices as ix
from src.benchmark_common import (
    K_NEIGHBORS,
    LR_RV,
    N_ITER_RV,
    PERPLEXITY,
    SEED,
    SOFTENING,
    TRUST_K,
    coord_path,
    get_device,
    pca_init,
    to_tensor,
)
from src.datasets import load_mnist
from src.rv_kernels import (
    compute_fuzzy_topological_kernel_torch,
    compute_gaussian_affinity_kernel_torch,
    default_weights,
    pairwise_sq_dists,
)

UMAP_A, UMAP_B = 1.929, 0.7915  # production UMAP output-kernel defaults


def within_between(Y: np.ndarray, labels: np.ndarray) -> float:
    g = Y.mean(0)
    within, between = 0.0, 0.0
    for c in np.unique(labels):
        Yc = Y[labels == c]
        ctr = Yc.mean(0)
        within += ((Yc - ctr) ** 2).sum()
        between += len(Yc) * ((ctr - g) ** 2).sum()
    return float(within / between)


# ── raw (un-centred, zero-diagonal) output affinities ────────────────────────

def raw_student_t(Y: torch.Tensor, nu: float = 1.0) -> torch.Tensor:
    d2 = pairwise_sq_dists(Y)
    G = (1.0 + d2 / nu) ** (-(nu + 1.0) / 2.0)
    return G * (1.0 - torch.eye(G.shape[0], device=Y.device, dtype=G.dtype))


def raw_umap_out(Y: torch.Tensor, a: float = UMAP_A, b: float = UMAP_B) -> torch.Tensor:
    d2 = pairwise_sq_dists(Y)
    G = 1.0 / (1.0 + a * (d2 + 1e-12) ** b)
    return G * (1.0 - torch.eye(G.shape[0], device=Y.device, dtype=G.dtype))


def align_embed(
    K_X: torch.Tensor,
    init: np.ndarray,
    device: str,
    raw_fn: object,
) -> tuple[np.ndarray, float]:
    """Adam ascent on A(Y) = tr(K_X · G/Z) / ‖K_X‖ (output globally normalized)."""
    Kx_norm = torch.sqrt((K_X * K_X).sum())
    Y = to_tensor(init, device).clone().requires_grad_(True)
    opt = torch.optim.Adam([Y], lr=LR_RV)
    for _ in range(N_ITER_RV):
        opt.zero_grad()
        G = raw_fn(Y)
        align = (K_X * G).sum() / (G.sum() * Kx_norm)
        (-align).backward()
        opt.step()
    with torch.no_grad():
        G = raw_fn(Y)
        align = float((K_X * G).sum() / (G.sum() * Kx_norm))
    return Y.detach().cpu().numpy(), align


def metric_row(label: str, align: str, Y: np.ndarray, ref: np.ndarray,
               X: np.ndarray, labels: np.ndarray) -> str:
    if Y is ref:
        proc, knn = "-", "-"
    else:
        proc = f"{ix.procrustes_disparity(Y, ref):.3f}"
        knn = f"{ix.knn_overlap(Y, ref, k=TRUST_K):.3f}"
    return (
        f"{label:<10} {align:>7} {proc:>7} {knn:>6} "
        f"{ix.trustworthiness(X, Y, k=TRUST_K):>7.3f} "
        f"{ix.ari(Y, labels):>6.3f} {within_between(Y, labels):>7.3f}"
    )


HDR = (
    f"{'variant':<10} {'align':>7} {'Proc':>7} {'kNN':>6} "
    f"{'Trust':>7} {'ARI':>6} {'w/b':>7}"
)


def main() -> None:
    device = get_device()
    print(f"device: {device}  (γ={SOFTENING}, MNIST, objective=tr(K_X·G/Z)/‖K_X‖)\n")
    ds = load_mnist(random_state=SEED)
    X_t = to_tensor(ds.X, device)
    w = default_weights(ds.n, device)
    init = pca_init(ds.X)

    K_tsne = compute_gaussian_affinity_kernel_torch(
        X_t,
        param={"perplexity": PERPLEXITY, "gamma": SOFTENING},
        weights=w,
        device=device,
    )
    K_umap = compute_fuzzy_topological_kernel_torch(
        X_t,
        param={"k": K_NEIGHBORS, "gamma": SOFTENING},
        weights=w,
        device=device,
    )

    methods = [
        ("t-SNE", "tsne", K_tsne, raw_student_t),
        ("UMAP", "umap", K_umap, raw_umap_out),
    ]
    for name, key, K_X, raw_fn in methods:
        ref = np.load(coord_path("approximations", ds.name, key, "reference"))
        rv = np.load(coord_path("approximations", ds.name, key, "framework"))
        Y_al, align = align_embed(K_X, init, device, raw_fn)
        print(f"=== {name} ===")
        print(HDR)
        print("-" * len(HDR))
        print(metric_row("reference", "-", ref, ref, ds.X, ds.labels))
        print(metric_row("RV (cos)", "-", rv, ref, ds.X, ds.labels))
        print(metric_row("alignment", f"{align:.4f}", Y_al, ref, ds.X, ds.labels))
        print()


if __name__ == "__main__":
    main()
