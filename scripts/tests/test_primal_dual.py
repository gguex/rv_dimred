"""
test_primal_dual.py  —  TEST C (new_article_plan.md §6, lead I.3)
=================================================================

Primal-dual prototype of the PUSH/PULL dynamics.  We reproduce the t-SNE-style
attraction-repulsion WITHOUT a KL divergence or a partition function, using only
kernels and a single Lagrange multiplier.

Same conditions as test_directional_solver.py:
  reduced MNIST (N_PER_DIGIT=50 -> n=500), Student-t output, K_X = adaptive
  Gaussian affinity (perplexity=30) with gamma=0.5 softening, PCA init.

Objective (maximize over Y, with lambda >= 0):
    L(Y, lambda) = <K_X, K_Y>            (PULL  -- Hilbert-Schmidt alignment)
                 - lambda * (vbar(Y) - c)   (PUSH  -- volume / crowding penalty)
where
    G_Y   = 1 / (1 + D^2(Y))             raw Student-t Gram   (diag = 1)
    K_Y   = Q G_Y Q^T                    centered output kernel in K_n
    vbar  = mean off-diagonal entry of G_Y  in (0,1)          <-- RAW Gram (key!)

KEY CAVEAT (memory rv-cosine-scale-invariance): the volume constraint MUST act on
the RAW Gram G_Y, NOT on the centered K_Y.  The RV cosine is scale-invariant, so
no penalty on K_Y can inject repulsion.  vbar on the raw Gram is exactly the
partition-function surrogate that t-SNE gets from its normalization.

Dual ascent enforces  vbar = c  (target mean affinity):
    lambda <- max(0, lambda + rho * (vbar - c)).
If vbar > c (too crowded) lambda rises -> stronger repulsion -> points spread.

We report, for (a) pure alignment lambda=0, (b) a fixed-lambda sweep, (c) the full
primal-dual with dual ascent, and (d) the reference gradient t-SNE (rv_embed):
RV, ARI (k-means vs labels), mean pairwise spread, final lambda.  Scatter plots are
saved to results/figures/temp/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from src.benchmark_common import (
    PERPLEXITY,
    SEED,
    SOFTENING,
    normalize_kernel,
    pca_init,
    to_tensor,
)
from src.datasets import load_mnist
from src.rv_kernels import (
    compute_gaussian_affinity_kernel_torch,
    default_weights,
    double_center,
    rv_coefficient,
)

N_PER_DIGIT = 50          # reduced MNIST: n = 500 (same as directional solver test)
D = 2
N_ITER = 800             # primal gradient steps
LR = 0.1                 # Adam lr on Y (match the rv_embed reference)
RHO = 2.0                # dual step size
DUAL_EVERY = 5           # primal steps between dual updates
TARGET_FRAC = 0.5        # target mean affinity c = TARGET_FRAC * vbar(PULL-only)
FIG_DIR = Path("results/figures/temp")
DEV = "cpu"


# ── output-kernel building blocks (torch, differentiable in Y) ────────────────

def gram_student(Y: torch.Tensor) -> torch.Tensor:
    """Raw Student-t Gram 1/(1+D^2), diag = 1."""
    d2 = torch.cdist(Y, Y) ** 2
    return 1.0 / (1.0 + d2)


def mean_offdiag(G: torch.Tensor) -> torch.Tensor:
    n = G.shape[0]
    return (G.sum() - torch.diagonal(G).sum()) / (n * (n - 1))


# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(Y: np.ndarray, K_X: torch.Tensor, w: torch.Tensor,
            labels: np.ndarray) -> tuple[float, float, float]:
    Yt = to_tensor(Y, DEV)
    K_Y = double_center(gram_student(Yt), w, DEV)
    rv = float(rv_coefficient(K_X, K_Y))
    km = KMeans(n_clusters=len(np.unique(labels)), n_init=10,
                random_state=SEED).fit_predict(Y)
    ari = float(adjusted_rand_score(labels, km))
    spread = float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))  # RMS radius
    return rv, ari, spread


# ── primal-dual / fixed-lambda optimizer ──────────────────────────────────────

def optimize(
    K_X: torch.Tensor, w: torch.Tensor, init: np.ndarray,
    lam0: float = 0.0, c: float | None = None, adapt_dual: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """Maximize <K_X,K_Y> - lam*(vbar - c) over Y by Adam.
    If adapt_dual, ascend lam toward the volume target c; else hold lam = lam0."""
    Y = torch.tensor(init, dtype=torch.float32, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([Y], lr=LR)
    lam = float(lam0)
    lam_traj = [lam]
    for it in range(N_ITER):
        opt.zero_grad()
        G = gram_student(Y)
        K_Y = double_center(G, w, DEV)
        K_Y_frob = (K_Y * K_Y).sum().sqrt().clamp_min(1e-10)
        align = (K_X * K_Y).sum() / K_Y_frob   # RV normalisé (||K_X||=1)
        vbar = mean_offdiag(G)                  # crowding on RAW Gram (PUSH target)
        loss = -(align - lam * vbar)            # maximize L => minimize -L
        loss.backward()
        opt.step()
        if adapt_dual and c is not None and (it + 1) % DUAL_EVERY == 0:
            with torch.no_grad():
                vbar_now = float(mean_offdiag(gram_student(Y)))
                lam = max(0.0, lam + RHO * (vbar_now - c))
        lam_traj.append(lam)
    return Y.detach().cpu().numpy(), lam_traj


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Test C  -  primal-dual PUSH/PULL,  reduced MNIST n=500, "
          f"Student-t, gamma={SOFTENING}\n")

    ds = load_mnist(n_per_digit=N_PER_DIGIT, random_state=SEED)
    n, labels = ds.n, ds.labels
    w = default_weights(n, DEV)
    X_t = to_tensor(ds.X, DEV)
    K_X = compute_gaussian_affinity_kernel_torch(
        X_t, param={"perplexity": PERPLEXITY, "gamma": SOFTENING}, weights=w, device=DEV
    )
    K_X = normalize_kernel(K_X)               # unit Frobenius => alignment O(1)
    init = pca_init(ds.X)

    # (a) pure alignment, lambda = 0  (PULL only -> no partition function)
    Y0, _ = optimize(K_X, w, init, lam0=0.0)
    vbar_pull = float(mean_offdiag(gram_student(to_tensor(Y0, DEV))))
    c = TARGET_FRAC * vbar_pull
    print(f"PULL-only mean affinity vbar = {vbar_pull:.4f}   "
          f"target c = {c:.4f} ({TARGET_FRAC:.0%} of PULL-only)\n")

    # (b) fixed-lambda sweep (the PUSH-PULL tradeoff)
    sweep = {}
    for lam in (0.5, 1.0, 2.0, 4.0):
        Yl, _ = optimize(K_X, w, init, lam0=lam)
        sweep[lam] = Yl

    # (c) full primal-dual (dual ascent to volume target c)
    Ypd, lam_traj = optimize(K_X, w, init, lam0=0.0, c=c, adapt_dual=True)

    # (d) reference: t-SNE sklearn
    print("  (calcul t-SNE sklearn...)", flush=True)
    Yref = TSNE(n_components=D, perplexity=PERPLEXITY,
                random_state=SEED).fit_transform(ds.X)

    # --- report ---
    print(f"  {'configuration':<28} {'RV':>7} {'ARI':>7} {'spread':>8} {'lambda':>8}")
    print("  " + "-" * 62)
    rv0, ari0, sp0 = metrics(Y0, K_X, w, labels)
    print(f"  {'lambda=0 (PULL only)':<28} {rv0:>7.4f} {ari0:>7.4f} "
          f"{sp0:>8.3f} {0.0:>8.3f}")
    for lam, Yl in sweep.items():
        rv, ari, sp = metrics(Yl, K_X, w, labels)
        print(f"  {'fixed lambda=' + f'{lam:.1f}':<28} {rv:>7.4f} {ari:>7.4f} "
              f"{sp:>8.3f} {lam:>8.3f}")
    rvpd, aripd, sppd = metrics(Ypd, K_X, w, labels)
    print(f"  {'primal-dual (auto lambda)':<28} {rvpd:>7.4f} {aripd:>7.4f} "
          f"{sppd:>8.3f} {lam_traj[-1]:>8.3f}")
    rvr, arir, spr = metrics(Yref, K_X, w, labels)
    print(f"  {'reference t-SNE (sklearn)':<28} {rvr:>7.4f} {arir:>7.4f} "
          f"{spr:>8.3f} {'--':>8}")
    print(f"\n  lambda trajectory (primal-dual): "
          f"{lam_traj[0]:.2f} -> {max(lam_traj):.2f} -> {lam_traj[-1]:.2f}")

    # --- figures: lambda=0 vs primal-dual vs reference ---
    panels = [("lambda=0 (PULL only)", Y0), (f"primal-dual c={c:.3f}", Ypd),
              ("t-SNE sklearn (ref)", Yref)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, Y) in zip(axes, panels):
        ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Test C: PUSH/PULL primal-dual (no KL, no partition function)")
    fig.tight_layout()
    out = FIG_DIR / "test_C_primal_dual.png"
    fig.savefig(out, dpi=130)
    print(f"\n  scatter saved -> {out}")


if __name__ == "__main__":
    main()
