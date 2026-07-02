"""
test_directional_solver.py  —  ISOLATED experiment (safe to delete)
===================================================================

Prototype of the DIRECTIONAL SOLVER (new_article_plan.md §4): solve the RV
embedding problem entirely in kernel/distance space, by alternating projection
between the two constraints whose intersection is the achievable manifold
S_d^kappa, instead of running gradient ascent on Y.

Output readout: Student-t,  g = (1 + d^2)^{-1}  =>  d^2 = 1/g - 1.

One iteration (alternating projection), starting from the current embedding Y:
  (B) ALIGNMENT: take the current raw Student-t Gram G = kappa(D^2(Y)); replace its
      *centred* component by the K_X direction (same Frobenius norm), keeping the
      uncentred marginal part the readout needs:
          G_B = (G - dc(G)) + ||dc(G)|| * K_X/||K_X||
  (A) GEOMETRY + READOUT: invert the readout to target distances d^2 = 1/G_B - 1
      (clamp the out-of-range entries), then project onto rank-d EDM by classical
      MDS -> new Y.  Recompute the true K_Y from Y for the RV trajectory.

We log the RV trajectory and the two distinct approximation sources flagged in the
plan: (i) number of clamped (out-of-range) entries, (ii) rank-truncation residual.
References: gradient ascent (rv_embed) and the linear-output ceiling RV_max(d)
(Prop. 3 of the plan; an upper bound the curved manifold need not attain).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.benchmark_common import (
    PERPLEXITY,
    SEED,
    SOFTENING,
    pca_init,
    rv_embed,
    to_tensor,
)
from src.datasets import load_mnist
from src.rv_kernels import (
    compute_gaussian_affinity_kernel_torch,
    default_weights,
    double_center,
    rv_coefficient,
)

EPS = 1e-9
N_PER_DIGIT = 50          # reduced MNIST: 500 points total
N_ITER_DIR = 80           # directional-solver iterations
D = 2


# ── numpy helpers ─────────────────────────────────────────────────────────────

def sqdist(Y: np.ndarray) -> np.ndarray:
    sq = (Y**2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * Y @ Y.T
    return np.maximum(d2, 0.0)


def dcenter(G: np.ndarray) -> np.ndarray:
    """Double-centering H G H (uniform weights; global scale irrelevant to RV)."""
    m = G.mean(0, keepdims=True)
    return G - m - m.T + G.mean()


def student_gram(Y: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + sqdist(Y))


def cmds_rank(D2: np.ndarray, d: int = D) -> tuple[np.ndarray, float]:
    """Classical MDS: embed a (rank-d) EDM into R^d; return Y and the rank residual."""
    B = -0.5 * dcenter(D2)
    B = 0.5 * (B + B.T)
    w, V = np.linalg.eigh(B)          # ascending
    w = w[::-1]
    V = V[:, ::-1]
    pos = np.clip(w[:d], 0.0, None)
    Y = V[:, :d] * np.sqrt(pos + EPS)
    total = float(np.clip(w, 0.0, None).sum()) + EPS
    dropped = float(np.clip(w[d:], 0.0, None).sum())
    return Y, dropped / total


# ── RV evaluation (torch, consistent with the rest of the codebase) ───────────

def rv_of(Y: np.ndarray, K_X: torch.Tensor, w: torch.Tensor) -> float:
    G = 1.0 / (1.0 + torch.cdist(to_tensor(Y, "cpu"), to_tensor(Y, "cpu")) ** 2)
    K_Y = double_center(G, w, "cpu")
    return float(rv_coefficient(K_X, K_Y))


def rv_ceiling(K_X: torch.Tensor, d: int = D) -> float:
    """Prop. 3 linear-output ceiling: sqrt( sum_top-d (lambda^+)^2 / sum lambda^2 )."""
    lam = torch.linalg.eigvalsh(K_X).cpu().numpy()
    lam_sorted = np.sort(lam)[::-1]
    pos = np.clip(lam_sorted[:d], 0.0, None)
    return float(np.sqrt((pos**2).sum() / (lam**2).sum()))


# ── directional solver ────────────────────────────────────────────────────────

def directional_solver(
    K_X_np: np.ndarray, init: np.ndarray, K_X: torch.Tensor, w: torch.Tensor,
    n_iter: int = N_ITER_DIR, eta: float = 1.0,
) -> tuple[np.ndarray, list[float], list[int], list[float]]:
    """Alternating projection with damping eta: the centred target is moved only a
    fraction eta toward the K_X direction (eta=1 = full swap)."""
    Kx_hat = K_X_np / (np.linalg.norm(K_X_np) + EPS)
    Y = init.copy()
    rv_traj, clamps, resids = [rv_of(Y, K_X, w)], [], []
    for _ in range(n_iter):
        G = student_gram(Y)                       # raw Student-t Gram, diag = 1
        Kc = dcenter(G)                           # centred component
        scale = np.linalg.norm(Kc)
        # (B) alignment: nudge the centred part toward the K_X direction
        T_c = (1.0 - eta) * Kc + eta * scale * Kx_hat
        G_B = (G - Kc) + T_c
        G_B = np.clip(G_B, EPS, 1.0)              # valid Student-t range
        # (A) readout inversion + rank-d EDM projection
        D2 = 1.0 / G_B - 1.0
        n_clamped = int((D2 < 0).sum())
        D2 = np.clip(D2, 0.0, None)
        np.fill_diagonal(D2, 0.0)
        Y, resid = cmds_rank(D2, D)
        rv_traj.append(rv_of(Y, K_X, w))
        clamps.append(n_clamped)
        resids.append(resid)
    return Y, rv_traj, clamps, resids


def linear_oneshot(K_X: torch.Tensor, w: torch.Tensor) -> float:
    """Sanity check (linear output): classical MDS of K_X = projection onto the
    rank-d PSD cone. RV should equal the Prop. 3 ceiling, in ONE step."""
    Kx = K_X.cpu().numpy()
    lam, V = np.linalg.eigh(0.5 * (Kx + Kx.T))
    lam = lam[::-1]
    V = V[:, ::-1]
    pos = np.clip(lam[:D], 0.0, None)
    Y = V[:, :D] * np.sqrt(pos + EPS)
    G = to_tensor(Y @ Y.T, "cpu")                 # linear output Gram
    return float(rv_coefficient(K_X, double_center(G, w, "cpu")))


def main() -> None:
    print(f"reduced MNIST: {N_PER_DIGIT}/digit, Student-t output, gamma={SOFTENING}\n")
    ds = load_mnist(n_per_digit=N_PER_DIGIT, random_state=SEED)
    n = ds.n
    w = default_weights(n, "cpu")
    X_t = to_tensor(ds.X, "cpu")
    K_X = compute_gaussian_affinity_kernel_torch(
        X_t, param={"perplexity": PERPLEXITY, "gamma": SOFTENING},
        weights=w, device="cpu",
    )
    K_X_np = K_X.cpu().numpy()
    init = pca_init(ds.X)

    ceiling = rv_ceiling(K_X, D)
    print(f"n = {n}")
    print(
        f"linear-output ceiling RV_max(2) = {ceiling:.4f}  "
        f"(Prop. 3; upper bound for LINEAR output only)\n"
    )

    # --- references ---
    rv_lin = linear_oneshot(K_X, w)
    print(f"linear one-shot (cMDS of K_X)  RV = {rv_lin:.4f}   "
          f"(should equal ceiling -> validates machinery)")
    _, rv_grad = rv_embed(K_X, init, "cpu", "student_t", 1.0, weights=w)
    print(f"gradient ascent (Student-t)    RV = {rv_grad:.4f}   "
          f"(target for the directional solver)\n")

    # --- directional solver: damping sweep ---
    print("directional solver (Student-t), damping sweep:")
    print(f"  {'eta':>5} {'startRV':>8} {'finalRV':>8} {'bestRV':>8} "
          f"{'meanRankResid':>14}")
    print("  " + "-" * 48)
    for eta in (1.0, 0.5, 0.2, 0.05):
        _, traj, clamps, resids = directional_solver(
            K_X_np, init, K_X, w, eta=eta
        )
        print(
            f"  {eta:>5.2f} {traj[0]:>8.4f} {traj[-1]:>8.4f} {max(traj):>8.4f} "
            f"{np.mean(resids):>14.4f}"
        )
    print(f"\n(gradient RV={rv_grad:.4f}, linear ceiling={ceiling:.4f})")


if __name__ == "__main__":
    main()
