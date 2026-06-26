"""
test_manifold_dimension.py  —  TEST D (new_article_plan.md §6)
==============================================================

Intrinsic dimension of the non-linear output manifold
    S_d^kappa = { Q kappa(D^2(Y)) Q^T : Y in R^{n x d} }   (Prop. 4).

We estimate dim S_d^kappa as the RANK of the Jacobian of the map
    Y  |->  K_Y = Q kappa(D^2(Y)) Q^T          (Student-t: kappa = 1/(1+D^2))
at a generic point Y (random), via torch autograd.

Predicted rank (continuous invariances quotiented out):
    dim = n*d - C(d+1, 2)
because D^2(Y) -- hence K_Y -- is invariant under
    - global translation of Y           (d dims)
    - rotation O(d) of Y                 (d(d-1)/2 dims)
    => total invariance d + d(d-1)/2 = d(d+1)/2 = C(d+1,2).

For d=2 this is the headline result  dim = 2n - 3.

We sweep d = 1, 2, 3 to confirm the GENERAL formula, and print the singular-value
gap around the predicted rank as the evidence (a clean cliff => well-defined rank).
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.rv_kernels import centering_operator, default_weights

N = 50                      # plan: N = 50 sample points
DS = (1, 2, 3)              # embedding dimensions to test
SEED = 0
torch.set_default_dtype(torch.float64)   # accurate singular values / rank


def k_map_factory(n: int, d: int, Q: torch.Tensor):
    """Return f: R^{n*d} -> R^{n*n}, the flattened centered Student-t output kernel."""

    def f(y_flat: torch.Tensor) -> torch.Tensor:
        Y = y_flat.reshape(n, d)
        sq = (Y * Y).sum(dim=1)
        d2 = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (Y @ Y.T)).clamp_min(0.0)
        G = 1.0 / (1.0 + d2)                 # Student-t readout
        K = Q @ G @ Q.T                      # centered kernel in K_n
        return K.reshape(-1)

    return f


def numerical_rank(sv: np.ndarray, n_in: int) -> tuple[int, float]:
    """Rank by the standard relative threshold tol = max(sv) * n_in * eps."""
    tol = sv.max() * n_in * np.finfo(sv.dtype).eps
    return int((sv > tol).sum()), float(tol)


def main() -> None:
    torch.manual_seed(SEED)
    w = default_weights(N, "cpu")
    Q = centering_operator(w, "cpu")

    print(f"Test D  -  intrinsic dimension of S_d^kappa  (Student-t), n = {N}\n")
    print(f"  {'d':>2} {'n*d':>5} {'C(d+1,2)':>9} {'predicted':>10} "
          f"{'measured':>9} {'sv[r-1]':>11} {'sv[r]':>11} {'gap(x)':>9}")
    print("  " + "-" * 72)

    for d in DS:
        Y0 = torch.randn(N * d)
        f = k_map_factory(N, d, Q)
        J = torch.autograd.functional.jacobian(f, Y0)      # (n*n, n*d)
        sv = torch.linalg.svdvals(J).cpu().numpy()
        n_in = N * d

        rank, _ = numerical_rank(sv, n_in)
        predicted = N * d - comb(d + 1, 2)

        sv_last = sv[rank - 1] if rank >= 1 else float("nan")     # last "live" sv
        sv_next = sv[rank] if rank < len(sv) else 0.0             # first "null" sv
        gap = (sv_last / sv_next) if sv_next > 0 else float("inf")

        flag = "OK" if rank == predicted else "<-- MISMATCH"
        print(f"  {d:>2} {n_in:>5} {comb(d + 1, 2):>9} {predicted:>10} "
              f"{rank:>9} {sv_last:>11.3e} {sv_next:>11.3e} {gap:>9.2e}  {flag}")

    print("\n  predicted = n*d - C(d+1,2)   (translation d + rotation d(d-1)/2)")
    print("  large gap(x) = clean spectral cliff => the rank is well defined")
    print(f"  headline (d=2): dim = 2n - 3 = {2 * N - 3}")


if __name__ == "__main__":
    main()
