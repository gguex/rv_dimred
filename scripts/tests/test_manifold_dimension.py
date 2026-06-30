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

We sweep d = 1, 2, 3 AND n in {30, 50, 80} to confirm the GENERAL formula holds at
several sample sizes (not a coincidence at n=50), and print the singular-value gap
around the predicted rank as the evidence (a clean cliff => well-defined rank).
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (for `src`)

import numpy as np
import torch

from src.rv_kernels import centering_operator, default_weights

NS = (30, 50, 80)          # sample sizes (rank must not be a coincidence at n=50)
DS = (1, 2, 3)              # embedding dimensions to test
N_SAMPLES = 5               # random base points Y0 per (n, d) (test genericity)
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
    print("Test D  -  intrinsic dimension of S_d^kappa  (Student-t)")
    print(f"  rank checked on {N_SAMPLES} random base points Y0 per (n, d) (genericity)")
    print(f"  sweeping n in {NS} to rule out a coincidence at any single n\n")

    all_ok = True
    for N in NS:
        w = default_weights(N, "cpu")
        Q = centering_operator(w, "cpu")

        print(f"  n = {N}   (ambient dim C(n,2) = {comb(N, 2)})")
        print(f"    {'d':>2} {'n*d':>5} {'C(d+1,2)':>9} {'predicted':>10} "
              f"{'ranks':>14} {'min gap(x)':>11}")
        print("    " + "-" * 65)

        for d in DS:
            f = k_map_factory(N, d, Q)
            n_in = N * d
            predicted = N * d - comb(d + 1, 2)

            ranks: list[int] = []
            min_gap = float("inf")
            for s in range(N_SAMPLES):
                torch.manual_seed(SEED + s)
                Y0 = torch.randn(n_in)
                J = torch.autograd.functional.jacobian(f, Y0)      # (n*n, n*d)
                sv = torch.linalg.svdvals(J).cpu().numpy()

                rank, _ = numerical_rank(sv, n_in)
                ranks.append(rank)
                sv_last = sv[rank - 1] if rank >= 1 else float("nan")
                sv_next = sv[rank] if rank < len(sv) else 0.0
                gap = (sv_last / sv_next) if sv_next > 0 else float("inf")
                min_gap = min(min_gap, gap)

            consistent = len(set(ranks)) == 1
            rank_str = str(ranks[0]) if consistent else "/".join(map(str, ranks))
            ok = consistent and ranks[0] == predicted
            all_ok = all_ok and ok
            flag = "OK" if ok else "<-- MISMATCH"
            print(f"    {d:>2} {n_in:>5} {comb(d + 1, 2):>9} {predicted:>10} "
                  f"{rank_str:>14} {min_gap:>11.2e}  {flag}")
        print(f"    headline (d=2): dim = 2n - 3 = {2 * N - 3}\n")

    print("  predicted = n*d - C(d+1,2)   (translation d + rotation d(d-1)/2)")
    print("  same rank across all Y0 and all n + large gap => well-defined GENERIC dimension")
    print(f"\n  ALL GENERIC: {all_ok}")


if __name__ == "__main__":
    main()
