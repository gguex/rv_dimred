"""
manifold_dim_run.py  —  art. §7.2  intrinsic dimension of the output manifold
=============================================================================

Validates Prop. 2 (promise of art. §4.2): the non-linear output manifold
    S_q^kappa = { Q kappa(D^2(Y)) Q^T : Y in R^{n x q} }
has intrinsic dimension
    dim = n*q - C(q+1, 2),
the C(q+1,2) invariances being global translation of Y (q dims) and rotation
O(q) of Y (q(q-1)/2 dims), which leave D^2(Y) -- hence K_Y -- fixed. For q=2
this is the headline  dim = 2n - 3.

We estimate dim S_q^kappa as the RANK of the Jacobian of
    Y  |->  K_Y = Q kappa(D^2(Y)) Q^T          (Student-t: kappa = 1/(1+D^2))
at generic (random) base points Y0, via torch autograd. We sweep
q in {1,2,3} AND n in {30,50,80,120} to show the formula holds across sample
sizes (not a coincidence at one n) and print the singular-value gap around the
predicted rank as evidence of a well-defined rank (a clean spectral cliff).

The rank is a property of the map, not of any dataset: base points are random,
so nothing here depends on MNIST / single-cell / Swiss-roll. Writes the sweep to
results/02_manifold_dim/indices/manifold_dim.csv.
"""

# ruff: noqa: E402, I001  (imports follow the sys.path bootstrap)
from __future__ import annotations

import csv
import sys
from collections.abc import Callable
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (for `src`)

import numpy as np
import torch

from src.benchmark_common import exp_indices_dir  # noqa: E402
from src.rv_kernels import centering_operator, default_weights  # noqa: E402

EXP = "02_manifold_dim"
NS = (30, 50, 80, 120)     # sample sizes (rank must not be a coincidence at one n)
DS = (1, 2, 3)             # embedding dimensions to test
N_SAMPLES = 5              # random base points Y0 per (n, q) (test genericity)
SEED = 0
CSV_FIELDS = [
    "n",
    "q",
    "n_in",
    "invariance",
    "predicted",
    "rank",
    "consistent",
    "min_gap",
]
torch.set_default_dtype(torch.float64)   # accurate singular values / rank


def k_map_factory(
    n: int, q: int, Q: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return f: R^{n*q} -> R^{n*n}, the flattened centered Student-t output kernel."""

    def f(y_flat: torch.Tensor) -> torch.Tensor:
        Y = y_flat.reshape(n, q)
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
    print("art. §7.2  -  intrinsic dimension of S_q^kappa  (Student-t)")
    print(f"  rank on {N_SAMPLES} random base points Y0 per (n, q) (genericity)")
    print(f"  sweeping n in {NS} to rule out a coincidence at any single n\n")

    rows: list[dict[str, object]] = []
    all_ok = True
    for N in NS:
        w = default_weights(N, "cpu")
        Q = centering_operator(w, "cpu")

        print(f"  n = {N}   (ambient dim C(n,2) = {comb(N, 2)})")
        print(f"    {'q':>2} {'n*q':>5} {'C(q+1,2)':>9} {'predicted':>10} "
              f"{'ranks':>14} {'min gap(x)':>11}")
        print("    " + "-" * 65)

        for q in DS:
            f = k_map_factory(N, q, Q)
            n_in = N * q
            invariance = comb(q + 1, 2)
            predicted = N * q - invariance

            ranks: list[int] = []
            min_gap = float("inf")
            for s in range(N_SAMPLES):
                torch.manual_seed(SEED + s)
                Y0 = torch.randn(n_in)
                J = torch.autograd.functional.jacobian(f, Y0)      # (n*n, n*q)
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
            print(f"    {q:>2} {n_in:>5} {invariance:>9} {predicted:>10} "
                  f"{rank_str:>14} {min_gap:>11.2e}  {flag}")
            rows.append(
                {
                    "n": N,
                    "q": q,
                    "n_in": n_in,
                    "invariance": invariance,
                    "predicted": predicted,
                    "rank": rank_str,
                    "consistent": ok,
                    "min_gap": min_gap,
                }
            )
        print(f"    headline (q=2): dim = 2n - 3 = {2 * N - 3}\n")

    print("  predicted = n*q - C(q+1,2)   (translation q + rotation q(q-1)/2)")
    print("  same rank across all Y0 and all n + large gap => well-defined dimension")
    print(f"\n  ALL GENERIC: {all_ok}")

    out = exp_indices_dir(EXP) / "manifold_dim.csv"
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
