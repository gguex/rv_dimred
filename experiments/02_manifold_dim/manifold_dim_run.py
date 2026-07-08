"""
manifold_dim_run.py  —  art. §6.3  intrinsic dimension of the output manifold
=============================================================================

Validates Prop. 2 (promise of art. §4.2) for the two canonical readouts of
art. §4.1: the non-linear output manifold
    S_q^kappa = { Q kappa(arg(Y)) Q^T : Y in R^{n x q} }
has intrinsic dimension
    (B) distance readout   arg(Y) = D^2(Y):   dim = n*q - C(q+1, 2),
    (A) dot-prod. readout  arg(Y) = Y Y^T:    dim = n*q - C(q, 2),
the invariances being the motions the readout cannot see: all rigid motions
(translation q + rotation q(q-1)/2) leave every distance fixed, while a
non-affine dot-product readout registers translations and only rotations
remain. For q=2 the distance case is the headline  dim = 2n - 3; the
dot-product sheet is thicker by exactly q. Each regular dot-product
configuration is also the certificate that Prop. 2(ii) calls for, settling
the generic alternative for the exponential profile.

We estimate dim S_q^kappa as the RANK of the Jacobian of
    Y  |->  K_Y = Q kappa(arg(Y)) Q^T
        (Student-t: kappa = 1/(1+t) on D^2;  exponential: kappa = exp(t/10)
         on Y Y^T, scaled to keep entries in a sane numerical range)
at generic (random) base points Y0, via torch autograd. We sweep
q in {1,2,3} AND n in {30,50,80,120} to show the formulas hold across sample
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
READOUTS = ("student_t", "dot_exp")   # (B) distance / (A) dot-product readout
N_SAMPLES = 5              # random base points Y0 per (n, q) (test genericity)
SEED = 0
CSV_FIELDS = [
    "readout",
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
    n: int, q: int, Q: torch.Tensor, readout: str
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return f: R^{n*q} -> R^{n*n}, the flattened centered output kernel."""

    def f(y_flat: torch.Tensor) -> torch.Tensor:
        Y = y_flat.reshape(n, q)
        if readout == "student_t":                # (B) distance readout
            sq = (Y * Y).sum(dim=1)
            d2 = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (Y @ Y.T)).clamp_min(0.0)
            G = 1.0 / (1.0 + d2)
        else:                                     # (A) dot-product readout
            G = torch.exp((Y @ Y.T) / 10.0)       # kappa' != 0, non-affine, analytic
        K = Q @ G @ Q.T                           # centered kernel in K_n
        return K.reshape(-1)

    return f


def invariance_dim(q: int, readout: str) -> int:
    """Motions the readout cannot see: rigid (B) vs rotations only (A)."""
    return comb(q + 1, 2) if readout == "student_t" else comb(q, 2)


def numerical_rank(sv: np.ndarray, n_in: int) -> tuple[int, float]:
    """Rank by the standard relative threshold tol = max(sv) * n_in * eps."""
    tol = sv.max() * n_in * np.finfo(sv.dtype).eps
    return int((sv > tol).sum()), float(tol)


def main() -> None:
    print("art. §6.3  -  intrinsic dimension of S_q^kappa  (two readouts)")
    print(f"  rank on {N_SAMPLES} random base points Y0 per (readout, n, q)")
    print(f"  sweeping n in {NS} to rule out a coincidence at any single n\n")

    rows: list[dict[str, object]] = []
    all_ok = True
    for readout in READOUTS:
        family = "(B) distance, Student-t" if readout == "student_t" \
            else "(A) dot-product, exponential"
        inv_label = "C(q+1,2)" if readout == "student_t" else "C(q,2)"
        print(f"  readout: {family}   predicted = n*q - {inv_label}")

        for N in NS:
            w = default_weights(N, "cpu")
            Q = centering_operator(w, "cpu")

            print(f"  n = {N}   (ambient dim C(n,2) = {comb(N, 2)})")
            print(f"    {'q':>2} {'n*q':>5} {inv_label:>9} {'predicted':>10} "
                  f"{'ranks':>14} {'min gap(x)':>11}")
            print("    " + "-" * 65)

            for q in DS:
                f = k_map_factory(N, q, Q, readout)
                n_in = N * q
                invariance = invariance_dim(q, readout)
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
                        "readout": readout,
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
            if readout == "student_t":
                print(f"    headline (q=2): dim = 2n - 3 = {2 * N - 3}\n")
            else:
                print(f"    thicker by q = translations (q=2: {2 * N - 1})\n")

    print("  (B) predicted = n*q - C(q+1,2)   (translation q + rotation q(q-1)/2)")
    print("  (A) predicted = n*q - C(q,2)     (rotations only; translations seen)")
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
