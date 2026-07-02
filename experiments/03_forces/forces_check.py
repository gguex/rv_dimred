"""
forces_check.py  —  art. §7.3(i)  attraction / repulsion power asymmetry
========================================================================

Validates Cor. 2 (RV gradient) and Prop. 6(b) (volume PUSH): the analytic force
laws the article states coincide with autograd of the project's *real* kernel code
to machine precision. Both the framework and t-SNE use the SAME Student-t output
q_ij = (1 + ||y_i - y_j||^2)^{-1}; only the OBJECTIVE differs (RV-cosine bilinear
alignment vs KL). We check three gradient identities:

  (A) RV attraction  (gradient of the RV numerator <K_X, K_Y>, K_Y the real
      compute_student_t_kernel_torch output):
          d/dy_k <K_X,K_Y> = 4 sum_j (K~_X)_kj  q_kj^2  (y_j - y_k)     [POWER 2]
      with K~_X = Q^T K_X Q.  -> matches q^2, does NOT match q^1.

  (B) t-SNE gradient (van der Maaten & Hinton 2008, eq. 8):
          dC/dy_k          = 4 sum_j (p - q)_kj q_kj    (y_k - y_j)     [POWER 1]
      -> the t-SNE attractive weight is p_kj q_kj^1, one power of q LESS than the RV.

  (C) Volume repulsion (the only PUSH available, Prop. 6c):
          d/dy_k (-log Z)  = (4/Z) sum_j q_kj^2 (y_k - y_j)             [POWER 2]
      -> identical to the t-SNE repulsive force.

Conclusion: RV attraction and the volume PUSH are BOTH q^2 (same chain rule through
the bounded readout kappa' = -q^2); t-SNE is asymmetric (q^1 attraction, q^2
repulsion). The framework is the RV-cosine cousin of t-SNE, not t-SNE -- the
gradient-level mechanism of "approximates t-SNE without reproducing it".

Run at n=2000 (the paper's scale): the identity is algebraic, so the machine-
precision match is scale-independent; running at n=2000 states it at the size used
by the rest of §7. Writes results/03_forces/indices/forces_check.csv.
"""

# ruff: noqa: E402, I001  (imports follow the sys.path bootstrap)
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (for `src`)

import torch

from src.benchmark_common import exp_indices_dir  # noqa: E402
from src.rv_kernels import (  # noqa: E402
    centering_operator,
    compute_linear_kernel_torch,
    compute_student_t_kernel_torch,
    default_weights,
)

EXP = "03_forces"
N, D = 2000, 2       # the paper's scale; identity is algebraic, so n is immaterial
SEED = 0
TOL = 1e-7           # float64 match threshold (abs. gradient magnitude grows with n)
CSV_FIELDS = ["check", "quantity", "max_abs_err", "tol", "pass"]
torch.set_default_dtype(torch.float64)


def dist2(Y: torch.Tensor) -> torch.Tensor:
    g = (Y * Y).sum(1, keepdim=True)
    return (g + g.T - 2.0 * Y @ Y.T).clamp_min(0.0)


def main() -> None:
    torch.manual_seed(SEED)
    w = default_weights(N, "cpu")
    Q = centering_operator(w, "cpu")
    Y0 = torch.randn(N, D)
    rows: list[dict[str, object]] = []

    def record(name: str, quantity: str, err: float, ok: bool, criterion: str) -> bool:
        tag = "OK " if ok else "<-- FAIL"
        print(f"  {tag}  {name:<38} max|auto - {quantity}| = {err:.2e}  ({criterion})")
        rows.append(
            {
                "check": name,
                "quantity": quantity,
                "max_abs_err": err,
                "tol": criterion,
                "pass": ok,
            }
        )
        return ok

    # A real centered input kernel K_X in K_n (any symmetric matrix would do for the
    # identity; we use the framework's own linear kernel to validate end to end).
    X = torch.randn(N, 5)
    K_X = compute_linear_kernel_torch(X, weights=w).detach()
    Ktil = Q.T @ K_X @ Q                                   # <K_X,K_Y> = <Ktil, G_Y>

    print(f"art. §7.3(i)  -  attraction / repulsion power asymmetry  (n={N}, q={D})\n")
    all_ok = True

    # ── (A) RV attraction: gradient of the real <K_X, K_Y> is q^2 (not q^1) ──────
    Ya = Y0.clone().requires_grad_(True)
    K_Y = compute_student_t_kernel_torch(Ya, param=1.0, weights=w)   # real code path
    obj = (K_X * K_Y).sum()                                          # = <K_X, K_Y>
    (gA,) = torch.autograd.grad(obj, Ya)
    with torch.no_grad():
        q = 1.0 / (1.0 + dist2(Y0))
        M2 = Ktil * q**2                                            # q^2 weighting
        cf_q2 = 4.0 * (M2 @ Y0 - M2.sum(1, keepdim=True) * Y0)
        M1 = Ktil * q                                              # q^1 (t-SNE-style)
        cf_q1 = 4.0 * (M1 @ Y0 - M1.sum(1, keepdim=True) * Y0)
    err_a2 = (gA - cf_q2).abs().max().item()
    err_a1 = (gA - cf_q1).abs().max().item()
    # the q^1 form must be distinguishable from autograd -- many orders of magnitude
    # above the q^2 match (absolute magnitudes shrink with n, so we compare ratios).
    distinct = err_a1 > 1e6 * max(err_a2, 1e-18)
    all_ok &= record("(A) RV attraction == q^2", "q^2", err_a2, err_a2 < TOL, "< tol")
    all_ok &= record("(A) RV attraction != q^1", "q^1", err_a1, distinct, ">> match")

    # ── (B) t-SNE gradient: attraction weight is p*q^1 (power 1) ─────────────────
    P = torch.rand(N, N)
    P = (P + P.T) / 2.0
    P.fill_diagonal_(0.0)
    P = (P / P.sum()).detach()
    Yb = Y0.clone().requires_grad_(True)
    Wb = 1.0 / (1.0 + dist2(Yb))
    Wb = Wb - torch.diag_embed(torch.diagonal(Wb))         # t-SNE excludes i=j
    Qb = Wb / Wb.sum()
    C = (P * torch.log((P + 1e-12) / (Qb + 1e-12))).sum()
    (gC,) = torch.autograd.grad(C, Yb)
    with torch.no_grad():
        Wt = 1.0 / (1.0 + dist2(Y0))
        Wt = Wt - torch.diag_embed(torch.diagonal(Wt))
        Qt = Wt / Wt.sum()
        S = (P - Qt) * Wt                                  # (p - q) * q^1
        cf_C = 4.0 * (S.sum(1, keepdim=True) * Y0 - S @ Y0)
    err_b = (gC - cf_C).abs().max().item()
    all_ok &= record("(B) t-SNE grad == (p-q)q", "(p-q)q", err_b, err_b < TOL, "< tol")

    # ── (C) repulsion: d/dy(-log Z) is q^2 == t-SNE repulsive force ──────────────
    Yc = Y0.clone().requires_grad_(True)
    Wc = 1.0 / (1.0 + dist2(Yc))
    Wc = Wc - torch.diag_embed(torch.diagonal(Wc))
    neg_log_Z = -torch.log(Wc.sum())
    (gZ,) = torch.autograd.grad(neg_log_Z, Yc)
    with torch.no_grad():
        Wr = 1.0 / (1.0 + dist2(Y0))
        Wr = Wr - torch.diag_embed(torch.diagonal(Wr))
        R = Wr**2
        cf_Z = (4.0 / Wr.sum()) * (R.sum(1, keepdim=True) * Y0 - R @ Y0)
    err_c = (gZ - cf_Z).abs().max().item()
    all_ok &= record("(C) -log Z repulsion == q^2", "q^2", err_c, err_c < TOL, "< tol")

    print(
        "\n  => RV attraction q^2 and volume PUSH q^2 (both via kappa' = -q^2);"
        "\n     t-SNE asymmetric: attraction q^1, repulsion q^2."
        "\n     Same Student-t output, different objective (RV vs KL)."
    )
    print(f"\n  ALL CHECKS PASS: {all_ok}")

    out = exp_indices_dir(EXP) / "forces_check.csv"
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
