"""
test_attraction_power.py  —  TEST H (new_article_plan.md §4: Prop. 5 cor. / Prop. 7b)
=====================================================================================

The attraction / repulsion POWER asymmetry between the RV-cosine framework and t-SNE.

Both methods use the SAME Student-t output q_ij = (1 + ||y_i - y_j||^2)^{-1}; only the
OBJECTIVE differs (RV bilinear alignment vs KL). We check, by autograd against the
project's real kernel code, the three gradient identities the article relies on:

  (A) RV attraction  (gradient of the RV numerator <K_X, K_Y>, K_Y the real
      compute_student_t_kernel_torch output):
          d/dy_k <K_X,K_Y> = 4 sum_j (K~_X)_kj  q_kj^2  (y_j - y_k)     [POWER 2]
      with K~_X = Q^T K_X Q.  -> matches q^2, does NOT match q^1.

  (B) t-SNE gradient (textbook, van der Maaten & Hinton 2008, eq. 8):
          dC/dy_k          = 4 sum_j (p - q)_kj q_kj    (y_k - y_j)     [attraction POWER 1]
      -> the t-SNE attractive weight is p_kj q_kj^1, one power of q LESS than the RV.

  (C) Volume repulsion (the only PUSH available, Prop. 7c):
          d/dy_k (-log Z)  = (4/Z) sum_j q_kj^2 (y_k - y_j)             [POWER 2]
      -> identical to the t-SNE repulsive force.

Conclusion: RV attraction and the volume PUSH are BOTH q^2 (same chain rule through the
bounded readout kappa' = -q^2); t-SNE is asymmetric (q^1 attraction, q^2 repulsion). The
framework is the RV-cosine cousin of t-SNE, not t-SNE -- the gradient-level mechanism of
"approximates t-SNE without reproducing it" (cf. experiments_notes.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (for `src`)

import torch

from src.rv_kernels import (
    centering_operator,
    compute_linear_kernel_torch,
    compute_student_t_kernel_torch,
    default_weights,
)

N, D = 14, 2
SEED = 0
TOL = 1e-9        # float64 match threshold
torch.set_default_dtype(torch.float64)


def dist2(Y: torch.Tensor) -> torch.Tensor:
    g = (Y * Y).sum(1, keepdim=True)
    return (g + g.T - 2.0 * Y @ Y.T).clamp_min(0.0)


def check(name: str, g_auto: torch.Tensor, g_closed: torch.Tensor) -> bool:
    err = (g_auto - g_closed).abs().max().item()
    ok = err < TOL
    print(f"  {'OK ' if ok else '<-- FAIL'}  {name:<46} max|autograd - closed| = {err:.2e}")
    return ok


def main() -> None:
    torch.manual_seed(SEED)
    w = default_weights(N, "cpu")
    Q = centering_operator(w, "cpu")
    Y0 = torch.randn(N, D)

    # A real centered input kernel K_X in K_n (any symmetric matrix would do for the
    # identity; we use the framework's own linear kernel to validate end to end).
    X = torch.randn(N, 5)
    K_X = compute_linear_kernel_torch(X, weights=w).detach()
    Ktil = Q.T @ K_X @ Q                                   # <K_X,K_Y> = <Ktil, G_Y>

    print(f"Test H  -  attraction / repulsion power asymmetry  (n={N}, d={D})\n")
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
    all_ok &= check("(A) RV attraction == q^2 closed form", gA, cf_q2)
    mism = (gA - cf_q1).abs().max().item()
    big = mism > 1e-3
    all_ok &= big
    print(f"  {'OK ' if big else '<-- FAIL'}  {'(A) RV attraction != q^1 (t-SNE) form':<46} "
          f"max|autograd - q^1| = {mism:.2e}  (must be LARGE)")

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
    all_ok &= check("(B) t-SNE grad == (p-q)*q^1 closed form", gC, cf_C)

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
    all_ok &= check("(C) -log Z repulsion == q^2 closed form", gZ, cf_Z)

    print(
        "\n  => RV attraction q^2 and volume PUSH q^2 (both via kappa' = -q^2);"
        "\n     t-SNE asymmetric: attraction q^1, repulsion q^2."
        "\n     Same Student-t output, different objective (RV vs KL)."
    )
    print(f"\n  ALL CHECKS PASS: {all_ok}")


if __name__ == "__main__":
    main()
