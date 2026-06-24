"""
test_diag_sweep.py  —  ISOLATED experiment (safe to delete)
============================================================

t-SNE and UMAP framework (γ=0.5 softening) on the 3 benchmark datasets.

For t-SNE: sweep perplexity.  For UMAP: sweep n_neighbors (k).
Each method × hyperparameter is run with two output-kernel variants:
  "full" = current production (G_ii = 1 before double-centering)
  "zero" = G_ii = 0  before double-centering

IMPORTANT: for every hp value, the library reference is also recomputed with
that SAME hp value so that Proc and kNN measure framework↔reference agreement
at matched hyperparameters.

Columns (framework rows):
  Proc  = Procrustes disparity vs reference at same hp (lower = closer)
  kNN   = kNN overlap vs reference at same hp (higher = closer)
  Trust = trustworthiness vs X
  ARI   = adjusted Rand index (skipped for swissroll which has no labels)
  w/b   = within/between scatter (lower = tighter classes)
  RV    = final RV of the framework embedding
Reference rows show Trust / ARI / w/b of the library output at the same hp.
Best framework row per (dataset, method) is marked with *.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import umap as umap_lib
from sklearn.manifold import TSNE

from src import indices as ix
from src.benchmark_common import (
    K_NEIGHBORS,
    PERPLEXITY,
    SEED,
    SOFTENING,
    TRUST_K,
    Q,
    get_device,
    pca_init,
    pca_init_sklearn,
    rv_embed,
    to_tensor,
)
from src.datasets import load_all
from src.rv_kernels import (
    compute_fuzzy_topological_kernel_torch,
    compute_gaussian_affinity_kernel_torch,
    default_weights,
    double_center,
    pairwise_sq_dists,
)

PERPLEXITIES = [5, 15, 30, 50, 100]
K_VALUES = [5, 15, 30, 50, 100]
UMAP_A, UMAP_B = 1.929, 0.7915  # min_dist=0.1 defaults


# ── output-kernel variants ────────────────────────────────────────────────────

def student_t_full(
    coords: torch.Tensor,
    param: float | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    from src.rv_kernels import compute_student_t_kernel_torch  # noqa: PLC0415
    return compute_student_t_kernel_torch(
        coords, param=param, weights=weights, device=device
    )


def student_t_zero(
    coords: torch.Tensor,
    param: float | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    nu = 1.0 if param is None else float(param)
    n = coords.shape[0]
    if weights is None:
        weights = default_weights(n, device)
    G = (1.0 + pairwise_sq_dists(coords) / nu) ** (-(nu + 1.0) / 2.0)
    mask = 1.0 - torch.eye(n, device=coords.device, dtype=G.dtype)
    return double_center(G * mask, weights, device)


def umap_full(
    coords: torch.Tensor,
    param: dict | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    from src.rv_kernels import compute_umap_kernel_torch  # noqa: PLC0415
    return compute_umap_kernel_torch(
        coords, param=param, weights=weights, device=device
    )


def umap_zero(
    coords: torch.Tensor,
    param: dict | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    p = param or {}
    a = float(p.get("a", UMAP_A))
    b = float(p.get("b", UMAP_B))
    n = coords.shape[0]
    if weights is None:
        weights = default_weights(n, device)
    G = 1.0 / (1.0 + a * (pairwise_sq_dists(coords) + 1e-12) ** b)
    mask = 1.0 - torch.eye(n, device=coords.device, dtype=G.dtype)
    return double_center(G * mask, weights, device)


# ── library references (recomputed per hp) ────────────────────────────────────

def ref_tsne(X: np.ndarray, init: np.ndarray, perplexity: float) -> np.ndarray:
    return np.asarray(
        TSNE(n_components=Q, perplexity=perplexity,
             init=init, random_state=SEED).fit_transform(X),
        dtype=np.float32,
    )


def ref_umap(X: np.ndarray, init: np.ndarray, n_neighbors: int) -> np.ndarray:
    return np.asarray(
        umap_lib.UMAP(
            n_neighbors=n_neighbors, n_components=Q,
            init=init, random_state=SEED,
        ).fit_transform(X),
        dtype=np.float32,
    )


# ── metrics ───────────────────────────────────────────────────────────────────

def within_between(Y: np.ndarray, labels: np.ndarray) -> float:
    g = Y.mean(0)
    within, between = 0.0, 0.0
    for c in np.unique(labels):
        Yc = Y[labels == c]
        ctr = Yc.mean(0)
        within += ((Yc - ctr) ** 2).sum()
        between += len(Yc) * ((ctr - g) ** 2).sum()
    return float(within / between)


def fw_metrics(
    Y: np.ndarray,
    ref: np.ndarray,
    X: np.ndarray,
    labels: np.ndarray | None,
) -> tuple[float, float, float, float | None, float | None]:
    proc = ix.procrustes_disparity(Y, ref)
    knn = ix.knn_overlap(Y, ref, k=TRUST_K)
    trust = ix.trustworthiness(X, Y, k=TRUST_K)
    if labels is None:
        return proc, knn, trust, None, None
    return proc, knn, trust, ix.ari(Y, labels), within_between(Y, labels)


def ref_metrics(
    Y: np.ndarray,
    X: np.ndarray,
    labels: np.ndarray | None,
) -> tuple[float, float | None, float | None]:
    trust = ix.trustworthiness(X, Y, k=TRUST_K)
    if labels is None:
        return trust, None, None
    return trust, ix.ari(Y, labels), within_between(Y, labels)


# ── formatting ────────────────────────────────────────────────────────────────

def _f(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "  -  "


HDR = (
    f"  {'hp':>5}  {'variant':>7}  "
    f"{'Proc':>6}  {'kNN':>6}  {'Trust':>6}  {'ARI':>6}  {'w/b':>6}  {'RV':>6}"
)


def fmt_ref(hp: int, trust: float, ari: float | None, wb: float | None) -> str:
    return (
        f"  {hp:>5}  {'ref':>7}  "
        f"{'---':>6}  {'---':>6}  {trust:.3f}  {_f(ari)}  {_f(wb)}  {'---':>6}"
    )


def fmt_fw(
    star: str,
    hp: int,
    diag: str,
    proc: float,
    knn: float,
    trust: float,
    ari: float | None,
    wb: float | None,
    rv: float,
) -> str:
    return (
        f"{star} {hp:>5}  {diag:>7}  "
        f"{proc:.3f}  {knn:.3f}  {trust:.3f}  {_f(ari)}  {_f(wb)}  {rv:.3f}"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = get_device()
    print(f"device: {device}  (γ={SOFTENING})\n")
    datasets = load_all(random_state=SEED)

    for ds in datasets.values():
        X_t = to_tensor(ds.X, device)
        w = default_weights(ds.n, device)
        init_fw = pca_init(ds.X)
        init_ref = pca_init_sklearn(ds.X)

        methods: list = [
            (
                "t-SNE", PERPLEXITIES,
                lambda hp: compute_gaussian_affinity_kernel_torch(
                    X_t,
                    param={"perplexity": hp, "gamma": SOFTENING},
                    weights=w, device=device,
                ),
                [(student_t_full, "full", 1.0), (student_t_zero, "zero", 1.0)],
                lambda hp: ref_tsne(ds.X, init_ref, hp),
            ),
            (
                "UMAP", K_VALUES,
                lambda hp: compute_fuzzy_topological_kernel_torch(
                    X_t,
                    param={"k": hp, "gamma": SOFTENING},
                    weights=w, device=device,
                ),
                [
                    (umap_full, "full", {"a": UMAP_A, "b": UMAP_B}),
                    (umap_zero, "zero", {"a": UMAP_A, "b": UMAP_B}),
                ],
                lambda hp: ref_umap(ds.X, init_ref, hp),
            ),
        ]

        for name, hp_values, input_fn, out_variants, ref_fn in methods:
            print("=" * 72)
            print(f"  {ds.name}  ·  {name}  (n={ds.n})")
            print("=" * 72)
            print(HDR)
            print("  " + "-" * (len(HDR) - 2))

            fw_rows: list[tuple[int, str, float, float, float,
                                float | None, float | None, float]] = []

            for hp in hp_values:
                Y_ref = ref_fn(hp)
                t_r, a_r, wb_r = ref_metrics(Y_ref, ds.X, ds.labels)
                print(fmt_ref(hp, t_r, a_r, wb_r))

                K_X = input_fn(hp)
                for out_fn, diag_lbl, out_param in out_variants:
                    Y, rv = rv_embed(
                        K_X, init_fw, device, out_fn, out_param, weights=w
                    )
                    proc, knn, trust, ari, wb = fw_metrics(
                        Y, Y_ref, ds.X, ds.labels
                    )
                    fw_rows.append((hp, diag_lbl, proc, knn, trust, ari, wb, rv))
                    print(fmt_fw(" ", hp, diag_lbl, proc, knn, trust, ari, wb, rv))

            # mark best framework row
            has_labels = fw_rows[0][5] is not None
            if has_labels:
                best = max(
                    range(len(fw_rows)),
                    key=lambda i: (fw_rows[i][5], -fw_rows[i][6]),  # type: ignore[operator]
                )
            else:
                best = max(range(len(fw_rows)), key=lambda i: fw_rows[i][4])
            hp, dl, proc, knn, trust, ari, wb, rv = fw_rows[best]
            print(
                f"\n  * best fw: hp={hp}, {dl} diag → "
                f"Proc={proc:.3f}  kNN={knn:.3f}  "
                f"Trust={trust:.3f}  ARI={_f(ari)}  w/b={_f(wb)}"
            )
            print()

    print(
        f"Default production: t-SNE perplexity={PERPLEXITY}, "
        f"UMAP k={K_NEIGHBORS}, diag=full"
    )


if __name__ == "__main__":
    main()
