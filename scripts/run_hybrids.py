"""
run_hybrids.py  —  §5.3.3  Hybrid methods (novelty / flexibility)
================================================================

Two interpolations with no reference implementation — they exist only because
the framework lets you add input kernels and swap the output tail:

  (1) Global-Local × Tail
        input  = α·K_linear + (1-α)·K_adaptiveGaussian
        output = Student-t(ν),  ν ∈ {∞ (Gaussian), 1, 0.5}
        α ∈ {0,.25,.5,.75,1}        (α=1,ν=∞ ≈ PCA ; α=0,ν=1 ≈ t-SNE)
      → full α×ν grid per dataset (the "map of DR space").

  (2) Unsupervised-Supervised
        input  = α·K_adaptiveGaussian + (1-α)·K_class
        output = Student-t(ν=1)
        α ∈ {0,.25,.5,.75,1}        (α=1 ≈ t-SNE ; α=0 ≈ class/LDA)
      → α-sweep on labelled datasets.

Plus quality-vs-sweep curves (trustworthiness, ARI) quantifying the trade-off.
Indices append to results/results_indices.csv.
"""

from __future__ import annotations

import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import indices as ix
from src.datasets import Dataset, load_all
from src.rv_kernels import (
    compute_class_kernel_torch,
    compute_gaussian_affinity_kernel_torch,
    compute_linear_kernel_torch,
    default_weights,
)
from src.benchmark_common import (
    PERPLEXITY,
    SEED,
    TRUST_K,
    IndexLog,
    fig_path,
    get_device,
    normalize_kernel,
    pca_init,
    rv_embed,
    to_tensor,
)

SECTION = "5.3.3"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
# ν=∞ rendered via the Gaussian (SNE) output kernel; finite ν via Student-t.
NUS: list[tuple[str, str, float | None]] = [
    ("inf", "gaussian", None),  # ν → ∞  (Gaussian / light tail)
    ("1.0", "student_t", 1.0),  # ν = 1  (classic t-SNE tail)
    ("0.5", "student_t", 0.5),  # ν = 0.5 (heavy tail)
]


def _scatter(ax: Any, ds: Dataset, Y: np.ndarray, title: str) -> None:
    ax.scatter(Y[:, 0], Y[:, 1], c=ds.color, cmap=ds.cmap, s=6, alpha=0.8, linewidths=0)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


# ══════════════════════════════════════════════════════════════════════════════
# (1) Global-Local × Tail
# ══════════════════════════════════════════════════════════════════════════════
def global_local_tail(ds: Dataset, device: str, log: IndexLog) -> None:
    X_t = to_tensor(ds.X, device)
    w = default_weights(ds.n, device)
    init = pca_init(ds.X)
    # unit-Frobenius normalised so α genuinely interpolates the two geometries
    K_lin = normalize_kernel(
        compute_linear_kernel_torch(X_t, weights=w, device=device)
    )
    K_gauss = normalize_kernel(
        compute_gaussian_affinity_kernel_torch(
            X_t, param={"perplexity": PERPLEXITY}, weights=w, device=device
        )
    )

    n_a, n_v = len(ALPHAS), len(NUS)
    fig, axes = plt.subplots(n_a, n_v, figsize=(n_v * 3.2, n_a * 3.2))
    # quality-vs-sweep accumulators: {nu_key: [values over alpha]}
    trust_curve: dict[str, list[float]] = {nk: [] for nk, _, _ in NUS}
    ari_curve: dict[str, list[float]] = {nk: [] for nk, _, _ in NUS}

    for i, alpha in enumerate(ALPHAS):
        K_in = alpha * K_lin + (1.0 - alpha) * K_gauss
        for j, (nk, out_kernel, nu) in enumerate(NUS):
            Y, rv = rv_embed(K_in, init, device, out_kernel, nu, weights=w)
            trust = ix.trustworthiness(ds.X, Y, k=TRUST_K)
            trust_curve[nk].append(trust)
            mname = f"globallocal_a{alpha}_nu{nk}"
            log.add(SECTION, ds.name, mname, "trustworthiness", trust, k=TRUST_K)
            log.add(SECTION, ds.name, mname, "rv_final", rv)
            if ds.has_labels:
                a = ix.ari(Y, ds.labels)
                ari_curve[nk].append(a)
                log.add(SECTION, ds.name, mname, "ari", a)
            nu_lbl = "ν=∞" if nk == "inf" else f"ν={nk}"
            _scatter(axes[i, j], ds, Y, f"α={alpha}  {nu_lbl}\nRV={rv:.3f}")
            if j == 0:
                axes[i, j].set_ylabel(f"α={alpha}", fontsize=9)

    fig.suptitle(
        f"{ds.name} — Global-Local (α·linear+(1-α)·adapt.Gauss) × Tail(ν)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(
        fig_path(SECTION, ds.name, "globallocal", "grid"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # quality-vs-sweep curves
    fig, axes = plt.subplots(
        1,
        2 if ds.has_labels else 1,
        figsize=(11 if ds.has_labels else 6, 4.5),
        squeeze=False,
    )
    for nk, _, _ in NUS:
        lbl = "ν=∞" if nk == "inf" else f"ν={nk}"
        axes[0, 0].plot(ALPHAS, trust_curve[nk], marker="o", label=lbl)
    axes[0, 0].set_xlabel("α (linear weight)")
    axes[0, 0].set_ylabel("Trustworthiness")
    axes[0, 0].set_title("Trustworthiness vs α")
    axes[0, 0].legend(fontsize=8)
    if ds.has_labels:
        for nk, _, _ in NUS:
            lbl = "ν=∞" if nk == "inf" else f"ν={nk}"
            axes[0, 1].plot(ALPHAS, ari_curve[nk], marker="o", label=lbl)
        axes[0, 1].set_xlabel("α (linear weight)")
        axes[0, 1].set_ylabel("ARI")
        axes[0, 1].set_title("ARI vs α")
        axes[0, 1].legend(fontsize=8)
    fig.suptitle(f"{ds.name} — Global-Local × Tail: quality vs sweep", fontsize=11)
    fig.tight_layout()
    fig.savefig(
        fig_path(SECTION, ds.name, "globallocal", "quality"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# (2) Unsupervised-Supervised  (labelled datasets only)
# ══════════════════════════════════════════════════════════════════════════════
def unsup_sup(ds: Dataset, device: str, log: IndexLog) -> None:
    X_t = to_tensor(ds.X, device)
    w = default_weights(ds.n, device)
    init = pca_init(ds.X)
    K_gauss = normalize_kernel(
        compute_gaussian_affinity_kernel_torch(
            X_t, param={"perplexity": PERPLEXITY}, weights=w, device=device
        )
    )
    K_class = normalize_kernel(
        compute_class_kernel_torch(
            X_t, param={"labels": ds.labels}, weights=w, device=device
        )
    )

    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(len(ALPHAS) * 3.2, 3.4))
    trust_curve, ari_curve = [], []
    for i, alpha in enumerate(ALPHAS):
        K_in = alpha * K_gauss + (1.0 - alpha) * K_class
        Y, rv = rv_embed(K_in, init, device, "student_t", 1.0, weights=w)
        trust = ix.trustworthiness(ds.X, Y, k=TRUST_K)
        a = ix.ari(Y, ds.labels)
        trust_curve.append(trust)
        ari_curve.append(a)
        mname = f"unsupsup_a{alpha}"
        log.add(SECTION, ds.name, mname, "trustworthiness", trust, k=TRUST_K)
        log.add(SECTION, ds.name, mname, "ari", a)
        log.add(SECTION, ds.name, mname, "rv_final", rv)
        _scatter(axes[i], ds, Y, f"α={alpha}\nTrust={trust:.3f} ARI={a:.3f}")
    fig.suptitle(
        f"{ds.name} — Unsup-Sup (α·adapt.Gauss+(1-α)·class) × Student-t(ν=1)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(
        fig_path(SECTION, ds.name, "unsupsup", "grid"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ALPHAS, trust_curve, marker="o", label="Trustworthiness")
    ax.plot(ALPHAS, ari_curve, marker="s", label="ARI")
    ax.set_xlabel("α (unsupervised weight)")
    ax.set_title(f"{ds.name} — Unsup-Sup quality vs α")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(
        fig_path(SECTION, ds.name, "unsupsup", "quality"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")
    log = IndexLog()
    datasets = load_all(random_state=SEED)

    for ds in datasets.values():
        print("=" * 64)
        print(f"§{SECTION}  dataset = {ds.name}  (n={ds.n})")
        print("=" * 64)
        t0 = time.time()
        global_local_tail(ds, device, log)
        print(f"  Global-Local × Tail grid done ({time.time() - t0:.1f}s)")
        if ds.has_labels:
            t0 = time.time()
            unsup_sup(ds, device, log)
            print(f"  Unsup-Sup α-sweep done ({time.time() - t0:.1f}s)")

    path = log.write(append=True)
    print(f"\nAppended indices → {path}")


if __name__ == "__main__":
    main()
