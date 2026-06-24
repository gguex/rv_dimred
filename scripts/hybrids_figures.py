"""
hybrids_figures.py  —  §5.3.3  figures from saved coordinates
=============================================================

Rebuilds, from results/coordinates/hybrids/*.npy (+ run_meta.csv for the RV
annotations):
  * Global-Local × Tail: the α×ν embedding grid, and quality-vs-α curves
    (trustworthiness, ARI for labelled datasets),
  * Unsupervised-Supervised: the α-sweep embedding row and its quality-vs-α curve
    (labelled datasets only).

Trustworthiness / ARI for the quality curves are recomputed from the coordinates
so the script is self-contained.
"""

from __future__ import annotations

import csv
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import indices as ix
from src.benchmark_common import (
    HYBRID_ALPHAS,
    HYBRID_NUS,
    SEED,
    TRUST_K,
    coord_path,
    fig_path,
    meta_path,
)
from src.datasets import Dataset, load_all

SECTION = "5.3.3"
FAMILY = "hybrids"


def load_rv_meta() -> dict[tuple[str, str], float]:
    meta: dict[tuple[str, str], float] = {}
    with meta_path(FAMILY).open() as f:
        for r in csv.DictReader(f):
            meta[(r["dataset"], r["key"])] = float(r["rv_final"])
    return meta


def _scatter(ax: Any, ds: Dataset, Y: np.ndarray, title: str) -> None:
    ax.scatter(Y[:, 0], Y[:, 1], c=ds.color, cmap=ds.cmap, s=6, alpha=0.8, linewidths=0)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _nu_label(nk: str) -> str:
    return "ν=∞" if nk == "inf" else f"ν={nk}"


def global_local(ds: Dataset, rv: dict[tuple[str, str], float]) -> None:
    n_a, n_v = len(HYBRID_ALPHAS), len(HYBRID_NUS)
    fig, axes = plt.subplots(n_a, n_v, figsize=(n_v * 3.2, n_a * 3.2))
    trust = {nk: [] for nk, _, _ in HYBRID_NUS}
    ari = {nk: [] for nk, _, _ in HYBRID_NUS}
    for i, alpha in enumerate(HYBRID_ALPHAS):
        for j, (nk, _, _) in enumerate(HYBRID_NUS):
            key = f"globallocal_a{alpha}_nu{nk}"
            Y = np.load(coord_path(FAMILY, ds.name, key, "embedding"))
            trust[nk].append(ix.trustworthiness(ds.X, Y, k=TRUST_K))
            if ds.has_labels:
                ari[nk].append(ix.ari(Y, ds.labels))
            _scatter(
                axes[i, j],
                ds,
                Y,
                f"α={alpha}  {_nu_label(nk)}\nRV={rv[(ds.name, key)]:.3f}",
            )
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

    # quality vs sweep
    fig, axes = plt.subplots(
        1,
        2 if ds.has_labels else 1,
        figsize=(11 if ds.has_labels else 6, 4.5),
        squeeze=False,
    )
    for nk, _, _ in HYBRID_NUS:
        axes[0, 0].plot(HYBRID_ALPHAS, trust[nk], marker="o", label=_nu_label(nk))
    axes[0, 0].set_xlabel("α (linear weight)")
    axes[0, 0].set_ylabel("Trustworthiness")
    axes[0, 0].set_title("Trustworthiness vs α")
    axes[0, 0].legend(fontsize=8)
    if ds.has_labels:
        for nk, _, _ in HYBRID_NUS:
            axes[0, 1].plot(HYBRID_ALPHAS, ari[nk], marker="o", label=_nu_label(nk))
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


def unsup_sup(ds: Dataset) -> None:
    fig, axes = plt.subplots(
        1, len(HYBRID_ALPHAS), figsize=(len(HYBRID_ALPHAS) * 3.2, 3.4)
    )
    trust, ari = [], []
    for i, alpha in enumerate(HYBRID_ALPHAS):
        Y = np.load(coord_path(FAMILY, ds.name, f"unsupsup_a{alpha}", "embedding"))
        t = ix.trustworthiness(ds.X, Y, k=TRUST_K)
        a = ix.ari(Y, ds.labels)
        trust.append(t)
        ari.append(a)
        _scatter(axes[i], ds, Y, f"α={alpha}\nTrust={t:.3f} ARI={a:.3f}")
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
    ax.plot(HYBRID_ALPHAS, trust, marker="o", label="Trustworthiness")
    ax.plot(HYBRID_ALPHAS, ari, marker="s", label="ARI")
    ax.set_xlabel("α (unsupervised weight)")
    ax.set_title(f"{ds.name} — Unsup-Sup quality vs α")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(
        fig_path(SECTION, ds.name, "unsupsup", "quality"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    datasets = load_all(random_state=SEED)
    rv = load_rv_meta()
    for ds in datasets.values():
        global_local(ds, rv)
        if ds.has_labels:
            unsup_sup(ds)
        print(f"  {ds.name}: hybrid grids + quality curves written")


if __name__ == "__main__":
    main()
