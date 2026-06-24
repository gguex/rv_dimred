"""
spectral_figures.py  —  §5.3.1  figures from saved coordinates
==============================================================

Builds, from results/coordinates/spectral/*.npy:
  * per (dataset × method): three Procrustes-aligned scatters in separate files
    (reference, framework_linear, framework_projector) — for side-by-side LaTeX,
  * per dataset: a summary bar chart comparing the two output kernels
    (Procrustes — lower is better; kNN overlap — higher is better), to read off
    which output kernel recovers each reference best.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import indices as ix
from src.benchmark_common import (
    SEED,
    SPECTRAL_METHODS,
    TRUST_K,
    coord_path,
    fig_path,
)
from src.datasets import Dataset, load_all

SECTION = "5.3.1"
FAMILY = "spectral"
OUTPUT_KERNELS = ["linear", "projector"]


def scatter(
    ds: Dataset, method_key: str, variant: str, Y: np.ndarray, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        Y[:, 0], Y[:, 1], c=ds.color, cmap=ds.cmap, s=10, alpha=0.85, linewidths=0
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(
        fig_path(SECTION, ds.name, method_key, variant), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def comparison_figure(
    ds: Dataset,
    names: list[str],
    proc: dict[str, list[float]],
    knn: dict[str, list[float]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(names))
    width = 0.38
    for j, ok in enumerate(OUTPUT_KERNELS):
        off = (j - 0.5) * width
        axes[0].bar(x + off, proc[ok], width, label=ok)
        axes[1].bar(x + off, knn[ok], width, label=ok)
    axes[0].set_title("Procrustes (lower = better recovery)")
    axes[0].set_ylabel("Procrustes disparity")
    axes[1].set_title("kNN overlap (higher = better)")
    axes[1].set_ylabel("kNN overlap")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=9, title="output kernel")
    fig.suptitle(f"{ds.name} — output-kernel comparison (§{SECTION})", fontsize=12)
    fig.tight_layout()
    fig.savefig(
        fig_path(SECTION, ds.name, "summary", "comparison"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    datasets = load_all(random_state=SEED)

    for ds in datasets.values():
        proc: dict[str, list[float]] = {ok: [] for ok in OUTPUT_KERNELS}
        knn: dict[str, list[float]] = {ok: [] for ok in OUTPUT_KERNELS}
        names: list[str] = []

        for m in SPECTRAL_METHODS:
            names.append(m.name)
            ref = np.load(coord_path(FAMILY, ds.name, m.key, "reference"))
            ref_std = None
            for ok in OUTPUT_KERNELS:
                fw = np.load(coord_path(FAMILY, ds.name, m.key, f"framework_{ok}"))
                d = ix.procrustes_disparity(fw, ref)
                proc[ok].append(d)
                knn[ok].append(ix.knn_overlap(fw, ref, k=TRUST_K))
                ref_std, fw_aligned, _ = ix.align_procrustes(ref, fw)
                scatter(
                    ds,
                    m.key,
                    ok,
                    fw_aligned,
                    f"{ds.name} — {m.name} ({ok})\nProcrustes={d:.4f}",
                )
            # reference frame is independent of which framework it aligned to
            scatter(
                ds, m.key, "reference", ref_std, f"{ds.name} — {m.name} (reference)"
            )

        comparison_figure(ds, names, proc, knn)
        print(f"  {ds.name}: scatters + comparison written")


if __name__ == "__main__":
    main()
