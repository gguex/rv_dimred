"""
run_spectral.py  —  §5.3.1  Spectral methods (exact recovery)
=============================================================

For every (spectral method × dataset):
  * reference embedding (named library implementation, output = linear),
  * framework embedding (input kernel maximised against a LINEAR output kernel),
  * identity indices  (Procrustes disparity, kNN overlap)  → expect ~0 / ~1,
  * quality indices   (trustworthiness, Q_local, ARI),
  * a Procrustes-aligned overlay figure.

All indices are appended to results/results_indices.csv (this script writes it
fresh; the other §5 scripts append). Figures follow the §0.5 naming convention.
"""

from __future__ import annotations

import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import indices as ix
from src.datasets import Dataset, load_all
from src.rv_kernels import default_weights
from src.section5_common import (
    SEED,
    SPECTRAL_METHODS,
    TRUST_K,
    IndexLog,
    fig_path,
    get_device,
    pca_init,
    rv_embed,
    to_tensor,
)

SECTION = "5.3.1"


def overlay_figure(
    ds: Dataset,
    method_name: str,
    key: str,
    Y_ref: np.ndarray,
    Y_fw: np.ndarray,
    disparity: float,
    rv: float,
) -> None:
    """Procrustes-aligned overlay: reference vs framework in a shared frame."""
    ref_std, fw_aligned, _ = ix.align_procrustes(Y_ref, Y_fw)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        ref_std[:, 0],
        ref_std[:, 1],
        c=ds.color,
        cmap=ds.cmap,
        s=24,
        alpha=0.55,
        marker="o",
        linewidths=0,
        label="reference",
    )
    ax.scatter(
        fw_aligned[:, 0],
        fw_aligned[:, 1],
        c=ds.color,
        cmap=ds.cmap,
        s=14,
        alpha=0.9,
        marker="x",
        linewidths=0.7,
        label="framework",
    )
    ax.set_title(
        f"{ds.name} — {method_name}\nProcrustes={disparity:.4f}  RV={rv:.4f}",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(
        fig_path(SECTION, ds.name, key, "overlay"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")
    log = IndexLog()
    datasets = load_all(random_state=SEED)

    for ds in datasets.values():
        print("=" * 64)
        print(f"§{SECTION}  dataset = {ds.name}  (n={ds.n}, d={ds.X.shape[1]})")
        print("=" * 64)
        X_t = to_tensor(ds.X, device)
        w = default_weights(ds.n, device)
        init = pca_init(ds.X)

        for m in SPECTRAL_METHODS:
            t0 = time.time()
            Y_ref = m.reference(ds.X, ds)
            K_in = m.input_kernel(X_t, ds, device, w)
            Y_fw, rv = rv_embed(K_in, init, device, output_kernel="linear", weights=w)

            disparity = ix.procrustes_disparity(Y_ref, Y_fw)
            knn = ix.knn_overlap(Y_ref, Y_fw, k=TRUST_K)

            # identity indices (framework vs reference)
            log.add(
                SECTION,
                ds.name,
                m.name,
                "procrustes",
                disparity,
                variant="overlay",
                k=TRUST_K,
            )
            log.add(
                SECTION,
                ds.name,
                m.name,
                "knn_overlap",
                knn,
                variant="overlay",
                k=TRUST_K,
            )
            log.add(SECTION, ds.name, m.name, "rv_final", rv, variant="framework")

            # quality indices for BOTH embeddings
            for variant, Y in (("framework", Y_fw), ("reference", Y_ref)):
                trust = ix.trustworthiness(ds.X, Y, k=TRUST_K)
                qnx = ix.qnx_curve(ds.X, Y, ix.QNX_KS)
                qloc = ix.q_local(qnx, ix.QNX_KS)
                log.add(
                    SECTION,
                    ds.name,
                    m.name,
                    "trustworthiness",
                    trust,
                    variant=variant,
                    k=TRUST_K,
                )
                log.add(SECTION, ds.name, m.name, "q_local", qloc, variant=variant)
                if ds.has_labels:
                    log.add(
                        SECTION,
                        ds.name,
                        m.name,
                        "ari",
                        ix.ari(Y, ds.labels),
                        variant=variant,
                    )

            overlay_figure(ds, m.name, m.key, Y_ref, Y_fw, disparity, rv)
            print(
                f"  {m.name:<22} proc={disparity:.4f} knn={knn:.3f} "
                f"rv={rv:.4f}  ({time.time() - t0:.1f}s)"
            )

    path = log.write(append=False)  # §5.3.1 runs first → fresh CSV
    print(f"\nSaved indices → {path}")


if __name__ == "__main__":
    main()
