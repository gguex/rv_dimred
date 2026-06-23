"""
run_approximations.py  —  §5.3.2  Approximations (geometric, not bit-exact)
==========================================================================

t-SNE and UMAP across the 3 datasets. Because the references are stochastic we
build a **stochastic floor**: 10 reference runs (different seeds, shared PCA
init) give a distribution of every index between reference runs. The framework's
distance to the reference is reported relative to that floor.

Outputs
  * indices appended to results/results_indices.csv (incl. floor mean±std rows
    and the UMAP random-init negative control),
  * side-by-side framework-vs-reference scatter per (method × dataset),
  * Q_NX(k) curve: framework vs reference vs floor band, per dataset,
  * UMAP PCA-init vs random-init control panel.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.manifold import TSNE

from src import indices as ix
from src.datasets import Dataset, load_all
from src.rv_kernels import (
    compute_fuzzy_topological_kernel_torch,
    compute_gaussian_affinity_kernel_torch,
    default_weights,
)
from src.section5_common import (
    K_NEIGHBORS,
    PERPLEXITY,
    SECTION_FLOOR_RUNS,
    SEED,
    TRUST_K,
    IndexLog,
    Q,
    fig_path,
    get_device,
    pca_init,
    pca_init_sklearn,
    rv_embed,
    to_tensor,
)

SECTION = "5.3.2"
N_FLOOR = SECTION_FLOOR_RUNS  # reference runs for the stochastic floor


# ── Reference embeddings ──────────────────────────────────────────────────────
def tsne_ref(X: np.ndarray, init: np.ndarray, seed: int) -> np.ndarray:
    return TSNE(
        n_components=Q,
        perplexity=PERPLEXITY,
        init=init,
        random_state=seed,
    ).fit_transform(X)


def umap_ref(X: np.ndarray, init: np.ndarray | str, seed: int) -> np.ndarray:
    return np.asarray(
        umap.UMAP(
            n_neighbors=K_NEIGHBORS,
            n_components=Q,
            init=init,
            random_state=seed,
        ).fit_transform(X)
    )


# ── Index helpers ─────────────────────────────────────────────────────────────
def all_indices(ds: Dataset, Y_a: np.ndarray, Y_b: np.ndarray) -> dict[str, float]:
    """Identity (a vs b) + quality (of Y_a) indices in one dict."""
    out = {
        "procrustes": ix.procrustes_disparity(Y_a, Y_b),
        "knn_overlap": ix.knn_overlap(Y_a, Y_b, k=TRUST_K),
        "trustworthiness": ix.trustworthiness(ds.X, Y_a, k=TRUST_K),
        "q_local": ix.q_local(ix.qnx_curve(ds.X, Y_a, ix.QNX_KS), ix.QNX_KS),
    }
    if ds.has_labels:
        out["ari"] = ix.ari(Y_a, ds.labels)
    return out


def scatter_pair(
    ds: Dataset, method: str, key: str, Y_fw: np.ndarray, Y_ref: np.ndarray
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, Y, title in ((axes[0], Y_fw, "framework"), (axes[1], Y_ref, "reference")):
        ax.scatter(
            Y[:, 0], Y[:, 1], c=ds.color, cmap=ds.cmap, s=8, alpha=0.8, linewidths=0
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{ds.name} — {method}", fontsize=11)
    fig.tight_layout()
    fig.savefig(
        fig_path(SECTION, ds.name, key, "overlay"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def qnx_figure(
    ds: Dataset,
    Y_fw_tsne: np.ndarray,
    Y_ref_tsne: np.ndarray,
    floor_curves: list[np.ndarray],
) -> None:
    """Q_NX(k): framework vs reference vs stochastic-floor band (t-SNE)."""
    ks = np.asarray(ix.QNX_KS)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    floor = np.vstack(floor_curves)
    lo, hi, mid = floor.min(0), floor.max(0), floor.mean(0)
    ax.fill_between(ks, lo, hi, color="grey", alpha=0.3, label="t-SNE floor (10 runs)")
    ax.plot(ks, mid, color="grey", lw=1, ls="--")
    ax.plot(
        ks,
        ix.qnx_curve(ds.X, Y_ref_tsne, ks),
        color="C0",
        lw=2,
        label="t-SNE reference",
    )
    ax.plot(
        ks,
        ix.qnx_curve(ds.X, Y_fw_tsne, ks),
        color="C3",
        lw=2,
        label="framework (t-SNE)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("Q_NX(k)")
    ax.set_title(f"{ds.name} — Q_NX(k)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path(SECTION, ds.name, "tsne", "qnx"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def umap_control_figure(
    ds: Dataset,
    Y_ref_pca: np.ndarray,
    Y_ref_rand: np.ndarray,
    d_internal: float,
    d_fw: float,
) -> None:
    """Negative control: reference UMAP with PCA init vs random init. Their mutual
    Procrustes (d_internal) bounds how much UMAP varies on its own; compare to the
    framework's distance to the PCA-init reference (d_fw)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, Y, title in (
        (axes[0], Y_ref_pca, "UMAP reference (PCA init)"),
        (axes[1], Y_ref_rand, "UMAP reference (random init)"),
    ):
        ax.scatter(
            Y[:, 0], Y[:, 1], c=ds.color, cmap=ds.cmap, s=8, alpha=0.8, linewidths=0
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{ds.name} — UMAP init control\n"
        f"ref PCA-vs-random Procrustes = {d_internal:.4f}   "
        f"framework-vs-ref = {d_fw:.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(
        fig_path(SECTION, ds.name, "umap", "control"), dpi=150, bbox_inches="tight"
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
        X_t = to_tensor(ds.X, device)
        w = default_weights(ds.n, device)
        init_fw = pca_init(ds.X)  # full-scale PCA — framework optimiser
        init_ref = pca_init_sklearn(ds.X)  # 1e-4 PCA — sklearn/UMAP convention

        # framework input kernels
        K_gauss = compute_gaussian_affinity_kernel_torch(
            X_t, param={"perplexity": PERPLEXITY}, weights=w, device=device
        )
        K_fuzzy = compute_fuzzy_topological_kernel_torch(
            X_t, param={"k": K_NEIGHBORS}, weights=w, device=device
        )

        # ---- framework embeddings (PCA init, deterministic) ----
        t0 = time.time()
        Y_fw_tsne, rv_t = rv_embed(
            K_gauss, init_fw, device, "student_t", 1.0, weights=w
        )
        Y_fw_umap, rv_u = rv_embed(K_fuzzy, init_fw, device, "umap", weights=w)
        dt = time.time() - t0
        print(f"  framework t-SNE RV={rv_t:.4f}  UMAP RV={rv_u:.4f}  ({dt:.1f}s)")

        # umap_ref0 is filled in the UMAP branch and reused by the control panel
        umap_ref0: np.ndarray | None = None

        for label, ref_fn, Y_fw, rv_fw, key in (
            ("t-SNE", tsne_ref, Y_fw_tsne, rv_t, "tsne"),
            ("UMAP", umap_ref, Y_fw_umap, rv_u, "umap"),
        ):
            # headline reference: shared PCA init, seed 0 → legitimate Procrustes
            Y_ref_pca = ref_fn(ds.X, init_ref, 0)

            # stochastic floor: 10 random-init runs capture run-to-run variation
            t0 = time.time()
            runs = [ref_fn(ds.X, "random", seed) for seed in range(N_FLOOR)]
            dt = time.time() - t0
            print(
                f"  {label}: headline + {N_FLOOR} random-init floor runs ({dt:.1f}s)"
            )

            # pairwise floor of identity indices between distinct reference runs
            floor: dict[str, list[float]] = {"procrustes": [], "knn_overlap": []}
            for a in range(N_FLOOR):
                for b in range(a + 1, N_FLOOR):
                    floor["procrustes"].append(
                        ix.procrustes_disparity(runs[a], runs[b])
                    )
                    floor["knn_overlap"].append(
                        ix.knn_overlap(runs[a], runs[b], k=TRUST_K)
                    )
            for iname, vals in floor.items():
                log.add(
                    SECTION,
                    ds.name,
                    label,
                    f"{iname}_floor_mean",
                    float(np.mean(vals)),
                    variant="floor",
                    init="random",
                    k=TRUST_K,
                )
                log.add(
                    SECTION,
                    ds.name,
                    label,
                    f"{iname}_floor_std",
                    float(np.std(vals)),
                    variant="floor",
                    init="random",
                    k=TRUST_K,
                )

            # quality floor (each reference run vs data/labels)
            for run_id, Yr in enumerate(runs):
                q = all_indices(ds, Yr, Y_ref_pca)
                for iname, val in q.items():
                    if iname in ("procrustes", "knn_overlap"):
                        continue
                    log.add(
                        SECTION,
                        ds.name,
                        label,
                        iname,
                        val,
                        variant="reference",
                        init="random",
                        k=TRUST_K,
                        run_id=run_id,
                    )

            # framework vs the headline (PCA-init) reference
            for iname, val in all_indices(ds, Y_fw, Y_ref_pca).items():
                log.add(
                    SECTION, ds.name, label, iname, val, variant="framework", k=TRUST_K
                )
            log.add(SECTION, ds.name, label, "rv_final", rv_fw, variant="framework")

            scatter_pair(ds, label, key, Y_fw, Y_ref_pca)

            if label == "t-SNE":
                floor_curves = [ix.qnx_curve(ds.X, Yr, ix.QNX_KS) for Yr in runs]
                qnx_figure(ds, Y_fw, Y_ref_pca, floor_curves)
            else:
                umap_ref0 = Y_ref_pca
                umap_rand0 = runs[0]

        # ---- UMAP negative control: reference PCA init vs random init (§0.3) ----
        # Procrustes(PCA-ref, random-ref) shows UMAP is intrinsically init-sensitive,
        # so the framework's distance to UMAP is not a defect of our method.
        assert umap_ref0 is not None
        d_internal = ix.procrustes_disparity(umap_ref0, umap_rand0)
        d_fw = ix.procrustes_disparity(Y_fw_umap, umap_ref0)
        log.add(
            SECTION,
            ds.name,
            "UMAP",
            "procrustes",
            d_internal,
            variant="reference",
            init="random_vs_pca",
            k=TRUST_K,
        )
        log.add(
            SECTION,
            ds.name,
            "UMAP",
            "procrustes",
            d_fw,
            variant="framework",
            init="pca",
            k=TRUST_K,
        )
        umap_control_figure(ds, umap_ref0, umap_rand0, d_internal, d_fw)
        print(
            f"  UMAP control: ref PCA-vs-random proc={d_internal:.4f}  "
            f"framework-vs-ref proc={d_fw:.4f}"
        )

    path = log.write(append=True)
    print(f"\nAppended indices → {path}")


if __name__ == "__main__":
    main()
