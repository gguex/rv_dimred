"""
benchmark_common.py
===================

Shared machinery for the dimensionality-reduction benchmarks: device handling,
the fixed hyperparameters and seeds, the shared PCA initialisation, reference
spectral embeddings, the RV-maximisation runner, the spectral-method registry,
the tidy-CSV writer and the figure/coordinate path helpers.

The aim is that every driver script (spectral / approximations / hybrids) builds
on the *same* kernels, init and indices so results are comparable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.linalg import eigh as scipy_eigh
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import (
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)
from sklearn.neighbors import NearestNeighbors

from src.datasets import Dataset
from src.rv_kernels import (
    compute_diffusion_kernel_torch,
    compute_geodesic_kernel_torch,
    compute_laplacian_kernel_torch,
    compute_linear_kernel_torch,
    compute_lle_kernel_torch,
    compute_rbf_kernel_torch,
    rv_dimred,
)

# ── Protocol constants (§0) ───────────────────────────────────────────────────
SEED = 0
Q = 2  # embedding dimension
K_NEIGHBORS = 15  # kNN for graph-based methods / UMAP n_neighbors
PERPLEXITY = 30  # t-SNE perplexity / adaptive-Gaussian input kernel
DIFFUSION_T = 10.0  # diffusion time (larger t sharpens the spectral gap → recovery)
N_ITER_RV = 500  # RV gradient-ascent iterations
LR_RV = 0.1  # RV gradient-ascent learning rate
TRUST_K = 15  # k for the scalar trustworthiness / kNN overlap

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
COORDS_ROOT = RESULTS_DIR / "coordinates"  # saved embeddings, one subdir per family

# Hybrid sweep grid (§5.3.3), shared by the run / indices / figures scripts.
HYBRID_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
# ν=∞ rendered via the Gaussian (SNE) output kernel; finite ν via Student-t.
HYBRID_NUS: list[tuple[str, str, float | None]] = [
    ("inf", "gaussian", None),  # ν → ∞  (Gaussian / light tail)
    ("1.0", "student_t", 1.0),  # ν = 1  (classic t-SNE tail)
    ("0.5", "student_t", 0.5),  # ν = 0.5 (heavy tail)
]

# tidy-CSV schema (§0.5)
CSV_COLUMNS = [
    "section",
    "dataset",
    "method",
    "variant",
    "init",
    "index_name",
    "k",
    "value",
    "run_id",
]


# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "mps", None) is not None and torch.mps.is_available():
        return "mps"
    return "cpu"


# ── Shared PCA initialisation (§0.2) ──────────────────────────────────────────
def pca_embedding(X: np.ndarray, q: int = Q) -> np.ndarray:
    """Plain PCA projection to q dims (also the PCA *reference* embedding)."""
    return PCA(n_components=q, random_state=SEED).fit_transform(X).astype(np.float32)


def pca_init(X: np.ndarray, q: int = Q) -> np.ndarray:
    """Framework init: full-scale PCA projection. Same PCA *directions* as the
    sklearn-convention init below, but at a numerically friendly scale so the RV
    gradient-ascent (Adam) converges for linear output kernels too. Procrustes/RV
    are scale-invariant, so comparing against a reference started from
    :func:`pca_init_sklearn` stays legitimate (§0.2)."""
    return pca_embedding(X, q)


def pca_init_sklearn(X: np.ndarray, q: int = Q) -> np.ndarray:
    """Reference init for sklearn t-SNE / UMAP: PCA scaled so PC1 has std 1e-4
    (sklearn's documented convention). Same directions as :func:`pca_init`."""
    Y0 = pca_embedding(X, q).copy()
    Y0 = Y0 / (Y0[:, 0].std() + 1e-12) * 1e-4
    return Y0.astype(np.float32)


def normalize_kernel(K: torch.Tensor) -> torch.Tensor:
    """Scale a kernel to unit Frobenius norm. RV is scale-invariant, so this leaves
    single-kernel results unchanged but makes a convex combination
    ``α·K1 + (1-α)·K2`` a genuine interpolation — otherwise the kernel with the
    larger norm dominates (e.g. linear ‖K‖≈9 vs adaptive-Gaussian ‖K‖≈3e-6)."""
    return K / (K.norm() + 1e-12)


def rbf_gamma(X: np.ndarray) -> float:
    """Median-distance gamma heuristic, matching rv_kernels' RBF default."""
    sq = (X**2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
    np.fill_diagonal(d2, 0.0)
    return float(1.0 / (np.median(d2[d2 > 0]) + 1e-12))


# ── RV runner ─────────────────────────────────────────────────────────────────
def to_tensor(X: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)


def rv_embed(
    K_in: torch.Tensor,
    init: np.ndarray,
    device: str,
    output_kernel: str | Callable[..., torch.Tensor] = "linear",
    output_param: Any = None,
    weights: torch.Tensor | None = None,
    n_iter: int = N_ITER_RV,
    lr: float = LR_RV,
) -> tuple[np.ndarray, float]:
    """Run RV gradient ascent from the shared init; return (embedding, final RV)."""
    Y, rv = rv_dimred(
        K_in,
        output_kernel=output_kernel,
        q=init.shape[1],
        weights=weights,
        output_param=output_param,
        n_iter=n_iter,
        lr=lr,
        device=device,
        init=to_tensor(init, device),
        verbose=False,
    )
    return Y.cpu().numpy(), float(rv)


# ── Diffusion-maps reference (same kNN affinity as rv_kernels) ─────────────────
def diffusion_maps_ref(
    X: np.ndarray, k: int = K_NEIGHBORS, t: float = DIFFUSION_T, q: int = Q
) -> np.ndarray:
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nbrs.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    sigma = float(np.median(dist[:, -1])) + 1e-12
    W = np.zeros((n, n))
    for i in range(n):
        W[i, idx[i]] = np.exp(-(dist[i] ** 2) / sigma**2)
    W = np.maximum(W, W.T)
    d_inv_sqrt = 1.0 / np.sqrt(W.sum(1) + 1e-12)
    L_sym = np.eye(n) - d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
    vals, vecs = scipy_eigh(L_sym, subset_by_index=[0, q])
    return (vecs[:, 1 : q + 1] * np.exp(-t * vals[1 : q + 1])).astype(np.float32)


# ── Spectral method registry (§5.1.1) ─────────────────────────────────────────
@dataclass
class SpectralMethod:
    key: str
    name: str
    # input kernel: (X_tensor, ds, device, weights) -> centred kernel tensor
    input_kernel: Callable[[torch.Tensor, Dataset, str, torch.Tensor], torch.Tensor]
    # reference embedding: (X_np, ds) -> (n, q) array
    reference: Callable[[np.ndarray, Dataset], np.ndarray]


def _k_pca_ref(X: np.ndarray, ds: Dataset) -> np.ndarray:
    return KernelPCA(
        n_components=Q, kernel="rbf", gamma=rbf_gamma(X), random_state=SEED
    ).fit_transform(X)


SPECTRAL_METHODS: list[SpectralMethod] = [
    SpectralMethod(
        key="pca",
        name="PCA",
        input_kernel=lambda X, ds, dev, w: compute_linear_kernel_torch(
            X, weights=w, device=dev
        ),
        reference=lambda X, ds: pca_embedding(X),
    ),
    SpectralMethod(
        key="kpca",
        name="Kernel PCA (RBF)",
        input_kernel=lambda X, ds, dev, w: compute_rbf_kernel_torch(
            X, param=rbf_gamma(X.detach().cpu().numpy()), weights=w, device=dev
        ),
        reference=_k_pca_ref,
    ),
    SpectralMethod(
        key="isomap",
        name="Isomap",
        input_kernel=lambda X, ds, dev, w: compute_geodesic_kernel_torch(
            X, param={"k": K_NEIGHBORS}, weights=w, device=dev
        ),
        reference=lambda X, ds: Isomap(
            n_neighbors=K_NEIGHBORS, n_components=Q
        ).fit_transform(X),
    ),
    SpectralMethod(
        key="lle",
        name="LLE",
        input_kernel=lambda X, ds, dev, w: compute_lle_kernel_torch(
            X, param={"k": K_NEIGHBORS}, weights=w, device=dev
        ),
        reference=lambda X, ds: LocallyLinearEmbedding(
            n_neighbors=K_NEIGHBORS, n_components=Q, random_state=SEED
        ).fit_transform(X),
    ),
    SpectralMethod(
        key="diffusion",
        name="Diffusion Maps",
        input_kernel=lambda X, ds, dev, w: compute_diffusion_kernel_torch(
            X, param={"k": K_NEIGHBORS, "t": DIFFUSION_T}, weights=w, device=dev
        ),
        reference=lambda X, ds: diffusion_maps_ref(X),
    ),
    SpectralMethod(
        key="laplacian",
        name="Laplacian Eigenmaps",
        input_kernel=lambda X, ds, dev, w: compute_laplacian_kernel_torch(
            X, param={"k": K_NEIGHBORS}, weights=w, device=dev
        ),
        reference=lambda X, ds: SpectralEmbedding(
            n_neighbors=K_NEIGHBORS, n_components=Q, random_state=SEED
        ).fit_transform(X),
    ),
]


# ── Tidy CSV (§0.5) ───────────────────────────────────────────────────────────
class IndexLog:
    """Accumulates tidy index rows and appends/writes results_indices.csv."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        section: str,
        dataset: str,
        method: str,
        index_name: str,
        value: float,
        variant: str = "framework",
        init: str = "pca",
        k: Any = "",
        run_id: Any = "",
    ) -> None:
        self.rows.append(
            {
                "section": section,
                "dataset": dataset,
                "method": method,
                "variant": variant,
                "init": init,
                "index_name": index_name,
                "k": k,
                "value": value,
                "run_id": run_id,
            }
        )

    def write(self, path: Path | None = None, append: bool = True) -> Path:
        path = path or (RESULTS_DIR / "results_indices.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        mode = "a" if (append and exists) else "w"
        with path.open(mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if mode == "w":
                writer.writeheader()
            writer.writerows(self.rows)
        return path


def fig_path(section: str, dataset: str, method: str, variant: str) -> Path:
    """results/fig_{section}_{dataset}_{method}_{variant}.png (§0.5)."""
    d = RESULTS_DIR / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"fig_{section}_{dataset}_{method}_{variant}.png"


def coords_dir(family: str) -> Path:
    """results/coordinates/{family}/ ('spectral' | 'approximations' | 'hybrids')."""
    d = COORDS_ROOT / family
    d.mkdir(parents=True, exist_ok=True)
    return d


def coord_path(family: str, dataset: str, key: str, role: str) -> Path:
    """results/coordinates/{family}/{dataset}__{key}__{role}.npy. ``role`` labels
    the embedding (e.g. 'reference', 'framework', 'framework_linear', 'embedding')."""
    return coords_dir(family) / f"{dataset}__{key}__{role}.npy"


def meta_path(family: str) -> Path:
    """results/coordinates/{family}/run_meta.csv (carries final RV per run, since
    RV cannot be recomputed from the output coordinates alone)."""
    return coords_dir(family) / "run_meta.csv"


def drop_sections(sections: set[str], path: Path | None = None) -> None:
    """Remove rows of the given ``section`` values from results_indices.csv, keeping
    every other row. Lets a driver re-run its own section without clobbering the
    others (e.g. §5.3.1 / §5.3.2 re-run while §5.3.3 results are preserved)."""
    path = path or (RESULTS_DIR / "results_indices.csv")
    if not path.exists():
        return
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["section"] not in sections]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_pair_figures(
    section: str,
    ds: Dataset,
    method_key: str,
    method_name: str,
    Y_fw: np.ndarray,
    Y_ref: np.ndarray,
    fw_note: str = "",
    ref_note: str = "",
) -> None:
    """Save a *framework* and a *reference* scatter as two separate files (for
    side-by-side placement in LaTeX). Both are put in a shared Procrustes frame
    (reference standardised, framework rigidly aligned onto it) so the two panels
    are directly comparable."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src import indices as ix

    ref_std, fw_aligned, _ = ix.align_procrustes(Y_ref, Y_fw)
    for variant, Y, note in (
        ("framework", fw_aligned, fw_note),
        ("reference", ref_std, ref_note),
    ):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            Y[:, 0], Y[:, 1], c=ds.color, cmap=ds.cmap, s=10, alpha=0.85, linewidths=0
        )
        title = f"{ds.name} — {method_name} ({variant})"
        if note:
            title += f"\n{note}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(
            fig_path(section, ds.name, method_key, variant),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
