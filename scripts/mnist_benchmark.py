"""
mnist_benchmark.py
==================
Compares rv_dimred embeddings against standard reference implementations on
MNIST (same 2000-point sample as mnist_combination.py) using:

  [Identité]  for spectral methods (PCA, Isomap, LLE, Laplacian Eigenmaps,
               Diffusion Maps, Kernel PCA, LDA):
               – Procrustes disparity  (≈ 0 → identical shape up to rigid alignment)
               – RV coefficient        (≈ 1 → linear kernels aligned)
               – kNN overlap           (local agreement between the two embeddings)

  [Qualité]   for t-SNE and UMAP (compared to a random stochastic floor):
               – Trustworthiness & Continuity
               – QNX(k) / BNX(k)  (fraction of k-NN preserved high→low / above chance)
               – kNN-accuracy (5-fold CV) / ARI (KMeans vs true labels)
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import umap
from scipy.linalg import eigh as scipy_eigh
from scipy.spatial import procrustes as scipy_procrustes
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import (
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
    trustworthiness,
)
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from src.rv_kernels import (
    compute_class_kernel_torch,
    compute_diffusion_kernel_torch,
    compute_fuzzy_topological_kernel_torch,
    compute_gaussian_affinity_kernel_torch,
    compute_geodesic_kernel_torch,
    compute_laplacian_kernel_torch,
    compute_linear_kernel_torch,
    compute_lle_kernel_torch,
    compute_rbf_kernel_torch,
    default_weights,
    rv_coefficient,
    rv_dimred,
)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"device: {device}\n")

# ── Data (same sample as mnist_combination.py) ────────────────────────────────
mnist_data = np.genfromtxt(
    Path(__file__).parent.parent / "data/mnist_test.csv", delimiter=",", skip_header=1
)
n_per_digit = 200
mnist_data = np.vstack(
    [mnist_data[mnist_data[:, 0] == i][:n_per_digit] for i in range(10)]
)
X_np = (mnist_data[:, 1:] / 255.0).astype(np.float32)
Y_labels = mnist_data[:, 0].astype(int)
n = X_np.shape[0]
print(f"n = {n} points,  d = {X_np.shape[1]} features\n")

X = torch.tensor(X_np, dtype=torch.float32, device=device)
w = default_weights(n, device)

# ── Hyperparameters ───────────────────────────────────────────────────────────
K_NEIGHBORS = 15  # kNN for graph-based methods
PERPLEXITY = 40  # t-SNE perplexity
DIFFUSION_T = 1.0  # diffusion time
Q = 2  # embedding dimension
K_EVAL = 10  # k used by all evaluation indices
N_ITER_RV = 500  # rv_dimred gradient-ascent iterations

# ── Shared PCA initialisation (all spectral rv_dimred runs start here) ────────
Y_pca_np = PCA(n_components=Q).fit_transform(X_np)
Y_pca_t = torch.tensor(Y_pca_np, dtype=torch.float32, device=device)

# ── gamma for RBF kernel (median-distance heuristic, same as rv_kernels) ─────
sq = (X_np**2).sum(axis=1)
d2_full = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X_np @ X_np.T), 0.0)
np.fill_diagonal(d2_full, 0.0)
gamma_rbf = float(1.0 / (np.median(d2_full[d2_full > 0]) + 1e-12))
del d2_full  # free memory


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════


def procrustes_dist(Y1: np.ndarray, Y2: np.ndarray) -> float:
    """Procrustes disparity after optimal rigid alignment (rotation + scale)."""
    _, _, d = scipy_procrustes(Y1, Y2)
    return float(d)


def rv_between(Y1: np.ndarray, Y2: np.ndarray) -> float:
    """RV coefficient between the centred linear kernels of Y1 and Y2."""
    k1 = compute_linear_kernel_torch(
        torch.tensor(Y1, dtype=torch.float32, device=device), weights=w, device=device
    )
    k2 = compute_linear_kernel_torch(
        torch.tensor(Y2, dtype=torch.float32, device=device), weights=w, device=device
    )
    return float(rv_coefficient(k1, k2))


def knn_overlap_2emb(Y1: np.ndarray, Y2: np.ndarray, k: int = K_EVAL) -> float:
    """Mean fraction of k-NN shared between two embeddings (no alignment needed)."""
    idx1 = NearestNeighbors(n_neighbors=k + 1).fit(Y1).kneighbors(Y1)[1][:, 1:]
    idx2 = NearestNeighbors(n_neighbors=k + 1).fit(Y2).kneighbors(Y2)[1][:, 1:]
    return float(np.mean([len(set(idx1[i]) & set(idx2[i])) / k for i in range(n)]))


def trust_score(X_high: np.ndarray, Y_emb: np.ndarray, k: int = K_EVAL) -> float:
    """Trustworthiness: no false neighbours in the embedding."""
    return float(trustworthiness(X_high, Y_emb, n_neighbors=k))


def cont_score(X_high: np.ndarray, Y_emb: np.ndarray, k: int = K_EVAL) -> float:
    """Continuity: original neighbours are preserved in the embedding."""
    # Roles swapped: look at kNN in X_high, penalise by rank in Y_emb.
    return float(trustworthiness(Y_emb, X_high, n_neighbors=k))


def qnx_bnx(
    X_high: np.ndarray, Y_emb: np.ndarray, k: int = K_EVAL
) -> tuple[float, float]:
    """QNX(k): fraction of k-NN preserved from high-dim to embedding.
    BNX(k) = QNX(k) - k/(n-1)  (above-chance component).
    """
    idx_h = NearestNeighbors(n_neighbors=k + 1).fit(X_high).kneighbors(X_high)[1][:, 1:]
    idx_e = NearestNeighbors(n_neighbors=k + 1).fit(Y_emb).kneighbors(Y_emb)[1][:, 1:]
    total = sum(len(set(idx_h[i]) & set(idx_e[i])) for i in range(n))
    qnx = total / (n * k)
    bnx = qnx - k / (n - 1)
    return float(qnx), float(bnx)


def knn_acc(Y_emb: np.ndarray, labels: np.ndarray, k: int = 5, cv: int = 5) -> float:
    """kNN classification accuracy (k=5, 5-fold cross-validation)."""
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), Y_emb, labels, cv=cv)
    return float(scores.mean())


def ari_score(Y_emb: np.ndarray, labels: np.ndarray, n_cl: int = 10) -> float:
    """Adjusted Rand Index between KMeans clustering and true labels."""
    pred = KMeans(n_clusters=n_cl, n_init=10, random_state=42).fit_predict(Y_emb)
    return float(adjusted_rand_score(labels, pred))


def _diffusion_maps_ref(
    X: np.ndarray, k: int = K_NEIGHBORS, t: float = DIFFUSION_T, q: int = Q
) -> np.ndarray:
    """Diffusion maps via eigenvectors of L_sym — same kNN affinity as rv_kernels."""
    _n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nbrs.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    sigma = float(np.median(dist[:, -1])) + 1e-12
    W = np.zeros((_n, _n))
    for i in range(_n):
        W[i, idx[i]] = np.exp(-(dist[i] ** 2) / sigma**2)
    W = np.maximum(W, W.T)
    d_inv_sqrt = 1.0 / np.sqrt(W.sum(1) + 1e-12)
    L_sym = np.eye(_n) - d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
    vals, vecs = scipy_eigh(L_sym, subset_by_index=[0, q])
    # skip eigenvalue ≈ 0 (index 0); scale remaining by exp(-t * lambda)
    return (vecs[:, 1 : q + 1] * np.exp(-t * vals[1 : q + 1])).astype(np.float32)


def _tic(label: str) -> float:
    print(f"  {label} ...", end=" ", flush=True)
    return time.time()


def _toc(t0: float) -> None:
    print(f"({time.time() - t0:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════════
# Reference embeddings
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("Reference embeddings")
print("=" * 62)

t0 = _tic("PCA")
Y_pca_ref = Y_pca_np.copy()  # already computed
_toc(t0)

t0 = _tic("Isomap")
Y_isomap_ref = Isomap(n_neighbors=K_NEIGHBORS, n_components=Q).fit_transform(X_np)
_toc(t0)

t0 = _tic("LLE")
Y_lle_ref = LocallyLinearEmbedding(
    n_neighbors=K_NEIGHBORS, n_components=Q, random_state=0
).fit_transform(X_np)
_toc(t0)

t0 = _tic("Laplacian Eigenmaps")
Y_lap_ref = SpectralEmbedding(
    n_neighbors=K_NEIGHBORS, n_components=Q, random_state=0
).fit_transform(X_np)
_toc(t0)

t0 = _tic("Diffusion Maps")
Y_diff_ref = _diffusion_maps_ref(X_np)
_toc(t0)

t0 = _tic("Kernel PCA (RBF)")
Y_kpca_ref = KernelPCA(n_components=Q, kernel="rbf", gamma=gamma_rbf).fit_transform(
    X_np
)
_toc(t0)

t0 = _tic("LDA")
Y_lda_ref = LinearDiscriminantAnalysis(n_components=Q).fit_transform(X_np, Y_labels)
_toc(t0)

t0 = _tic("t-SNE")
Y_tsne_ref = TSNE(n_components=Q, perplexity=PERPLEXITY, random_state=0).fit_transform(
    X_np
)
_toc(t0)

t0 = _tic("UMAP")
Y_umap_ref = umap.UMAP(
    n_neighbors=K_NEIGHBORS, n_components=Q, random_state=0
).fit_transform(X_np)
_toc(t0)

# ══════════════════════════════════════════════════════════════════════════════
# Input kernels (computed once, reused for rv_dimred)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("Input kernels")
print("=" * 62)

t0 = _tic("Linear")
K_linear = compute_linear_kernel_torch(X, weights=w, device=device)
_toc(t0)

t0 = _tic("Geodesic (Isomap)")
K_geodesic = compute_geodesic_kernel_torch(
    X, param={"k": K_NEIGHBORS}, weights=w, device=device
)
_toc(t0)

t0 = _tic("LLE")
K_lle = compute_lle_kernel_torch(X, param={"k": K_NEIGHBORS}, weights=w, device=device)
_toc(t0)

t0 = _tic("Laplacian")
K_laplacian = compute_laplacian_kernel_torch(
    X, param={"k": K_NEIGHBORS}, weights=w, device=device
)
_toc(t0)

t0 = _tic("Diffusion")
K_diffusion = compute_diffusion_kernel_torch(
    X, param={"k": K_NEIGHBORS, "t": DIFFUSION_T}, weights=w, device=device
)
_toc(t0)

t0 = _tic("RBF")
K_rbf = compute_rbf_kernel_torch(X, param=gamma_rbf, weights=w, device=device)
_toc(t0)

t0 = _tic("Class (LDA)")
K_class = compute_class_kernel_torch(
    X, param={"labels": Y_labels}, weights=w, device=device
)
_toc(t0)

t0 = _tic("Gaussian affinity (t-SNE)")
K_gauss = compute_gaussian_affinity_kernel_torch(
    X, param={"perplexity": PERPLEXITY}, weights=w, device=device
)
_toc(t0)

t0 = _tic("Fuzzy topological (UMAP)")
K_fuzzy = compute_fuzzy_topological_kernel_torch(
    X, param={"k": K_NEIGHBORS}, weights=w, device=device
)
_toc(t0)

# ══════════════════════════════════════════════════════════════════════════════
# rv_dimred embeddings
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print(f"rv_dimred optimisation  ({N_ITER_RV} iterations, PCA init)")
print("=" * 62)


def _rv_embed(
    name: str,
    K_in: torch.Tensor,
    output: str = "linear",
    out_param: object = None,
) -> np.ndarray:
    t0 = time.time()
    print(f"  [{name}] ...", end=" ", flush=True)
    Y_opt, rv_val = rv_dimred(
        K_in,
        output_kernel=output,
        q=Q,
        weights=w,
        output_param=out_param,
        n_iter=N_ITER_RV,
        lr=0.1,
        device=device,
        init=Y_pca_t,
        verbose=False,
    )
    print(f"RV = {rv_val:.4f}  ({time.time() - t0:.1f}s)")
    return Y_opt.cpu().numpy()


Y_pca_rv = _rv_embed("PCA", K_linear, "linear")
Y_isomap_rv = _rv_embed("Isomap", K_geodesic, "linear")
Y_lle_rv = _rv_embed("LLE", K_lle, "linear")
Y_lap_rv = _rv_embed("Laplacian E.", K_laplacian, "linear")
Y_diff_rv = _rv_embed("Diffusion Maps", K_diffusion, "linear")
Y_kpca_rv = _rv_embed("Kernel PCA", K_rbf, "linear")
Y_lda_rv = _rv_embed("LDA", K_class, "linear")
Y_tsne_rv = _rv_embed("t-SNE", K_gauss, "student_t", 1.0)
Y_umap_rv = _rv_embed("UMAP", K_fuzzy, "umap")

# ══════════════════════════════════════════════════════════════════════════════
# [Identité] table — spectral methods
# ══════════════════════════════════════════════════════════════════════════════
spectral_pairs = [
    ("PCA", Y_pca_ref, Y_pca_rv),
    ("Isomap", Y_isomap_ref, Y_isomap_rv),
    ("LLE", Y_lle_ref, Y_lle_rv),
    ("Laplacian E.", Y_lap_ref, Y_lap_rv),
    ("Diffusion Maps", Y_diff_ref, Y_diff_rv),
    ("Kernel PCA (RBF)", Y_kpca_ref, Y_kpca_rv),
    ("LDA", Y_lda_ref, Y_lda_rv),
]

out_dir = Path(__file__).parent.parent / "results" / "mnist"
out_dir.mkdir(parents=True, exist_ok=True)

# ── [Identité] table — spectral methods ──────────────────────────────────────
print("\n" + "=" * 62)
print("[Identité]  rv_dimred  vs.  reference  (spectral methods)")
print(f"  kNN overlap at k = {K_EVAL}")
print("  Procrustes ≈ 0 and RV ≈ 1 → embeddings are equivalent")
print("=" * 62)
hdr = f"{'Method':<20} {'Procrustes':>12} {'RV':>8} {'kNN overlap':>12}"
print(hdr)
print("-" * len(hdr))

identity_rows: list[dict[str, object]] = []
for name, Y_ref, Y_rv in spectral_pairs:
    proc = procrustes_dist(Y_ref, Y_rv)
    rv = rv_between(Y_ref, Y_rv)
    knn = knn_overlap_2emb(Y_ref, Y_rv)
    print(f"{name:<20} {proc:>12.4f} {rv:>8.4f} {knn:>12.4f}")
    identity_rows.append(
        {"method": name, "procrustes": proc, "rv": rv, "knn_overlap": knn}
    )

# ── [Qualité] table — t-SNE and UMAP ─────────────────────────────────────────
rng = np.random.default_rng(42)
Y_random = rng.standard_normal((n, Q)).astype(np.float32)

quality_rows: list[tuple[str, np.ndarray]] = [
    ("Random (floor)", Y_random),
    ("t-SNE (ref)", Y_tsne_ref),
    ("rv_dimred t-SNE", Y_tsne_rv),
    ("UMAP (ref)", np.asarray(Y_umap_ref)),
    ("rv_dimred UMAP", Y_umap_rv),
]

print("\n" + "=" * 74)
print(f"[Qualité]  t-SNE & UMAP  (k = {K_EVAL})")
print(f"  QNX floor (chance) ≈ {K_EVAL / (n - 1):.4f}")
print("=" * 74)
hdr2 = (
    f"{'Method':<22} {'Trust':>7} {'Cont':>7} "
    f"{'QNX':>7} {'BNX':>8} {'kNN-acc':>9} {'ARI':>7}"
)
print(hdr2)
print("-" * len(hdr2))

quality_data: list[dict[str, object]] = []
for name, Y_emb in quality_rows:
    t_sc = trust_score(X_np, Y_emb)
    c_sc = cont_score(X_np, Y_emb)
    qnx, bnx = qnx_bnx(X_np, Y_emb)
    kacc = knn_acc(Y_emb, Y_labels)
    ar = ari_score(Y_emb, Y_labels)
    print(
        f"{name:<22} {t_sc:>7.4f} {c_sc:>7.4f} "
        f"{qnx:>7.4f} {bnx:>8.4f} {kacc:>9.4f} {ar:>7.4f}"
    )
    quality_data.append(
        {
            "method": name,
            "trustworthiness": t_sc,
            "continuity": c_sc,
            "qnx": qnx,
            "bnx": bnx,
            "knn_accuracy": kacc,
            "ari": ar,
        }
    )

# ── Save CSV results ──────────────────────────────────────────────────────────
identity_path = out_dir / "benchmark_identity.csv"
quality_path = out_dir / "benchmark_quality.csv"

with identity_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "procrustes", "rv", "knn_overlap"])
    writer.writeheader()
    writer.writerows(identity_rows)

with quality_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "method",
            "trustworthiness",
            "continuity",
            "qnx",
            "bnx",
            "knn_accuracy",
            "ari",
        ],
    )
    writer.writeheader()
    writer.writerows(quality_data)

print(f"\nSaved → {identity_path}")
print(f"Saved → {quality_path}")
print("\nDone.")
