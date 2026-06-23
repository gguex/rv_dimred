"""
indices.py
==========

Similarity (identity) and quality indices for §5 (protocol §0.4).

Identity indices compare two *embeddings* (framework vs reference):
    * ``procrustes_disparity`` — disparity after optimal rigid alignment (~0 = match)
    * ``knn_overlap``          — mean Jaccard-style overlap of their kNN graphs

Quality indices compare an *embedding* to the high-dimensional data / labels:
    * ``trustworthiness``      — scalar at fixed k (sklearn)
    * ``qnx_curve`` / ``q_local`` — Q_NX(k) preservation curve and its AUC summary
    * ``ari``                  — adjusted Rand index of k-means(embedding) vs labels

``align_procrustes`` returns the reference-aligned framework embedding for overlay
plots (§5.3.1).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import procrustes as _scipy_procrustes
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness as _sk_trustworthiness
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# kNN helper
# ---------------------------------------------------------------------------
def _knn_indices(Y: np.ndarray, k: int) -> np.ndarray:
    """(n, k) indices of the k nearest neighbours of each row (self excluded)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Y)
    return nn.kneighbors(Y, return_distance=False)[:, 1:]


# ---------------------------------------------------------------------------
# Identity indices (embedding vs embedding)
# ---------------------------------------------------------------------------
def procrustes_disparity(Y1: np.ndarray, Y2: np.ndarray) -> float:
    """Procrustes disparity in [0, 1] after optimal translation/scale/rotation."""
    _, _, disparity = _scipy_procrustes(np.asarray(Y1), np.asarray(Y2))
    return float(disparity)


def align_procrustes(
    Y_ref: np.ndarray, Y_mov: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Standardise ``Y_ref`` and rigidly align ``Y_mov`` onto it.

    Returns (ref_std, mov_aligned, disparity) — both outputs share a frame so
    they can be superimposed in an overlay scatterplot.
    """
    ref_std, mov_aligned, disparity = _scipy_procrustes(
        np.asarray(Y_ref), np.asarray(Y_mov)
    )
    return ref_std, mov_aligned, float(disparity)


def knn_overlap(Y1: np.ndarray, Y2: np.ndarray, k: int = 15) -> float:
    """Mean fraction of shared k nearest neighbours between two embeddings."""
    i1, i2 = _knn_indices(Y1, k), _knn_indices(Y2, k)
    n = Y1.shape[0]
    return float(np.mean([len(set(i1[i]) & set(i2[i])) / k for i in range(n)]))


# ---------------------------------------------------------------------------
# Quality indices (embedding vs high-dim data / labels)
# ---------------------------------------------------------------------------
def trustworthiness(X_high: np.ndarray, Y_emb: np.ndarray, k: int = 15) -> float:
    """Trustworthiness at neighbourhood size k (no false neighbours)."""
    return float(_sk_trustworthiness(X_high, Y_emb, n_neighbors=k))


def qnx_curve(
    X_high: np.ndarray, Y_emb: np.ndarray, ks: np.ndarray | list[int]
) -> np.ndarray:
    """Q_NX(k) for each k in ``ks``: mean fraction of the high-dim k-NN preserved
    in the embedding. Computed once at k_max then truncated for efficiency."""
    ks = np.asarray(ks, dtype=int)
    k_max = int(ks.max())
    n = X_high.shape[0]
    idx_h = _knn_indices(X_high, k_max)
    idx_e = _knn_indices(Y_emb, k_max)
    # Precompute per-point sorted-membership for cumulative overlap by rank.
    out = np.empty(len(ks), dtype=float)
    sets_h = [set(idx_h[i]) for i in range(n)]
    for j, k in enumerate(ks):
        total = sum(len(sets_h[i] & set(idx_e[i, :k])) for i in range(n))
        out[j] = total / (n * k)
    return out


def q_local(qnx: np.ndarray, ks: np.ndarray | list[int]) -> float:
    """Scalar summary of a Q_NX curve: mean over the evaluated k grid (AUC/range)."""
    return float(np.mean(np.asarray(qnx)))


def ari(Y_emb: np.ndarray, labels: np.ndarray, random_state: int = 42) -> float:
    """Adjusted Rand index of k-means(embedding) vs true labels (k = #classes)."""
    n_clusters = len(np.unique(labels))
    pred = KMeans(
        n_clusters=n_clusters, n_init=10, random_state=random_state
    ).fit_predict(Y_emb)
    return float(adjusted_rand_score(labels, pred))


# k grid for the Q_NX(k) curve (§5.3.2 figure 2)
QNX_KS = [1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
