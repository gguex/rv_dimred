"""
rv_kernels.py
=============

Kernel "lego pieces" for RV-coefficient dimensionality reduction.

Every kernel maps a coordinate / similarity input to a *centered* kernel matrix

        K = Q G Q^T ,     Q = diag(sqrt(f)) (I - 1 f^T)

so that K lies in the weighted kernel space K_n (i.e. K sqrt(f) = 0).
All kernels share the same signature

        compute_*_kernel_torch(coords, param=None, weights=None, device='cpu') -> (n,n) torch.Tensor

so that an INPUT kernel and an OUTPUT kernel can be plugged into the optimiser
like two interchangeable bricks (see `rv_dimred` and the __main__ demo).

----------------------------------------------------------------------------
INPUT vs OUTPUT kernels
----------------------------------------------------------------------------
* OUTPUT kernels (linear, polynomial, gaussian, student_t, umap) are written in
  pure differentiable PyTorch: K_Y is a function of the embedding Y = `coords`,
  and autograd back-propagates through them during gradient ascent on RV.

* INPUT kernels (linear, geodesic, lle, laplacian, diffusion, gaussian_affinity,
  fuzzy_topological, rbf, class) are computed ONCE from the fixed data X.
  Graph-based ones rely on numpy/scipy/sklearn internally and return a constant
  tensor; they need not be differentiable.

`param` is kernel-specific and documented in each docstring (None = sensible
defaults). It can be a scalar, a tuple, or a dict.
----------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

# scipy / sklearn are only needed for the graph-based INPUT kernels.
# Declared as Any so type checkers accept usages even when the import is absent.
expm: Any = None
pinvh: Any = None
shortest_path: Any = None
csgraph_laplacian: Any = None
NearestNeighbors: Any = None
kneighbors_graph: Any = None

try:
    from scipy.linalg import expm, pinvh
    from scipy.sparse.csgraph import laplacian as csgraph_laplacian
    from scipy.sparse.csgraph import shortest_path
    from sklearn.neighbors import NearestNeighbors, kneighbors_graph

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# ===========================================================================
# Core helpers
# ===========================================================================
def default_weights(n: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Uniform relative weights f_i = 1/n."""
    return torch.ones(n, device=device) / n


def centering_operator(
    weights: torch.Tensor, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Q = diag(sqrt(f)) (I - 1 f^T). Note: Q is NOT symmetric when f is non-uniform."""
    n = weights.shape[0]
    H = torch.eye(n, device=device) - torch.outer(torch.ones(n, device=device), weights)
    Q = torch.diag(torch.sqrt(weights)) @ H
    return Q


def double_center(
    G: torch.Tensor, weights: torch.Tensor, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Map a raw Gram/affinity matrix G to the centered kernel K = Q G Q^T in K_n.

    Differentiable in G (Q depends only on the fixed weights), so OUTPUT kernels
    that build G from the embedding remain autograd-friendly.
    """
    Q = centering_operator(weights, device=device)
    return Q @ G @ Q.T


def _as_tensor(coords: Any, device: str | torch.device = "cpu") -> torch.Tensor:
    if isinstance(coords, torch.Tensor):
        return coords.to(device)
    return torch.as_tensor(np.asarray(coords), dtype=torch.float32, device=device)


def _np(coords: np.ndarray | torch.Tensor) -> np.ndarray:
    """Detach to a numpy array (for graph-based input kernels)."""
    if isinstance(coords, torch.Tensor):
        return coords.detach().cpu().numpy()
    return np.asarray(coords, dtype=np.float64)


def pairwise_sq_dists(Y: torch.Tensor) -> torch.Tensor:
    """(n,n) matrix of squared Euclidean distances, differentiable, clamped >= 0."""
    sq = (Y * Y).sum(dim=1)
    d2 = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (Y @ Y.T)
    return d2.clamp_min(0.0)


# ===========================================================================
# OUTPUT kernels  (differentiable in the embedding Y = coords)
# ===========================================================================
def compute_linear_kernel_torch(
    coords: np.ndarray | torch.Tensor,
    param: Any = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Linear / dot-product kernel: G = Y Y^T.  Recovers PCA / cMDS (spectral readout)."""
    coords = _as_tensor(coords, device)
    n = coords.shape[0]
    weights = default_weights(n, device) if weights is None else weights.to(device)
    G = coords @ coords.T
    return double_center(G, weights, device)


def compute_polynomial_kernel_torch(
    coords: np.ndarray | torch.Tensor,
    param: dict[str, Any] | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Polynomial dot-product kernel: G = (Y Y^T / q + c)^d.  -> 'curved' embeddings.

    param: dict {'c': float (default 1.0), 'd': int (default 2)}.
    The 1/q normalisation (q = output dim) keeps the kernel well-scaled.
    """
    coords = _as_tensor(coords, device)
    n, q = coords.shape[0], coords.shape[1]
    weights = default_weights(n, device) if weights is None else weights.to(device)
    param = param or {}
    c = float(param.get("c", 1.0))
    d = int(param.get("d", 2))
    G = (coords @ coords.T / q + c) ** d
    return double_center(G, weights, device)


def compute_gaussian_kernel_torch(
    coords: np.ndarray | torch.Tensor,
    param: float | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Gaussian / RBF output kernel: G_ij = exp(-||y_i - y_j||^2 / (2 sigma^2)).

    Light-tailed neighbour readout -> SNE-like behaviour.
    param: sigma (float, default 1.0).
    """
    coords = _as_tensor(coords, device)
    n = coords.shape[0]
    weights = default_weights(n, device) if weights is None else weights.to(device)
    sigma = 1.0 if param is None else float(param)
    d2 = pairwise_sq_dists(coords)
    G = torch.exp(-d2 / (2.0 * sigma**2))
    return double_center(G, weights, device)


def compute_student_t_kernel_torch(
    coords: np.ndarray | torch.Tensor,
    param: float | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Student-t output kernel with nu degrees of freedom:

        G_ij = (1 + ||y_i - y_j||^2 / nu)^{-(nu+1)/2}

    nu = 1   -> the classic t-SNE kernel (1 + ||.||^2)^{-1}
    nu -> inf -> recovers the Gaussian (SNE) kernel
    nu < 1   -> heavier tails (Kobak et al. 2020)
    param: nu (float, default 1.0).
    """
    coords = _as_tensor(coords, device)
    n = coords.shape[0]
    weights = default_weights(n, device) if weights is None else weights.to(device)
    nu = 1.0 if param is None else float(param)
    d2 = pairwise_sq_dists(coords)
    G = (1.0 + d2 / nu) ** (-(nu + 1.0) / 2.0)
    return double_center(G, weights, device)


def compute_umap_kernel_torch(
    coords: np.ndarray | torch.Tensor,
    param: dict[str, Any] | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """UMAP rational output kernel: G_ij = (1 + a ||y_i - y_j||^{2b})^{-1}.

    Generalises Student-t (a = b = 1). Defaults a=1.929, b=0.7915 match UMAP's
    min_dist=0.1. A tiny eps stabilises the b<1 power at zero distance.
    param: dict {'a': float, 'b': float}.
    """
    coords = _as_tensor(coords, device)
    n = coords.shape[0]
    weights = default_weights(n, device) if weights is None else weights.to(device)
    param = param or {}
    a = float(param.get("a", 1.929))
    b = float(param.get("b", 0.7915))
    d2 = pairwise_sq_dists(coords)
    G = 1.0 / (1.0 + a * (d2 + 1e-12) ** b)
    return double_center(G, weights, device)


# ===========================================================================
# INPUT kernels  (computed once from the fixed data X)
# ===========================================================================
def compute_geodesic_kernel_torch(
    coords: np.ndarray | torch.Tensor,
    param: dict[str, Any] | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Isomap input kernel: G = -1/2 D_geo^2, D_geo = symmetrised shortest paths
    on a kNN graph.  Recovers Isomap with a linear output.

    param: dict {'k': int (default 10)}.
    """
    if not _HAS_SCIPY:
        raise ImportError("geodesic kernel requires scipy + sklearn")
    X = _np(coords)
    n = X.shape[0]
    k = int((param or {}).get("k", 10))
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    graph = nbrs.kneighbors_graph(mode="distance")  # sparse kNN distances
    Dgeo = shortest_path(graph, method="D", directed=False)  # all-pairs geodesics
    # patch disconnected pairs (inf) with the largest finite geodesic
    finite = np.isfinite(Dgeo)
    Dgeo[~finite] = Dgeo[finite].max() if finite.any() else 0.0
    Dgeo = 0.5 * (Dgeo + Dgeo.T)
    G = -0.5 * (Dgeo**2)
    weights = default_weights(n, device) if weights is None else weights
    return double_center(_as_tensor(G, device), weights.to(device), device)


def compute_lle_kernel_torch(
    coords: np.ndarray | torch.Tensor,
    param: dict[str, Any] | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """LLE input kernel: G = pinv(Phi), Phi = (I - W)^T (I - W), W = local
    reconstruction weights.  The top eigenvectors of pinv(Phi) = the bottom
    eigenvectors of Phi = the standard LLE embedding.

    Regularisation matches sklearn's ``barycenter_weights``: R = reg·trace(C)
    added to the local Gram diagonal (no 1/k factor), so the framework reproduces
    sklearn ``LocallyLinearEmbedding`` faithfully.

    param: dict {'k': int (default 10), 'reg': float (default 1e-3)}.
    """
    if not _HAS_SCIPY:
        raise ImportError("LLE kernel requires sklearn")
    X = _np(coords)
    n = X.shape[0]
    p = param or {}
    k = int(p.get("k", 10))
    reg = float(p.get("reg", 1e-3))
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nbrs.kneighbors(X)
    idx = idx[:, 1:]  # drop self
    W = np.zeros((n, n))
    ones = np.ones(k)
    for i in range(n):
        Z = X[idx[i]] - X[i]  # (k, p): neighbours centred on x_i
        C = Z @ Z.T  # (k, k) local Gram
        trace = np.trace(C)
        R = reg * trace if trace > 0 else reg  # sklearn convention (no /k)
        C.flat[:: k + 1] += R  # Tikhonov regularisation on the diagonal
        w = np.linalg.solve(C, ones)
        w /= w.sum()
        W[i, idx[i]] = w
    I = np.eye(n)
    Phi = (I - W).T @ (I - W)
    G = pinvh(
        Phi
    )  # symmetric pseudo-inverse: top eigvecs <-> bottom nonzero eigvecs of Phi
    weights = default_weights(n, device) if weights is None else weights
    return double_center(_as_tensor(G, device), weights.to(device), device)


def _knn_affinity(
    X: np.ndarray,
    k: int,
    sigma: float | None = None,
    binary: bool = False,
) -> np.ndarray:
    """Symmetric kNN affinity W (numpy). Heat weights exp(-d^2/sigma^2) by default."""
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nbrs.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    if sigma is None:
        sigma = float(np.median(dist[:, -1])) + 1e-12  # local-scale heuristic
    W = np.zeros((n, n))
    for i in range(n):
        if binary:
            W[i, idx[i]] = 1.0
        else:
            W[i, idx[i]] = np.exp(-(dist[i] ** 2) / (sigma**2))
    return np.maximum(W, W.T)  # symmetrise


def compute_laplacian_kernel_torch(
    coords: np.ndarray | torch.Tensor,
    param: dict[str, Any] | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Laplacian-eigenmaps input kernel: a *binary* symmetric kNN connectivity graph
    A and the *normalised* Laplacian L_sym = I - D^{-1/2} A D^{-1/2}. The kernel is
    G = pinv(L_sym), whose top eigenvectors are the bottom eigenvectors of L_sym.

    Convention note. This yields the *symmetric-normalised* Laplacian eigenmaps
    (the raw eigenvectors u_i of L_sym), consistent with our Diffusion-Maps kernel.
    sklearn's ``SpectralEmbedding`` instead applies a *degree recovery*, returning
    D^{-1/2} u_i (the random-walk / Belkin-Niyogi eigenmaps, generalised problem
    L v = mu D v). The two coincide when degrees are uniform but diverge when they
    are heterogeneous: e.g. on single-cell data (degree CV ~ 0.76) the framework vs
    sklearn Procrustes is ~0.10, entirely explained by the D^{-1/2} factor; on
    Swiss-roll (CV ~ 0.10) it is negligible. This is a normalisation convention, not
    an error.

    param: dict {'k': int (default 10)}.
    """
    if not _HAS_SCIPY:
        raise ImportError("laplacian kernel requires scipy + sklearn")
    X = _np(coords)
    n = X.shape[0]
    k = int((param or {}).get("k", 10))
    A = kneighbors_graph(X, k, mode="connectivity", include_self=True)
    A = 0.5 * (A + A.T)  # symmetrise the binary connectivity
    A = np.asarray(A.todense())
    L_sym = csgraph_laplacian(A, normed=True)  # I - D^{-1/2} A D^{-1/2}
    G = pinvh(L_sym)  # symmetric pseudo-inverse
    weights = default_weights(n, device) if weights is None else weights
    return double_center(_as_tensor(G, device), weights.to(device), device)


def compute_diffusion_kernel_torch(
    coords: np.ndarray | torch.Tensor,
    param: dict[str, Any] | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Diffusion-map input kernel: heat kernel G = exp(-t L_sym),
    L_sym = I - D^{-1/2} W D^{-1/2}.  Top eigenvectors = diffusion coordinates;
    t (diffusion time) sets the analysis scale.

    param: dict {'k': int (default 10), 't': float (default 1.0), 'sigma': float}.
    """
    if not _HAS_SCIPY:
        raise ImportError("diffusion kernel requires scipy + sklearn")
    X = _np(coords)
    n = X.shape[0]
    p = param or {}
    t = float(p.get("t", 1.0))
    W = _knn_affinity(X, int(p.get("k", 10)), p.get("sigma", None))
    d = W.sum(1)
    d_inv_sqrt = 1.0 / np.sqrt(d + 1e-12)
    L_sym = np.eye(n) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
    G = expm(-t * L_sym)
    weights = default_weights(n, device) if weights is None else weights
    return double_center(_as_tensor(G, device), weights.to(device), device)


def _perplexity_probabilities(
    D2: np.ndarray,
    perplexity: float,
    tol: float = 1e-5,
    max_iter: int = 60,
) -> np.ndarray:
    """Conditional probabilities p_{j|i} with per-point bandwidths matching
    `perplexity` (classic t-SNE Hbeta binary search).  D2: (n,n) sq distances."""
    n = D2.shape[0]
    P = np.zeros((n, n))
    target = np.log(perplexity)
    for i in range(n):
        mask = np.arange(n) != i
        Di = D2[i, mask]
        beta, lo, hi = 1.0, -np.inf, np.inf
        Pi: np.ndarray = np.exp(-Di * beta)
        sumP: float = float(Pi.sum()) + 1e-12
        for _ in range(max_iter):
            Pi = np.exp(-Di * beta)
            sumP = float(Pi.sum()) + 1e-12
            H = np.log(sumP) + beta * (Di * Pi).sum() / sumP
            diff = H - target
            if abs(diff) < tol:
                break
            if diff > 0:  # entropy too high -> increase beta
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = beta / 2 if lo == -np.inf else (beta + lo) / 2
        P[i, mask] = Pi / sumP
    return P


def compute_gaussian_affinity_kernel_torch(
    coords: np.ndarray | torch.Tensor,
    param: dict[str, Any] | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Adaptive-Gaussian (perplexity) input kernel = the symmetrised t-SNE P matrix.
    With a Student-t output this gives t-SNE-like embeddings.

    param: dict {'perplexity': float (default 30), 'gamma': float (default 1.0)}.
           gamma softens the affinity via G = (P / max P)^gamma. gamma=0.5 is the
           Bhattacharyya/Hellinger variant (sharpens the intra/inter contrast,
           §5.3.2); gamma=1.0 leaves P unchanged (up to the RV-invariant scale).
    """
    if not _HAS_SCIPY:
        raise ImportError("gaussian-affinity kernel requires sklearn for distances")
    X = _np(coords)
    n = X.shape[0]
    p = param or {}
    perp = float(p.get("perplexity", 30.0))
    gamma = float(p.get("gamma", 1.0))
    sq = (X**2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    Pcond = _perplexity_probabilities(D2, perp)
    P = (Pcond + Pcond.T) / (2.0 * n)
    G = (P / (P.max() + 1e-12)) ** gamma
    weights = default_weights(n, device) if weights is None else weights
    return double_center(_as_tensor(G, device), weights.to(device), device)


def compute_fuzzy_topological_kernel_torch(
    coords: np.ndarray | torch.Tensor,
    param: dict[str, Any] | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """UMAP fuzzy-topological input kernel.

    Local distances are offset by rho_i (distance to nearest neighbour) and scaled
    by sigma_i (binary-searched so sum_j w_ij = log2(k)). The fuzzy set union gives
        G = W + W^T - W o W^T.

    param: dict {'k': int (default 15), 'gamma': float (default 1.0)}.
           gamma softens the affinity via G = (G / max G)^gamma (§5.3.2); gamma=1.0
           leaves G unchanged (up to the RV-invariant scale).
    """
    if not _HAS_SCIPY:
        raise ImportError("fuzzy-topological kernel requires sklearn")
    X = _np(coords)
    n = X.shape[0]
    p = param or {}
    k = int(p.get("k", 15))
    gamma = float(p.get("gamma", 1.0))
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nbrs.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    rho = dist[:, 0]  # nearest-neighbour distance
    target = np.log2(k)
    W = np.zeros((n, n))
    for i in range(n):
        d = np.maximum(dist[i] - rho[i], 0.0)
        sigma, lo, hi = 1.0, 0.0, np.inf
        for _ in range(64):  # binary search on sigma_i
            psum = np.exp(-d / sigma).sum()
            if abs(psum - target) < 1e-5:
                break
            if psum > target:
                hi = sigma
                sigma = (lo + hi) / 2
            else:
                lo = sigma
                sigma = sigma * 2 if hi == np.inf else (lo + hi) / 2
        W[i, idx[i]] = np.exp(-d / sigma)
    G = W + W.T - W * W.T  # probabilistic (fuzzy) union
    G = (G / (G.max() + 1e-12)) ** gamma
    weights = default_weights(n, device) if weights is None else weights
    return double_center(_as_tensor(G, device), weights.to(device), device)


# ---------------------------------------------------------------------------
# Cached input-affinity helpers (for hyperparameter sweeps)
# ---------------------------------------------------------------------------
# The expensive part of the t-SNE / UMAP input kernels (the perplexity binary
# search, the fuzzy-set construction) depends ONLY on perplexity / n_neighbors,
# NOT on the softening exponent gamma. These return the raw symmetric affinity so
# a sweep can compute it once per neighbour hyperparameter and re-apply gamma +
# double-centering cheaply for every gamma via `soften_and_center`.
def gaussian_affinity_base(
    coords: np.ndarray | torch.Tensor, perplexity: float = 30.0
) -> np.ndarray:
    """Symmetric t-SNE affinity P (pre-softening, pre-centering)."""
    if not _HAS_SCIPY:
        raise ImportError("gaussian-affinity kernel requires sklearn for distances")
    X = _np(coords)
    n = X.shape[0]
    sq = (X**2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    Pcond = _perplexity_probabilities(D2, float(perplexity))
    return (Pcond + Pcond.T) / (2.0 * n)


def fuzzy_topological_base(
    coords: np.ndarray | torch.Tensor, k: int = 15
) -> np.ndarray:
    """UMAP fuzzy-topological affinity G = W + W^T - W o W^T (pre-softening, pre-centering)."""
    if not _HAS_SCIPY:
        raise ImportError("fuzzy-topological kernel requires sklearn")
    X = _np(coords)
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=int(k) + 1).fit(X)
    dist, idx = nbrs.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    rho = dist[:, 0]
    target = np.log2(int(k))
    W = np.zeros((n, n))
    for i in range(n):
        d = np.maximum(dist[i] - rho[i], 0.0)
        sigma, lo, hi = 1.0, 0.0, np.inf
        for _ in range(64):
            psum = np.exp(-d / sigma).sum()
            if abs(psum - target) < 1e-5:
                break
            if psum > target:
                hi = sigma
                sigma = (lo + hi) / 2
            else:
                lo = sigma
                sigma = sigma * 2 if hi == np.inf else (lo + hi) / 2
        W[i, idx[i]] = np.exp(-d / sigma)
    return W + W.T - W * W.T


def soften_and_center(
    base: np.ndarray,
    gamma: float,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Apply the softening G = (base / max base)^gamma and double-center to K_n.
    `base` is a raw affinity from gaussian_affinity_base / fuzzy_topological_base."""
    n = base.shape[0]
    weights = default_weights(n, device) if weights is None else weights.to(device)
    G = (base / (base.max() + 1e-12)) ** float(gamma)
    return double_center(_as_tensor(G, device), weights, device)


def compute_rbf_kernel_torch(
    coords: np.ndarray | torch.Tensor,
    param: float | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Mercer RBF input kernel: G_ij = exp(-gamma ||x_i - x_j||^2).  Recovers
    Kernel PCA (Gaussian kernel) with a linear output. Pure torch (also usable
    as a differentiable output kernel if desired).

    param: gamma (float). Default uses the median-distance heuristic.
    """
    coords = _as_tensor(coords, device)
    n = coords.shape[0]
    weights = default_weights(n, device) if weights is None else weights.to(device)
    d2 = pairwise_sq_dists(coords)
    gamma: float
    if param is None:
        med = (
            torch.median(d2[d2 > 0])
            if (d2 > 0).any()
            else torch.tensor(1.0, device=device)
        )
        gamma = float(1.0 / (med + 1e-12))
    else:
        gamma = float(param)
    G = torch.exp(-gamma * d2)
    return double_center(G, weights, device)


def compute_class_kernel_torch(
    coords: np.ndarray | torch.Tensor,
    param: dict[str, Any] | None = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Inter-class (label) kernel K_Z = P_Z K_X P_Z^T, where
        P_Z = Z (Z^T Pi Z)^{-1} Z^T Pi
    relocates every point onto its (weighted) class centroid, so all within-class
    variance is removed and only the between-class (centroid) structure remains.

    NOTE: this is NOT LDA. With a linear output, RV-maximising against K_Z drives
    every class to a single point (zero within-class spread), whereas LDA keeps
    within-class spread along the discriminant axes. K_Z is the fully-supervised
    (between-class only) limit; use it as the alpha=1 endpoint of a supervised
    interpolation, not as an LDA surrogate.

    param: dict {'labels': array-like (n,), 'base_kernel': (n,n) tensor or None}.
           If base_kernel is None, the linear kernel of `coords` is used as K_X.
    """
    coords = _as_tensor(coords, device)
    n = coords.shape[0]
    weights = default_weights(n, device) if weights is None else weights.to(device)
    p = param or {}
    labels = np.asarray(p["labels"])
    classes = np.unique(labels)
    Z = torch.zeros((n, len(classes)), device=device)
    for c_idx, c in enumerate(classes):
        Z[torch.as_tensor(labels == c, device=device), c_idx] = 1.0
    Pi = torch.diag(weights)
    ZtPiZ = Z.T @ Pi @ Z
    P_Z = Z @ torch.linalg.inv(ZtPiZ) @ Z.T @ Pi
    K_X = p.get("base_kernel", None)
    if K_X is None:
        K_X = compute_linear_kernel_torch(coords, weights=weights, device=device)
    return P_Z @ K_X @ P_Z.T


# ===========================================================================
# Registry  (pick bricks by name)
# ===========================================================================
INPUT_KERNELS: dict[str, Callable[..., torch.Tensor]] = {
    "linear": compute_linear_kernel_torch,
    "geodesic": compute_geodesic_kernel_torch,
    "lle": compute_lle_kernel_torch,
    "laplacian": compute_laplacian_kernel_torch,
    "diffusion": compute_diffusion_kernel_torch,
    "gaussian_affinity": compute_gaussian_affinity_kernel_torch,
    "fuzzy_topological": compute_fuzzy_topological_kernel_torch,
    "rbf": compute_rbf_kernel_torch,
    "class": compute_class_kernel_torch,
}

OUTPUT_KERNELS: dict[str, Callable[..., torch.Tensor]] = {
    "linear": compute_linear_kernel_torch,
    "polynomial": compute_polynomial_kernel_torch,
    "gaussian": compute_gaussian_kernel_torch,
    "student_t": compute_student_t_kernel_torch,
    "umap": compute_umap_kernel_torch,
}


# ===========================================================================
# RV coefficient and gradient-ascent driver
# ===========================================================================
def rv_coefficient(
    K1: torch.Tensor, K2: torch.Tensor, eps: float = 1e-12, hollow: bool = False
) -> torch.Tensor:
    """RV(K1, K2) = <K1,K2> / (||K1|| ||K2||), using <A,B> = sum(A*B) for symmetric A,B.

    If ``hollow`` is True, both diagonals are zeroed before the cosine (the *hollow-RV*
    objective). The alignment then ignores the self-terms K_ii, which on the output
    side act as a spread-suppressing 'global tether' (sum_i K_ii^2 is a near-constant
    floor in ||K_Y||); removing them frees the embedding to spread, recovering the
    neighbour-embedding regime (cf. scripts/tests/test_diag_energy.py)."""
    if hollow:
        off = 1.0 - torch.eye(K1.shape[0], device=K1.device, dtype=K1.dtype)
        K1 = K1 * off
        K2 = K2 * off
    num = (K1 * K2).sum()
    den = torch.sqrt((K1 * K1).sum() * (K2 * K2).sum()) + eps
    return num / den


# ===========================================================================
# Spectral (closed-form) solvers  —  exact RV optima on the linear cone
# ---------------------------------------------------------------------------
# Theorem 2: on the cone S_d = {K >= 0, rank(K) <= d}, maximising the RV cosine
# is a Frobenius projection, solved by truncating K_X to its top-d eigenpairs
# (Eckart-Young / classical MDS). No iteration: eigh gives the global optimum,
# which reaches the alignment ceiling RV_max(d) of Prop. 3 exactly. The solvers are
# modular in the *input* kernel: feed any centred K_X = Q G_X^kappa Q^T built by the
# INPUT_KERNELS bricks, and the right spectral method follows. Two readouts differ
# only in the axis scaling: `linear` uses the sqrt(lambda) (PCA/MDS) scaling;
# `orthonormal` uses balanced unit axes (the native readout of methods defined as
# orthonormal eigenvectors, e.g. Laplacian Eigenmaps, LLE).
# ===========================================================================
def _truncated_eig(K_X: torch.Tensor, q: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-q eigenpairs of a symmetric kernel, eigenvalues sorted descending.

    The eigendecomposition runs on CPU (a single dense op; MPS has no eigh), then
    the result is moved back to the kernel's device.
    """
    lam, U = torch.linalg.eigh(K_X.cpu())  # ascending order; eigh unsupported on MPS
    lam, U = lam.flip(0)[:q], U.flip(1)[:, :q]
    return lam.to(K_X.device), U.to(K_X.device)


def rv_ceiling(K_X: torch.Tensor, q: int = 2) -> float:
    """Alignment ceiling RV_max(q) of Prop. 1 (the Frobenius 'explained variance'):
    RV_max(q) = sqrt( sum_{j<=q} (lambda_j^+)^2 / ||K_X||_F^2 ), computed from the
    top-q eigenvalues of K_X (clipped at 0) and the full Frobenius norm — no full
    eigendecomposition needed. No output kernel with a linear readout can exceed it,
    and the clipped truncation of Theorem 2 attains it exactly."""
    lam, _ = _truncated_eig(K_X, q)
    num = (lam.clamp_min(0.0) ** 2).sum()
    den = (K_X * K_X).sum()
    return float(torch.sqrt(num / (den + 1e-12)))


def spectral_embed_linear(
    K_X: torch.Tensor,
    q: int = 2,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, float]:
    """Closed-form RV optimum for a LINEAR output kernel (Theorem 2 / classical MDS).

    Truncates K_X to its top-q positive eigenpairs and reads coordinates with the
    sqrt(lambda) axis scaling Y = Pi^{-1/2} U_q Lambda_q^{1/2} (anisotropic, like
    PCA / cMDS / Isomap / Kernel PCA). Negative eigenvalues are clipped (PSD
    projection). Returns (Y, rv) where rv = RV_max(q) (Prop. 3).
    """
    n = K_X.shape[0]
    weights = default_weights(n, device) if weights is None else weights.to(device)
    lam, U = _truncated_eig(K_X, q)
    lam = lam.clamp_min(0.0)  # PSD projection: clip negative eigenvalues
    Y = torch.diag(1.0 / torch.sqrt(weights)) @ U @ torch.diag(torch.sqrt(lam))
    K_Y = double_center(Y @ Y.T, weights, device)
    return Y.detach(), float(rv_coefficient(K_X, K_Y))


def spectral_embed_orthonormal(
    K_X: torch.Tensor,
    q: int = 2,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, float]:
    """Closed-form RV optimum with ORTHONORMAL (balanced) axes Y = Pi^{-1/2} U_q --
    no sqrt(lambda) scaling. The induced output kernel is the rank-q projector
    U_q U_q^T. This is the native readout for spectral methods defined as orthonormal
    eigenvectors (Laplacian Eigenmaps, LLE), where spectral_embed_linear's sqrt(lambda)
    scaling would stretch the axes by sqrt(lambda_j) and diverge from the reference.
    Returns (Y, rv).
    """
    n = K_X.shape[0]
    weights = default_weights(n, device) if weights is None else weights.to(device)
    _, U = _truncated_eig(K_X, q)
    Y = torch.diag(1.0 / torch.sqrt(weights)) @ U
    K_Y = U @ U.T  # projector onto the top-q eigenspace (Q Y = U, orthonormal)
    return Y.detach(), float(rv_coefficient(K_X, K_Y))


def rv_dimred(
    K_X: torch.Tensor,
    output_kernel: str | Callable[..., torch.Tensor] = "student_t",
    q: int = 2,
    weights: torch.Tensor | None = None,
    output_param: Any = None,
    n_iter: int = 500,
    lr: float = 1e-1,
    device: str | torch.device = "cpu",
    init: np.ndarray | torch.Tensor | None = None,
    verbose: bool = False,
    hollow: bool = False,
) -> tuple[torch.Tensor, float]:
    """Maximise RV(K_X, K_Y(Y)) over the embedding Y by autograd gradient ascent.

    K_X            : (n,n) fixed input kernel (any INPUT_KERNELS output).
    output_kernel  : name in OUTPUT_KERNELS, or a callable with the kernel signature.
    hollow         : if True, optimise the hollow-RV (both diagonals zeroed) — the
                     neighbour-embedding objective that frees the spread.
    Returns (Y, final_rv).
    """
    n = K_X.shape[0]
    weights = default_weights(n, device) if weights is None else weights.to(device)
    out_fn: Callable[..., torch.Tensor] = (
        OUTPUT_KERNELS[output_kernel]
        if isinstance(output_kernel, str)
        else output_kernel
    )
    Y = (
        (torch.randn(n, q, device=device) * 1e-2)
        if init is None
        else _as_tensor(init, device).clone()
    )
    Y.requires_grad_(True)
    opt = torch.optim.Adam([Y], lr=lr)
    rv: torch.Tensor = torch.tensor(0.0)
    for it in range(n_iter):
        opt.zero_grad()
        K_Y = out_fn(Y, param=output_param, weights=weights, device=device)
        rv = rv_coefficient(K_X, K_Y, hollow=hollow)
        (-rv).backward()  # maximise RV
        opt.step()
        if verbose and (it % max(1, n_iter // 10) == 0):
            print(f"  iter {it:4d}  RV = {rv.item():.4f}")
    return Y.detach(), rv.item()
