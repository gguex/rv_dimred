"""
temp_log_volume_scan_Kx_diag0.py
================================
Fine scan over small lambda values for the log(Z) objective.
Input kernel: compute_gaussian_affinity_kernel_torch (from src/rv_kernels.py),
with only the diagonal of K_X set to zero. K_Y retains its full diagonal, so
its ||K_Y||_F continues to penalize global expansion.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from src.benchmark_common import (SEED, SOFTENING, normalize_kernel, pca_init, to_tensor)
from src.datasets import load_mnist
from src.rv_kernels import (compute_gaussian_affinity_kernel_torch, default_weights, double_center, rv_coefficient)

N_PER_DIGIT = 50
D = 2
N_ITER = 800
LR = 0.1
DEV = "cpu"
FIG_DIR = Path("results/figures/temp")

PERP = 30
LAMBDAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def gram_student(Y: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(Y, Y) ** 2
    return 1.0 / (1.0 + d2)

def log_volume(G: torch.Tensor) -> torch.Tensor:
    Z = G.sum() - G.diagonal().sum()
    return torch.log(Z.clamp_min(1e-9))

def metrics(Y, K_X_hollow, w, labels):
    Yt = to_tensor(Y, DEV)
    K_Y = double_center(gram_student(Yt), w, DEV)
    # rv = <K_X_hollow, K_Y> / (||K_X_hollow|| * ||K_Y||)
    # K_X_hollow has 0 diagonal, K_Y has its full diagonal
    rv = float(rv_coefficient(K_X_hollow, K_Y))
    km = KMeans(n_clusters=len(np.unique(labels)), n_init=10, random_state=SEED).fit_predict(Y)
    ari = float(adjusted_rand_score(labels, km))
    spread = float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))
    return rv, ari, spread

def optimize(K_X_hollow, w, init, lam: float):
    Y = torch.tensor(init, dtype=torch.float32, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([Y], lr=LR)
    for _ in range(N_ITER):
        opt.zero_grad()
        G_Y = gram_student(Y)
        K_Y = double_center(G_Y, w, DEV)
        
        # PULL with a zero K_X diagonal but the usual ||K_Y||.
        K_Y_frob = (K_Y * K_Y).sum().sqrt().clamp_min(1e-10)
        pull = (K_X_hollow * K_Y).sum() / K_Y_frob
        
        loss = -(pull - lam * log_volume(G_Y))
        loss.backward()
        opt.step()
    return Y.detach().cpu().numpy()

from src.rv_kernels import _perplexity_probabilities

def compute_custom_input_kernel(coords: np.ndarray | torch.Tensor, perp: float, gamma: float, w: torch.Tensor):
    X = coords.detach().cpu().numpy()
    n = X.shape[0]
    sq = (X**2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    Pcond = _perplexity_probabilities(D2, perp)
    P = (Pcond + Pcond.T) / (2.0 * n)
    G = (P / (P.max() + 1e-12)) 
    G_tensor = torch.tensor(G, dtype=torch.float32, device=DEV)
    
    # Explicitly set the input Gram matrix diagonal to zero before double_center.
    G_tensor.fill_diagonal_(0.0)

    sums_gauss = torch.sum(G_tensor, axis=1)
    G_tensor = G_tensor / (sums_gauss[:, np.newaxis] + 1e-15)
    G_tensor = (G_tensor + G_tensor.T) / 2
    G_tensor = G_tensor ** gamma
    
    return double_center(G_tensor, w, DEV)

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_mnist(n_per_digit=N_PER_DIGIT, random_state=SEED)
    n, labels = ds.n, ds.labels
    w = default_weights(n, DEV)
    init = pca_init(ds.X)
    X_t = to_tensor(ds.X, DEV)
    
    # 1. Obtain K_X through the alternative kernel
    K_X_raw = compute_custom_input_kernel(X_t, PERP, SOFTENING, w)
    
    # 2. Normalize
    K_X = normalize_kernel(K_X_raw)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    print("Lambda scan | alternative kernel (raw G_X with zero diagonal)")
    print(f"{'Lambda':<8} | {'RV':>7} | {'ARI':>7} | {'spread':>8}")
    print("-" * 37)

    for i, lam in enumerate(LAMBDAS):
        Y = optimize(K_X, w, init, lam=lam)
        rv, ari, sp = metrics(Y, K_X, w, labels)
        print(f"{lam:<8.2f} | {rv:>7.4f} | {ari:>7.4f} | {sp:>8.3f}")
        
        ax = axes[i]
        ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
        ax.set_title(f"Lam={lam}\nARI={ari:.3f} SP={sp:.1f}")
        ax.set_xticks([]); ax.set_yticks([])

    # Reference t-SNE
    print("Fitting t-SNE reference...")
    Yref = TSNE(n_components=D, perplexity=PERP, random_state=SEED).fit_transform(ds.X)
    rv, ari, sp = metrics(Yref, K_X, w, labels)
    print(f"{'t-SNE':<8} | {rv:>7.4f} | {ari:>7.4f} | {sp:>8.3f}")
    
    ax = axes[7]
    ax.scatter(Yref[:, 0], Yref[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
    ax.set_title(f"t-SNE ref (Perp=30)\nARI={ari:.3f} SP={sp:.1f}")
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Lambda scan for log(Z) with an alternative input kernel (G_X diag=0)", fontsize=16)
    fig.tight_layout()
    out = FIG_DIR / "temp_log_volume_scan_Kx_diag0.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved figure to {out}")

if __name__ == "__main__":
    main()
