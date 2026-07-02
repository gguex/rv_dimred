"""
temp_log_volume_scan.py
=======================
Scan fin sur les petites valeurs de lambda pour l'objectif log(Z).
Perplexité fixe à 30.
Kernel d'entrée : compute_gaussian_affinity_kernel_torch (celui de src/rv_kernels.py)
avec MISE A ZERO DE LA DIAGONALE du kernel K_X (et K_Y) pour le calcul du RV.
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
from src.rv_kernels import (compute_gaussian_affinity_kernel_torch, default_weights, double_center)

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

def rv_hollow(K1: torch.Tensor, K2: torch.Tensor) -> torch.Tensor:
    """RV avec la diagonale mise à 0 explicitement."""
    # On crée des copies pour ne pas modifier les tenseurs originaux
    K1_h = K1.clone()
    K2_h = K2.clone()
    K1_h.fill_diagonal_(0.0)
    K2_h.fill_diagonal_(0.0)
    
    num = (K1_h * K2_h).sum()
    den = torch.sqrt((K1_h * K1_h).sum() * (K2_h * K2_h).sum()) + 1e-12
    return num / den

def metrics(Y, K_X, w, labels):
    Yt = to_tensor(Y, DEV)
    K_Y = double_center(gram_student(Yt), w, DEV)
    rv = float(rv_hollow(K_X, K_Y))
    km = KMeans(n_clusters=len(np.unique(labels)), n_init=10, random_state=SEED).fit_predict(Y)
    ari = float(adjusted_rand_score(labels, km))
    spread = float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))
    return rv, ari, spread

def optimize(K_X, w, init, lam: float):
    Y = torch.tensor(init, dtype=torch.float32, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([Y], lr=LR)
    for _ in range(N_ITER):
        opt.zero_grad()
        G_Y = gram_student(Y)
        K_Y = double_center(G_Y, w, DEV)
        
        # PULL avec diagonale à 0
        pull = rv_hollow(K_X, K_Y)
        
        loss = -(pull - lam * log_volume(G_Y))
        loss.backward()
        opt.step()
    return Y.detach().cpu().numpy()

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_mnist(n_per_digit=N_PER_DIGIT, random_state=SEED)
    n, labels = ds.n, ds.labels
    w = default_weights(n, DEV)
    init = pca_init(ds.X)
    X_t = to_tensor(ds.X, DEV)
    
    # K_X via rv_kernels (t-SNE input kernel avec perplexité)
    K_X_raw = compute_gaussian_affinity_kernel_torch(
        X_t, param={"perplexity": PERP, "gamma": SOFTENING}, weights=w, device=DEV
    )
    # Note: G dans gaussian_affinity a DÉJÀ 0 sur la diagonale. 
    # Le centrage (Q G Q^T) remplit la diagonale. On met à 0 la diagonale de K_X_raw
    # pour s'assurer que même après ou avant centrage, on joue sur du "hollow".
    K_X = normalize_kernel(K_X_raw)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    print("Scan Lambda (log volume) | K_X = gaussian_affinity (rv_kernels) + diag=0")
    print(f"{'Lambda':<8} | {'RV_0':>7} | {'ARI':>7} | {'spread':>8}")
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

    fig.suptitle("Scan Lambda log(Z) avec K_X adaptatif et RV sans diagonale", fontsize=16)
    fig.tight_layout()
    out = FIG_DIR / "temp_log_volume_scan_Kdiag0.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved figure to {out}")

if __name__ == "__main__":
    main()
