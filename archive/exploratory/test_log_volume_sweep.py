"""
test_log_volume_sweep.py  —  Vérification de lambda sur log(Z)
==============================================================
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from src.benchmark_common import (PERPLEXITY, SEED, SOFTENING, normalize_kernel, pca_init, to_tensor)
from src.datasets import load_mnist
from src.rv_kernels import (compute_gaussian_affinity_kernel_torch, default_weights, double_center, rv_coefficient)

N_PER_DIGIT = 50
D = 2
N_ITER = 800
LR = 0.1
DEV = "cpu"

def gram_student(Y: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(Y, Y) ** 2
    return 1.0 / (1.0 + d2)

def log_volume(G: torch.Tensor) -> torch.Tensor:
    Z = G.sum() - G.diagonal().sum()
    return torch.log(Z.clamp_min(1e-9))

def optimize(K_X, w, init, lam: float = 1.0) -> np.ndarray:
    Y = torch.tensor(init, dtype=torch.float32, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([Y], lr=LR)
    for _ in range(N_ITER):
        opt.zero_grad()
        G = gram_student(Y)
        K_Y = double_center(G, w, DEV)
        K_Y_frob = (K_Y * K_Y).sum().sqrt().clamp_min(1e-10)
        pull = (K_X * K_Y).sum() / K_Y_frob
        loss = -(pull - lam * log_volume(G))
        loss.backward()
        opt.step()
    return Y.detach().cpu().numpy()

def metrics(Y, K_X, w, labels):
    Yt = to_tensor(Y, DEV)
    K_Y = double_center(gram_student(Yt), w, DEV)
    rv  = float(rv_coefficient(K_X, K_Y))
    km  = KMeans(n_clusters=len(np.unique(labels)), n_init=10, random_state=SEED).fit_predict(Y)
    ari = float(adjusted_rand_score(labels, km))
    spread = float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))
    return rv, ari, spread

def main():
    ds = load_mnist(n_per_digit=N_PER_DIGIT, random_state=SEED)
    n, labels = ds.n, ds.labels
    w  = default_weights(n, DEV)
    Xt = to_tensor(ds.X, DEV)
    K_X = compute_gaussian_affinity_kernel_torch(
        Xt, param={"perplexity": PERPLEXITY, "gamma": SOFTENING}, weights=w, device=DEV
    )
    K_X = normalize_kernel(K_X)
    init = pca_init(ds.X)

    print("Test log_volume sweep")
    print(f"{'lambda':<10} | {'RV':>7} | {'ARI':>7} | {'spread':>8}")
    print("-" * 40)
    for lam in [1.0, 0.1, 0.01, 0.001, 0.0001, 1e-5]:
        Y = optimize(K_X, w, init, lam=lam)
        rv, ari, spread = metrics(Y, K_X, w, labels)
        print(f"{lam:<10} | {rv:>7.4f} | {ari:>7.4f} | {spread:>8.3f}")

if __name__ == "__main__":
    main()
