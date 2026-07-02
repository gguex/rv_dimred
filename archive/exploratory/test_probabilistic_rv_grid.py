"""
test_probabilistic_rv_grid.py  —  TEST E (suite)
=================================================

Grille croisée (Perplexité x Lambda) pour explorer l'interaction entre
la topologie locale (PULL) et la force de répulsion globale (PUSH).

Objectif (maximiser sur Y) :
    L(Y, lambda) = RV(K_X, K_Y)  -  lambda * <G_neg, G_Y>
où
    RV(K_X, K_Y) = <K_X, K_Y> / ||K_Y||   (PULL normalisé)
    G_neg = 1 - G_X                       (G_X : affinités brutes dans [0,1])
    G_Y = 1 / (1 + D^2(Y))                (Gram Student-t brut)

On génère un graphique (lignes = perplexités, colonnes = lambdas + t-SNE ref).
"""

from __future__ import annotations

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from src.benchmark_common import (
    SEED,
    SOFTENING,
    normalize_kernel,
    pca_init,
    to_tensor,
)
from src.datasets import load_mnist
from src.rv_kernels import (
    compute_gaussian_affinity_kernel_torch,
    _perplexity_probabilities,
    default_weights,
    double_center,
    rv_coefficient,
)

N_PER_DIGIT = 50          # reduced MNIST: n = 500
D = 2
N_ITER = 800             # primal gradient steps
LR = 0.1                 # Adam lr
FIG_DIR = Path("results/figures/temp")
DEV = "cpu"

PERPLEXITIES = [10, 30, 100]
LAMBDAS = [0.0, 1.0, 10.0]

# ── helpers ───────────────────────────────────────────────────────────────────

def gram_student(Y: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(Y, Y) ** 2
    return 1.0 / (1.0 + d2)

def get_raw_gaussian_affinity(X: np.ndarray, perplexity: float, gamma: float) -> torch.Tensor:
    n = X.shape[0]
    sq = (X**2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    Pcond = _perplexity_probabilities(D2, perplexity)
    P = (Pcond + Pcond.T) / (2.0 * n)
    G = (P / (P.max() + 1e-12)) ** gamma
    return torch.tensor(G, dtype=torch.float32, device=DEV)

# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(Y: np.ndarray, K_X: torch.Tensor, w: torch.Tensor,
            labels: np.ndarray) -> tuple[float, float, float]:
    Yt = to_tensor(Y, DEV)
    K_Y = double_center(gram_student(Yt), w, DEV)
    rv = float(rv_coefficient(K_X, K_Y))
    km = KMeans(n_clusters=len(np.unique(labels)), n_init=10,
                random_state=SEED).fit_predict(Y)
    ari = float(adjusted_rand_score(labels, km))
    spread = float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))
    return rv, ari, spread


# ── optimizer ─────────────────────────────────────────────────────────────────

def optimize(
    K_X: torch.Tensor, G_neg: torch.Tensor, w: torch.Tensor, init: np.ndarray,
    lam: float = 0.0,
) -> np.ndarray:
    Y = torch.tensor(init, dtype=torch.float32, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([Y], lr=LR)
    n = G_neg.shape[0]
    
    for it in range(N_ITER):
        opt.zero_grad()
        G_Y = gram_student(Y)
        K_Y = double_center(G_Y, w, DEV)
        K_Y_frob = (K_Y * K_Y).sum().sqrt().clamp_min(1e-10)
        align = (K_X * K_Y).sum() / K_Y_frob
        
        repulsion = ((G_neg * G_Y).sum() - (torch.diag(G_neg)*torch.diag(G_Y)).sum()) / (n * (n - 1))
        
        loss = -(align - lam * repulsion)
        loss.backward()
        opt.step()
        
    return Y.detach().cpu().numpy()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Test E (Grid) - RV probabiliste, MNIST n=500\n")

    ds = load_mnist(n_per_digit=N_PER_DIGIT, random_state=SEED)
    n, labels = ds.n, ds.labels
    w = default_weights(n, DEV)
    init = pca_init(ds.X)
    X_t = to_tensor(ds.X, DEV)

    fig, axes = plt.subplots(len(PERPLEXITIES), len(LAMBDAS) + 1, 
                             figsize=(4 * (len(LAMBDAS) + 1), 4 * len(PERPLEXITIES)))
    
    print(f"{'Perp':<6} | {'Lambda':<6} | {'RV':>7} | {'ARI':>7} | {'spread':>8}")
    print("-" * 43)

    for row, perp in enumerate(PERPLEXITIES):
        # 1. K_X (centré) pour le PULL (dépend de la perplexité)
        K_X = compute_gaussian_affinity_kernel_torch(
            X_t, param={"perplexity": perp, "gamma": SOFTENING}, weights=w, device=DEV
        )
        K_X = normalize_kernel(K_X)
        
        # 2. G_neg (brut) pour le PUSH (dépend de la perplexité)
        G_X = get_raw_gaussian_affinity(ds.X, perp, SOFTENING)
        G_neg = 1.0 - G_X
        
        for col, lam in enumerate(LAMBDAS):
            Y_res = optimize(K_X, G_neg, w, init, lam=lam)
            rv, ari, sp = metrics(Y_res, K_X, w, labels)
            
            print(f"{perp:<6} | {lam:<6.1f} | {rv:>7.4f} | {ari:>7.4f} | {sp:>8.3f}")
            
            ax = axes[row, col]
            ax.scatter(Y_res[:, 0], Y_res[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
            ax.set_title(f"Perp={perp}, Lam={lam}\nARI={ari:.3f} SP={sp:.1f}")
            ax.set_xticks([]); ax.set_yticks([])

        # Calcul t-SNE ref pour cette perplexité
        Yref = TSNE(n_components=D, perplexity=perp, random_state=SEED).fit_transform(ds.X)
        rvr, arir, spr = metrics(Yref, K_X, w, labels)
        print(f"{perp:<6} | {'t-SNE':<6} | {rvr:>7.4f} | {arir:>7.4f} | {spr:>8.3f}")
        print("-" * 43)
        
        ax_ref = axes[row, -1]
        ax_ref.scatter(Yref[:, 0], Yref[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
        ax_ref.set_title(f"t-SNE (Perp={perp})\nARI={arir:.3f} SP={spr:.1f}")
        ax_ref.set_xticks([]); ax_ref.set_yticks([])

    fig.suptitle("RV Probabiliste vs t-SNE : Effet croisé Perplexité / Lambda", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = FIG_DIR / "test_E_grid.png"
    fig.savefig(out, dpi=130)
    print(f"\nScatter grid saved -> {out}")


if __name__ == "__main__":
    main()
