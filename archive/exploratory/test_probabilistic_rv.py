"""
test_probabilistic_rv.py  —  TEST E (exploration: RV probabiliste / Negative Sampling)
====================================================================================

Exploration de l'idée de "Noyau Répulsif" (Negative Sampling type UMAP) appliquée
au cadre RV (réf. idea_push_pull.md).

Au lieu d'une contrainte de volume globale, on pénalise explicitement la similarité
entre les paires qui sont des "non-voisins" dans l'espace d'entrée.

Objectif (maximiser sur Y) :
    L(Y, lambda) = RV(K_X, K_Y)  -  lambda * <G_neg, G_Y>
où
    RV(K_X, K_Y) = <K_X, K_Y> / ||K_Y||   (PULL normalisé)
    G_neg = 1 - G_X                       (G_X : affinités brutes dans [0,1])
    G_Y = 1 / (1 + D^2(Y))                (Gram Student-t brut)

On teste cette approche (a) sans répulsion, (b) avec un balayage de lambda,
et on compare à (c) t-SNE sklearn (référence).
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

from src.benchmark_common import (
    PERPLEXITY,
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


# ── helpers ───────────────────────────────────────────────────────────────────

def gram_student(Y: torch.Tensor) -> torch.Tensor:
    """Raw Student-t Gram 1/(1+D^2), diag = 1."""
    d2 = torch.cdist(Y, Y) ** 2
    return 1.0 / (1.0 + d2)

def get_raw_gaussian_affinity(X: np.ndarray, perplexity: float, gamma: float) -> torch.Tensor:
    """Calcule G_X brut (avant centrage), dans [0, 1]."""
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
    spread = float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))  # RMS radius
    return rv, ari, spread


# ── optimizer ─────────────────────────────────────────────────────────────────

def optimize(
    K_X: torch.Tensor, G_neg: torch.Tensor, w: torch.Tensor, init: np.ndarray,
    lam: float = 0.0,
) -> np.ndarray:
    """Maximize RV(K_X,K_Y) - lam * mean_offdiag(G_neg * G_Y) over Y by Adam."""
    Y = torch.tensor(init, dtype=torch.float32, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([Y], lr=LR)
    n = G_neg.shape[0]
    
    for it in range(N_ITER):
        opt.zero_grad()
        G_Y = gram_student(Y)
        K_Y = double_center(G_Y, w, DEV)
        K_Y_frob = (K_Y * K_Y).sum().sqrt().clamp_min(1e-10)
        align = (K_X * K_Y).sum() / K_Y_frob   # RV normalisé (||K_X||=1)
        
        # Repulsion cible les non-voisins :
        # On normalise par n*(n-1) pour avoir une métrique ~ moyenne hors-diagonale
        repulsion = ((G_neg * G_Y).sum() - (torch.diag(G_neg)*torch.diag(G_Y)).sum()) / (n * (n - 1))
        
        loss = -(align - lam * repulsion)
        loss.backward()
        opt.step()
        
    return Y.detach().cpu().numpy()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Test E  -  RV probabiliste (Negative Sampling), reduced MNIST n=500\n")

    ds = load_mnist(n_per_digit=N_PER_DIGIT, random_state=SEED)
    n, labels = ds.n, ds.labels
    w = default_weights(n, DEV)
    
    # 1. K_X (centré, normé) pour le PULL
    X_t = to_tensor(ds.X, DEV)
    K_X = compute_gaussian_affinity_kernel_torch(
        X_t, param={"perplexity": PERPLEXITY, "gamma": SOFTENING}, weights=w, device=DEV
    )
    K_X = normalize_kernel(K_X)
    
    # 2. G_neg (brut) pour le PUSH
    G_X = get_raw_gaussian_affinity(ds.X, PERPLEXITY, SOFTENING)
    G_neg = 1.0 - G_X
    
    init = pca_init(ds.X)

    # (a) pure alignment, lambda = 0
    Y0 = optimize(K_X, G_neg, w, init, lam=0.0)

    # (b) sweep
    sweep = {}
    for lam in (1.0, 5.0, 10.0, 20.0):
        Yl = optimize(K_X, G_neg, w, init, lam=lam)
        sweep[lam] = Yl

    # (c) reference: t-SNE sklearn
    print("  (calcul t-SNE sklearn...)", flush=True)
    Yref = TSNE(n_components=D, perplexity=PERPLEXITY,
                random_state=SEED).fit_transform(ds.X)

    # --- report ---
    print(f"  {'configuration':<28} {'RV':>7} {'ARI':>7} {'spread':>8}")
    print("  " + "-" * 54)
    rv0, ari0, sp0 = metrics(Y0, K_X, w, labels)
    print(f"  {'lambda=0 (PULL only)':<28} {rv0:>7.4f} {ari0:>7.4f} {sp0:>8.3f}")
    
    for lam, Yl in sweep.items():
        rv, ari, sp = metrics(Yl, K_X, w, labels)
        print(f"  {'fixed lambda=' + f'{lam:.1f}':<28} {rv:>7.4f} {ari:>7.4f} {sp:>8.3f}")
        
    rvr, arir, spr = metrics(Yref, K_X, w, labels)
    print(f"  {'reference t-SNE (sklearn)':<28} {rvr:>7.4f} {arir:>7.4f} {spr:>8.3f}")

    # --- figures ---
    # Choix de la meilleure config lambda pour le plot (arbitrairement lambda=10)
    Y_best = sweep[10.0]
    
    panels = [("lambda=0 (PULL only)", Y0), ("Negative Sampling (lam=10)", Y_best),
              ("t-SNE sklearn (ref)", Yref)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, Y) in zip(axes, panels):
        ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Test E: RV Probabiliste (Negative Sampling type UMAP)")
    fig.tight_layout()
    out = FIG_DIR / "test_E_probabilistic_rv.png"
    fig.savefig(out, dpi=130)
    print(f"\n  scatter saved -> {out}")


if __name__ == "__main__":
    main()
