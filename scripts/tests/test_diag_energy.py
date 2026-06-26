"""
test_diag_energy.py  —  Vérification du mécanisme métrique full-RV vs hollow-RV
==============================================================================

But : valider numériquement l'argument théorique de I.5 (plan), point 3.

Fait exact (poids uniformes) : pour un noyau CENTRÉ, K_ii = -r_i où
r_i = sum_{j != i} K_ij (degré / somme de ligne hors-diagonale). Donc l'énergie
diagonale ||diag(K_Y)||^2 = sum_i K_ii^2 = sum_i r_i^2 est EXACTEMENT le terme
que la métrique M = I + D*D du RV plein ajoute (sur-pondération des degrés).

Prédiction : maximiser le RV PLEIN gonfle sum_i r_i^2 (collapse / cœurs denses),
tandis que le RV HOLLOW (diagonales à 0) la garde basse et produit un étalement
proche de celui du t-SNE.

On suit, le long de l'optimisation, pour chaque objectif :
    - spread (rayon RMS)
    - frac_diag = ||diag(K_Y)||^2 / ||K_Y||^2   (fraction d'énergie diagonale)
    - sum_r2   = sum_i r_i^2                     (énergie de degré, absolue)
et on compare au point t-SNE sklearn.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from src.benchmark_common import (SEED, SOFTENING, normalize_kernel, pca_init, to_tensor)
from src.datasets import load_mnist
from src.rv_kernels import (compute_gaussian_affinity_kernel_torch, default_weights,
                            double_center, rv_coefficient)

N_PER_DIGIT = 50          # MNIST réduit : n = 500
D = 2
N_ITER = 800
LR = 0.1
DEV = "cpu"
PERP = 30
CHECKPOINTS = [0, 50, 100, 200, 400, 800]


def gram_student(Y: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(Y, Y) ** 2
    return 1.0 / (1.0 + d2)


def rv_hollow(K1: torch.Tensor, K2: torch.Tensor) -> torch.Tensor:
    K1h = K1.clone(); K2h = K2.clone()
    K1h.fill_diagonal_(0.0); K2h.fill_diagonal_(0.0)
    num = (K1h * K2h).sum()
    den = torch.sqrt((K1h * K1h).sum() * (K2h * K2h).sum()) + 1e-12
    return num / den


def diag_stats(K_Y: torch.Tensor) -> tuple[float, float]:
    """frac_diag = ||diag||^2/||K||^2 ; sum_r2 = sum_i K_ii^2 (= sum_i r_i^2)."""
    diag = K_Y.diagonal()
    e_diag = float((diag * diag).sum())
    e_tot = float((K_Y * K_Y).sum())
    return e_diag / e_tot, e_diag


def eval_Y(Y_np: np.ndarray, K_X: torch.Tensor, w: torch.Tensor,
           labels: np.ndarray) -> dict:
    Yt = to_tensor(Y_np, DEV)
    K_Y = double_center(gram_student(Yt), w, DEV)
    frac, sum_r2 = diag_stats(K_Y)
    km = KMeans(n_clusters=len(np.unique(labels)), n_init=10,
                random_state=SEED).fit_predict(Y_np)
    return {
        "rv_full": float(rv_coefficient(K_X, K_Y)),
        "rv_hollow": float(rv_hollow(K_X, K_Y)),
        "ari": float(adjusted_rand_score(labels, km)),
        "spread": float(np.sqrt(((Y_np - Y_np.mean(0)) ** 2).sum(1).mean())),
        "frac_diag": frac,
        "sum_r2": sum_r2,
    }


def optimize(K_X: torch.Tensor, w: torch.Tensor, init: np.ndarray, hollow: bool):
    Y = torch.tensor(init, dtype=torch.float32, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([Y], lr=LR)
    traj = {}
    for it in range(N_ITER + 1):
        if it in CHECKPOINTS:
            traj[it] = Y.detach().cpu().numpy().copy()
        if it == N_ITER:
            break
        opt.zero_grad()
        K_Y = double_center(gram_student(Y), w, DEV)
        if hollow:
            pull = rv_hollow(K_X, K_Y)
        else:
            K_Y_frob = (K_Y * K_Y).sum().sqrt().clamp_min(1e-10)
            pull = (K_X * K_Y).sum() / K_Y_frob   # ||K_X|| = 1 (normalisé)
        (-pull).backward()
        opt.step()
    return traj


def main() -> None:
    print("test_diag_energy : full-RV vs hollow-RV, MNIST n=500, perp=30\n")
    ds = load_mnist(n_per_digit=N_PER_DIGIT, random_state=SEED)
    labels = ds.labels
    w = default_weights(ds.n, DEV)
    init = pca_init(ds.X)
    X_t = to_tensor(ds.X, DEV)

    K_X = compute_gaussian_affinity_kernel_torch(
        X_t, param={"perplexity": PERP, "gamma": SOFTENING}, weights=w, device=DEV)
    K_X = normalize_kernel(K_X)

    traj_full = optimize(K_X, w, init, hollow=False)
    traj_hollow = optimize(K_X, w, init, hollow=True)

    hdr = f"{'iter':>5} | {'spread':>7} | {'frac_diag':>9} | {'sum_r2':>10} | {'ARI':>6}"
    for name, traj in (("FULL-RV", traj_full), ("HOLLOW-RV", traj_hollow)):
        print(f"--- {name} : trajectoire ---")
        print(hdr)
        print("-" * len(hdr))
        for it in CHECKPOINTS:
            m = eval_Y(traj[it], K_X, w, labels)
            print(f"{it:>5} | {m['spread']:>7.3f} | {m['frac_diag']:>9.4f} | "
                  f"{m['sum_r2']:>10.4f} | {m['ari']:>6.3f}")
        print()

    # Référence t-SNE
    print("Fitting t-SNE reference...")
    Yref = TSNE(n_components=D, perplexity=PERP, random_state=SEED).fit_transform(ds.X)

    print("\n=== Comparaison finale (iter=800) ===")
    cols = (f"{'config':<14} | {'spread':>7} | {'frac_diag':>9} | {'sum_r2':>10} | "
            f"{'ARI':>6} | {'RV_full':>7} | {'RV_hol':>7}")
    print(cols)
    print("-" * len(cols))
    for name, Y in (("full-RV", traj_full[N_ITER]),
                    ("hollow-RV", traj_hollow[N_ITER]),
                    ("t-SNE sklearn", Yref)):
        m = eval_Y(Y, K_X, w, labels)
        print(f"{name:<14} | {m['spread']:>7.3f} | {m['frac_diag']:>9.4f} | "
              f"{m['sum_r2']:>10.4f} | {m['ari']:>6.3f} | {m['rv_full']:>7.4f} | "
              f"{m['rv_hollow']:>7.4f}")


if __name__ == "__main__":
    main()
