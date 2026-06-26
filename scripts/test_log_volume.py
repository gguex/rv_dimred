"""
test_log_volume.py  —  Vérification numérique de push_pull_proposition.md §4
=============================================================================

Vérifie que le terme de volume g(Z) = log(Z), Z = sum_{i!=j} g_ij, suffit à
reproduire l'esthétique t-SNE à partir du RV normalisé (cosinus de Frobenius).

VERSION 2 : PULL = RV normalisé  (pas le produit HS brut).
Le PULL correct est le cosinus RV(K_X, K_Y) = <K_X, K_Y> / (||K_X|| ||K_Y||).
Puisque K_X est déjà normalisé (||K_X||=1 après normalize_kernel), il suffit
de diviser par ||K_Y|| dans l'optimizer.

Objectif (maximiser sur Y) :
    L(Y) = RV(K_X, K_Y)   -   g(Z)          (PULL normalisé + PUSH)

On compare quatre cas sur le même init PCA :
  (a) RV pur           : max RV(K_X, K_Y)              (PULL normalisé seul)
  (b) + vol. linéaire  : max RV(K_X, K_Y) - lam * Z    (g = id)
  (c) + log-vol        : max RV(K_X, K_Y) - log Z       (g = log, la cible)
  (d) référence t-SNE  : rv_embed Student-t  (= cas (a) avec plus d'itérations)

Même conditions : MNIST réduit n=500, Student-t, K_X affine adaptatif gamma=0.5.
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
from sklearn.metrics import adjusted_rand_score

from src.benchmark_common import (
    PERPLEXITY, SEED, SOFTENING,
    normalize_kernel, pca_init, rv_embed, to_tensor,
)
from src.datasets import load_mnist
from src.rv_kernels import (
    compute_gaussian_affinity_kernel_torch,
    default_weights, double_center, rv_coefficient,
)

N_PER_DIGIT = 50
D = 2
N_ITER = 800
LR = 0.1
LAM_LINEAR = 0.5       # fixed lambda for the linear-volume case (best from Test C)
FIG_DIR = Path("results/figures/temp")
DEV = "cpu"


def gram_student(Y: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(Y, Y) ** 2
    return 1.0 / (1.0 + d2)


def log_volume(G: torch.Tensor) -> torch.Tensor:
    """log Z, Z = sum of off-diagonal entries of raw Gram."""
    Z = G.sum() - G.diagonal().sum()
    return torch.log(Z.clamp_min(1e-9))


def lin_volume(G: torch.Tensor) -> torch.Tensor:
    return G.sum() - G.diagonal().sum()


def optimize(K_X, w, init, push: str = "none", lam: float = 1.0) -> np.ndarray:
    """push: 'none' | 'linear' | 'log'.

    PULL = RV normalisé : <K_X, K_Y> / ||K_Y||
    (||K_X||=1 car normalize_kernel a déjà été appliqué)
    """
    Y = torch.tensor(init, dtype=torch.float32, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([Y], lr=LR)
    for _ in range(N_ITER):
        opt.zero_grad()
        G = gram_student(Y)
        K_Y = double_center(G, w, DEV)
        K_Y_frob = (K_Y * K_Y).sum().sqrt().clamp_min(1e-10)
        pull = (K_X * K_Y).sum() / K_Y_frob   # RV normalisé (||K_X||=1)
        if push == "linear":
            loss = -(pull - lam * lin_volume(G))
        elif push == "log":
            loss = -(pull - log_volume(G))
        else:
            loss = -pull
        loss.backward()
        opt.step()
    return Y.detach().cpu().numpy()


def metrics(Y, K_X, w, labels):
    Yt = to_tensor(Y, DEV)
    K_Y = double_center(gram_student(Yt), w, DEV)
    rv  = float(rv_coefficient(K_X, K_Y))
    km  = KMeans(n_clusters=len(np.unique(labels)), n_init=10, random_state=SEED).fit_predict(Y)
    ari = float(adjusted_rand_score(labels, km))
    spread = float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean())  )
    Z = float((gram_student(Yt).sum() - gram_student(Yt).diagonal().sum()))
    lam_eff = 1.0 / Z   # effective lambda for log-volume = g'(Z) = 1/Z
    return rv, ari, spread, Z, lam_eff


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_mnist(n_per_digit=N_PER_DIGIT, random_state=SEED)
    n, labels = ds.n, ds.labels
    w  = default_weights(n, DEV)
    Xt = to_tensor(ds.X, DEV)
    K_X = compute_gaussian_affinity_kernel_torch(
        Xt, param={"perplexity": PERPLEXITY, "gamma": SOFTENING}, weights=w, device=DEV
    )
    K_X = normalize_kernel(K_X)
    init = pca_init(ds.X)

    print("Vérification : RV normalisé + log-volume  ~  t-SNE ?  (v2 : PULL = RV cosinus)")
    print(f"n={n}, Student-t, gamma={SOFTENING}, {N_ITER} steps, lr={LR}\n")

    cases = [
        ("(a) RV pur      (lambda=0)",     optimize(K_X, w, init, push="none")),
        (f"(b) + vol. lin. (lambda={LAM_LINEAR})",
         optimize(K_X, w, init, push="linear", lam=LAM_LINEAR)),
        ("(c) + log-vol   (g=log)",        optimize(K_X, w, init, push="log")),
    ]
    Yref, _ = rv_embed(K_X, init, DEV, "student_t", 1.0, weights=w)
    cases.append(("(d) référence rv_embed",  Yref))

    print(f"  {'cas':<34} {'RV':>6} {'ARI':>6} {'spread':>7} {'Z':>10} {'1/Z (=lam_log)':>15}")
    print("  " + "-" * 82)
    for label, Y in cases:
        rv, ari, sp, Z, lam_eff = metrics(Y, K_X, w, labels)
        print(f"  {label:<34} {rv:>6.4f} {ari:>6.4f} {sp:>7.3f} {Z:>10.1f} {lam_eff:>15.5f}")

    # figures : les 4 cas côte à côte
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (label, Y) in zip(axes, cases):
        ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        "PULL = <K_X, K_Y>  |  PUSH : rien / lam·Z / log Z / référence t-SNE\n"
        "(g=log donne la répulsion canonique t-SNE : nabla(-log Z) = 2 sum_j q_ij g_ij (y_i-y_j))"
    )
    fig.tight_layout()
    out = FIG_DIR / "test_log_volume.png"
    fig.savefig(out, dpi=130)
    print(f"\n  figure -> {out}")


if __name__ == "__main__":
    main()
