"""
test_log_det.py  —  Piste 2 : régularisation spectrale log-det comme PUSH
=========================================================================

VERSION 2 : PULL = RV normalisé (cosinus de Frobenius), référence = t-SNE sklearn.

Objectif (maximiser sur Y) :
    L(Y) = RV(K_X, K_Y)  +  lambda * log det*(K_Y)

où det*(K_Y) est le pseudo-déterminant de K_Y (produit des n-1 valeurs propres
non nulles ; K_Y ∈ K_n => K_Y·1 = 0 => une vp structurellement nulle).

Mécanisme attendu :
  - RV pur => K_Y s'aligne sur K_X (bon résultat visuel grâce au dénominateur)
  - grad log det*(K_Y) = (K_Y)^+ : log-barrière spectrale dans K_n,
    empêche l'effondrement de rang du à l'alignement cosinus

Différence clé avec test_log_volume.py :
  - Ici le PUSH agit dans l'espace FORME K_n via le spectre de K_Y
  - Pas de mode volume 11^T nécessaire

Même conditions : MNIST n=500, Student-t, K_X affine adaptatif gamma=0.5,
init PCA, Adam lr=0.1, 800 steps.

Cas comparés :
  (a)   RV pur (lambda=0)
  (b–f) RV + lambda·log det*(K_Y)  pour lambda in {0.001, 0.01, 0.1, 1.0, 10.0}
  (g)   référence t-SNE sklearn (perplexity=30)
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
    PERPLEXITY, SEED, SOFTENING,
    normalize_kernel, pca_init, to_tensor,
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
LAMBDAS = [0.001, 0.01, 0.1, 1.0, 10.0]
FIG_DIR = Path("results/figures/temp")
DEV = "cpu"


def gram_student(Y: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(Y, Y) ** 2
    return 1.0 / (1.0 + d2)


def pseudo_log_det(K: torch.Tensor) -> torch.Tensor:
    """Log pseudo-déterminant : somme des logs des n-1 vp non nulles.

    K_Y ∈ K_n a exactement une vp nulle structurelle (K_Y @ 1 = 0).
    On saute la plus petite vp (index 0 après tri croissant).
    """
    ev = torch.linalg.eigvalsh(K)          # ordre croissant, réelles (symétrique)
    ev_nonzero = ev[1:].clamp_min(1e-10)   # skip la vp structurellement nulle
    return ev_nonzero.log().sum()


def optimize(K_X, w, init, lam: float = 0.0) -> np.ndarray:
    """PULL = RV normalisé = <K_X, K_Y> / ||K_Y||  (||K_X||=1 après normalize_kernel)."""
    Y = torch.tensor(init, dtype=torch.float32, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([Y], lr=LR)
    for _ in range(N_ITER):
        opt.zero_grad()
        G = gram_student(Y)
        K_Y = double_center(G, w, DEV)
        K_Y_frob = (K_Y * K_Y).sum().sqrt().clamp_min(1e-10)
        pull = (K_X * K_Y).sum() / K_Y_frob   # RV normalisé
        if lam > 0:
            loss = -(pull + lam * pseudo_log_det(K_Y))
        else:
            loss = -pull
        loss.backward()
        opt.step()
    return Y.detach().cpu().numpy()


def metrics(Y, K_X, w, labels):
    Yt = to_tensor(Y, DEV)
    G = gram_student(Yt)
    K_Y = double_center(G, w, DEV)
    rv     = float(rv_coefficient(K_X, K_Y))
    km     = KMeans(n_clusters=len(np.unique(labels)), n_init=10,
                    random_state=SEED).fit_predict(Y)
    ari    = float(adjusted_rand_score(labels, km))
    spread = float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))
    ev     = torch.linalg.eigvalsh(K_Y).detach().cpu().numpy()
    ev_nz  = ev[1:]
    spec_ratio = float(ev_nz[-1] / max(ev_nz[0], 1e-12))
    pld    = float(pseudo_log_det(K_Y))
    return rv, ari, spread, spec_ratio, pld


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

    print("Piste 2 (v2) — PUSH spectral : RV(K_X, K_Y) + lambda * log det*(K_Y)")
    print(f"n={n}, Student-t, gamma={SOFTENING}, {N_ITER} steps, lr={LR}\n")
    cols = f"  {'cas':<30} {'RV':>7} {'ARI':>7}"
    cols += f" {'spread':>8} {'spec_ratio':>11} {'log_det*':>10}"
    print(cols)
    print("  " + "-" * 80)

    # (a) RV pur
    Y0 = optimize(K_X, w, init, lam=0.0)
    rv, ari, sp, sr, pld = metrics(Y0, K_X, w, labels)
    print(f"  {'(a) RV pur (lambda=0)':<30} {rv:>7.4f} {ari:>7.4f} {sp:>8.3f} {sr:>11.2e} {pld:>10.2f}")

    # (b–f) RV + lambda * log det*
    best_ari, best_lam, best_Y = -1.0, None, None
    for lam in LAMBDAS:
        Yl = optimize(K_X, w, init, lam=lam)
        rv, ari, sp, sr, pld = metrics(Yl, K_X, w, labels)
        print(f"  {'lambda=' + str(lam):<30} {rv:>7.4f} {ari:>7.4f} {sp:>8.3f} {sr:>11.2e} {pld:>10.2f}")
        if ari > best_ari:
            best_ari, best_lam, best_Y = ari, lam, Yl

    # (g) référence t-SNE sklearn
    print("  (calcul t-SNE sklearn...)", flush=True)
    Yref = TSNE(n_components=D, perplexity=PERPLEXITY,
                random_state=SEED).fit_transform(ds.X)
    rv, ari, sp, sr, pld = metrics(Yref, K_X, w, labels)
    print(f"  {'(g) t-SNE sklearn':<30} {rv:>7.4f} {ari:>7.4f} {sp:>8.3f} {sr:>11.2e} {pld:>10.2f}")

    print(f"\n  meilleur lambda log-det : {best_lam}  (ARI={best_ari:.4f})")

    # figures : RV pur / meilleur log-det / t-SNE sklearn
    panels = [
        ("RV pur (λ=0)", Y0),
        (f"RV + log det* λ={best_lam}\n(meilleur ARI)", best_Y),
        ("t-SNE sklearn\n(référence)", Yref),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, Y) in zip(axes, panels):
        ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        "Piste 2 (v2) : PUSH spectral  RV(K_X,K_Y) + λ·log det*(K_Y)\n"
        "PULL = cosinus RV normalisé  |  référence = t-SNE sklearn"
    )
    fig.tight_layout()
    out = FIG_DIR / "test_log_det.png"
    fig.savefig(out, dpi=130)
    print(f"\n  figure -> {out}")


if __name__ == "__main__":
    main()
