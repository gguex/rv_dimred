"""
tether_run.py  —  art. §6.5  the diagonal tether: full-RV vs hollow-RV vs t-SNE
===============================================================================

Validates Prop. 5 + the justification Lemma (promise of art. §5.1-5.2). Exact fact
(uniform weights, centered kernel): K_ii = -r_i with r_i = sum_{j!=i} K_ij, so the
diagonal energy ||diag(K_Y)||^2 = sum_i K_ii^2 = sum_i r_i^2 is exactly the degree
term the full-RV metric M = I + D*D over-weights. Maximizing the FULL RV pours
energy into that diagonal (dense-core collapse); the HOLLOW RV (both diagonals
zeroed) keeps it down and spreads like t-SNE.

Along each optimization we record the energy split of K_Y:
    e_diag   = sum_i K_ii^2  (= sum_i r_i^2, the degree floor)
    e_hollow = ||K̊_Y||^2     (= ||K_Y||^2 - e_diag, the structural part)
    frac_diag = e_diag / ||K_Y||^2
    spread   = RMS radius of Y
plus the final ARI / RV_full / RV_hollow, and a t-SNE reference. This produces the
data behind Figure 1 (triptych + energy panel): MNIST n=2000, perplexity 30,
gamma 0.5, seed 0.

Writes to results/04_tether/:
    coordinates/{full_rv,hollow_rv,tsne}.npy, labels.npy
    indices/tether_trajectory.csv   (per-iteration energy split, both objectives)
    indices/tether_final.csv        (final comparison, 3 configs)
"""

# ruff: noqa: E402, I001  (imports follow the sys.path bootstrap)
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (for `src`)

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from src.benchmark_common import (  # noqa: E402
    LR_RV,
    N_ITER_RV,
    PERPLEXITY,
    Q,
    SEED,
    SOFTENING,
    exp_coords_dir,
    exp_indices_dir,
    get_device,
    normalize_kernel,
    pca_init,
    to_tensor,
)
from src.datasets import load_mnist  # noqa: E402
from src.rv_kernels import (  # noqa: E402
    default_weights,
    double_center,
    gaussian_affinity_base,
    rv_coefficient,
    soften_and_center,
)

EXP = "04_tether"
N_PER_DIGIT = 200          # MNIST n = 2000
N_ITER = N_ITER_RV         # 500
TRAJ_FIELDS = ["config", "iter", "spread", "e_diag", "e_hollow", "frac_diag"]
FINAL_FIELDS = [
    "config",
    "spread",
    "frac_diag",
    "e_diag",
    "e_hollow",
    "ari",
    "rv_full",
    "rv_hollow",
]


def gram_student(Y: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(Y, Y) ** 2
    return 1.0 / (1.0 + d2)


def energy_split(K_Y: torch.Tensor) -> tuple[float, float, float]:
    """e_diag = sum_i K_ii^2 ; e_hollow = ||K_Y||^2 - e_diag ; frac_diag."""
    diag = K_Y.diagonal()
    e_diag = float((diag * diag).sum())
    e_tot = float((K_Y * K_Y).sum())
    return e_diag, e_tot - e_diag, e_diag / (e_tot + 1e-30)


def spread_of(Y: np.ndarray) -> float:
    return float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))


def optimize(
    K_X: torch.Tensor, w: torch.Tensor, init: np.ndarray, device: str, hollow: bool
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Adam ascent on the (full or hollow) RV; returns final Y and the per-iteration
    energy-split trajectory."""
    Y = torch.tensor(init, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([Y], lr=LR_RV)
    config = "hollow_rv" if hollow else "full_rv"
    traj: list[dict[str, object]] = []
    for it in range(N_ITER + 1):
        K_Y = double_center(gram_student(Y), w, device)
        with torch.no_grad():
            e_diag, e_hollow, frac = energy_split(K_Y)
            traj.append(
                {
                    "config": config,
                    "iter": it,
                    "spread": spread_of(Y.detach().cpu().numpy()),
                    "e_diag": e_diag,
                    "e_hollow": e_hollow,
                    "frac_diag": frac,
                }
            )
        if it == N_ITER:
            break
        opt.zero_grad()
        if hollow:
            pull = rv_coefficient(K_X, K_Y, hollow=True)
        else:
            pull = (K_X * K_Y).sum() / (K_Y.norm() + 1e-10)  # ||K_X|| = 1
        (-pull).backward()
        opt.step()
    return Y.detach().cpu().numpy(), traj


def final_stats(
    name: str, Y: np.ndarray, K_X: torch.Tensor, w: torch.Tensor,
    labels: np.ndarray, device: str,
) -> dict[str, object]:
    K_Y = double_center(gram_student(to_tensor(Y, device)), w, device)
    e_diag, e_hollow, frac = energy_split(K_Y)
    km = KMeans(n_clusters=len(np.unique(labels)), n_init=10, random_state=SEED)
    ari = adjusted_rand_score(labels, km.fit_predict(Y))
    return {
        "config": name,
        "spread": spread_of(Y),
        "frac_diag": frac,
        "e_diag": e_diag,
        "e_hollow": e_hollow,
        "ari": float(ari),
        "rv_full": float(rv_coefficient(K_X, K_Y)),
        "rv_hollow": float(rv_coefficient(K_X, K_Y, hollow=True)),
    }


def main() -> None:
    device = get_device()
    print(f"device: {device}")
    print(f"art. §6.5  tether: full-RV vs hollow-RV vs t-SNE  "
          f"(MNIST n={10 * N_PER_DIGIT}, perp={PERPLEXITY}, gamma={SOFTENING})\n")

    ds = load_mnist(n_per_digit=N_PER_DIGIT, random_state=SEED)
    labels = ds.labels
    w = default_weights(ds.n, device)
    init = pca_init(ds.X)
    X_t = to_tensor(ds.X, device)

    base = gaussian_affinity_base(X_t, PERPLEXITY)
    K_X = normalize_kernel(soften_and_center(base, SOFTENING, w, device))

    Y_full, traj_full = optimize(K_X, w, init, device, hollow=False)
    Y_hollow, traj_hollow = optimize(K_X, w, init, device, hollow=True)
    print("Fitting t-SNE reference...")
    ref = TSNE(n_components=Q, perplexity=PERPLEXITY, random_state=SEED)
    Y_tsne = np.asarray(ref.fit_transform(ds.X), dtype=np.float32)

    # save coordinates for the triptych
    cdir = exp_coords_dir(EXP)
    np.save(cdir / "full_rv.npy", Y_full.astype(np.float32))
    np.save(cdir / "hollow_rv.npy", Y_hollow.astype(np.float32))
    np.save(cdir / "tsne.npy", Y_tsne)
    np.save(cdir / "labels.npy", labels)

    # trajectory CSV
    idir = exp_indices_dir(EXP)
    with (idir / "tether_trajectory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAJ_FIELDS)
        writer.writeheader()
        writer.writerows(traj_full + traj_hollow)

    # final comparison
    finals = [
        final_stats("full_rv", Y_full, K_X, w, labels, device),
        final_stats("hollow_rv", Y_hollow, K_X, w, labels, device),
        final_stats("tsne", Y_tsne, K_X, w, labels, device),
    ]
    with (idir / "tether_final.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_FIELDS)
        writer.writeheader()
        writer.writerows(finals)

    hdr = (f"{'config':<11} | {'spread':>7} | {'frac_diag':>9} | {'e_diag':>10} | "
           f"{'e_hollow':>10} | {'ARI':>5} | {'RV_full':>7} | {'RV_hol':>7}")
    print("\n=== final (iter=500) ===")
    print(hdr)
    print("-" * len(hdr))
    for m in finals:
        print(f"{m['config']:<11} | {m['spread']:>7.3f} | {m['frac_diag']:>9.4f} | "
              f"{m['e_diag']:>10.3e} | {m['e_hollow']:>10.3e} | {m['ari']:>5.3f} | "
              f"{m['rv_full']:>7.4f} | {m['rv_hollow']:>7.4f}")
    print(f"\nSaved coordinates → {cdir}\nSaved indices → {idir}")


if __name__ == "__main__":
    main()
