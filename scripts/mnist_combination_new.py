from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.rv_kernels import (
    compute_fuzzy_topological_kernel_torch,
    compute_gaussian_affinity_kernel_torch,
    compute_gaussian_kernel_torch,
    compute_geodesic_kernel_torch,
    compute_linear_kernel_torch,
    compute_lle_kernel_torch,
    compute_student_t_kernel_torch,
    compute_umap_kernel_torch,
    default_weights,
    rv_dimred,
)

# --------------------------------------------------------------
# Device
# --------------------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"device: {device}")

# --------------------------------------------------------------
# Data Loading and Preprocessing
# --------------------------------------------------------------
mnist_data = np.genfromtxt(
    Path(__file__).parent.parent / "data/mnist_test.csv", delimiter=",", skip_header=1
)

n_per_digit = 200
mnist_data = np.vstack(
    [mnist_data[mnist_data[:, 0] == i][:n_per_digit] for i in range(10)]
)

mnist_images = (mnist_data[:, 1:] / 255.0).astype(np.float32)
mnist_labels = mnist_data[:, 0].astype(int)
n = mnist_images.shape[0]

X = torch.tensor(mnist_images, dtype=torch.float32, device=device)
w = default_weights(n, device)

# --------------------------------------------------------------
# Input Kernels  (computed once)
# --------------------------------------------------------------
k_neighbors = 15
perplexity = 40

print("Computing input kernels…")
kernels_in = [
    compute_linear_kernel_torch(X, weights=w, device=device),
    compute_geodesic_kernel_torch(
        X, param={"k": k_neighbors}, weights=w, device=device
    ),
    compute_lle_kernel_torch(X, param={"k": k_neighbors}, weights=w, device=device),
    compute_gaussian_affinity_kernel_torch(
        X, param={"perplexity": perplexity}, weights=w, device=device
    ),
    compute_fuzzy_topological_kernel_torch(
        X, param={"k": k_neighbors}, weights=w, device=device
    ),
]
kernel_in_names = ["Linear", "Geodesic", "LLE", "Adapt. Gaussian", "Fuzzy Topo."]

# --------------------------------------------------------------
# Output Kernels
# --------------------------------------------------------------
kernel_out_functions = [
    compute_linear_kernel_torch,
    compute_student_t_kernel_torch,
    compute_gaussian_kernel_torch,
    compute_umap_kernel_torch,
]
kernel_out_params = [None, 1.0, None, None]
kernel_out_names = ["Linear", "Student-t", "Gaussian", "UMAP"]

n_in = len(kernels_in)
n_out = len(kernel_out_functions)

# --------------------------------------------------------------
# Warm start: PCA initialisation
# --------------------------------------------------------------
Y_pca = torch.tensor(
    PCA(n_components=2).fit_transform(mnist_images), dtype=torch.float32, device=device
)

# --------------------------------------------------------------
# Grid optimisation
# --------------------------------------------------------------
RV_matrix = np.zeros((n_in, n_out))
Y_grid: list[list[np.ndarray]] = []

for i in range(n_in):
    Y_row: list[np.ndarray] = []
    for j in range(n_out):
        print(f"  [{i + 1}/{n_in}] {kernel_in_names[i]}  ×  {kernel_out_names[j]}")
        Y_opt, rv_final = rv_dimred(
            kernels_in[i],
            output_kernel=kernel_out_functions[j],
            q=2,
            weights=w,
            output_param=kernel_out_params[j],
            n_iter=500,
            lr=0.1,
            device=device,
            init=Y_pca,
            verbose=False,
        )
        print(f"    RV = {rv_final:.6f}")
        RV_matrix[i, j] = rv_final
        Y_row.append(Y_opt.cpu().numpy())
    Y_grid.append(Y_row)

# --------------------------------------------------------------
# Plot
# --------------------------------------------------------------
out_dir = Path(__file__).parent.parent / "results" / "mnist"
out_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(n_in, n_out, figsize=(n_out * 4, n_in * 4))
for i in range(n_in):
    for j in range(n_out):
        ax = axes[i, j]
        Y = Y_grid[i][j]
        ax.scatter(Y[:, 0], Y[:, 1], c=mnist_labels, cmap="tab10", s=3, alpha=0.7)
        ax.set_title(
            f"{kernel_in_names[i]}  ×  {kernel_out_names[j]}\nRV={RV_matrix[i, j]:.4f}",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(kernel_in_names[i], fontsize=9)
        if i == 0:
            ax.set_xlabel(kernel_out_names[j], fontsize=9)
            ax.xaxis.set_label_position("top")

plt.suptitle("MNIST — RV kernel combinations", fontsize=12, y=1.01)
plt.tight_layout()
out_path = out_dir / "mnist_combinations.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")
