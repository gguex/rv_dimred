import numpy as np
import torch

from src.rv_kernels import (
    compute_class_kernel_torch,
    compute_geodesic_kernel_torch,
    default_weights,
    rv_dimred,
)

torch.manual_seed(0)
np.random.seed(0)

# toy data: 3 Gaussian blobs in 10-D
n_per, dim = 60, 10
centers = np.random.randn(3, dim) * 6
X = np.vstack([c + np.random.randn(n_per, dim) for c in centers]).astype(np.float32)
labels = np.repeat(np.arange(3), n_per)
Xt = torch.as_tensor(X)
n = X.shape[0]
w = default_weights(n)

# --- Brick 1: choose an INPUT kernel (target geometry) ---
K_X = compute_geodesic_kernel_torch(Xt, param={"k": 10}, weights=w)

# --- Brick 2: choose an OUTPUT kernel (model geometry) ---  here: Student-t (t-SNE-like)
Y, rv = rv_dimred(
    K_X,
    output_kernel="student_t",
    q=2,
    weights=w,
    output_param=1.0,
    n_iter=300,
    lr=0.5,
    verbose=True,
)
print(f"Geodesic + Student-t  ->  RV = {rv:.4f}")

# --- Supervised mix: K(alpha) = (1-alpha) K_X + alpha K_Z  (a pure lego op) ---
K_Z = compute_class_kernel_torch(Xt, param={"labels": labels}, weights=w)
alpha = 0.5
K_mix = (1 - alpha) * K_X + alpha * K_Z
Y2, rv2 = rv_dimred(
    K_mix,
    output_kernel="student_t",
    q=2,
    weights=w,
    output_param=1.0,
    n_iter=300,
    lr=0.5,
)
print(f"Soft-supervised (alpha={alpha})  ->  RV = {rv2:.4f}")
