"""Check that iterative linear-kernel optimization attains the RV ceiling.

The spectral result gives the alignment ceiling RV_max(q) attainable by a
linear-output embedding. This script checks the iterative side of that result:
plain RV gradient ascent (Adam, PCA initialization) with a linear output kernel
reaches RV_max on a non-trivial input kernel, with no eigendecomposition in the
loop.

Input kernel: the adaptive-Gaussian t-SNE affinity (perplexity 30) softened at
gamma = 0.5, matching the input kernel used by the nonlinear experiments.

Writes results/01_spectral/indices/ceiling_check.csv (one row per dataset):
rv_gradient, rv_max, gap = rv_max - rv_gradient.
"""

# ruff: noqa: E402, I001  (imports follow the sys.path bootstrap)
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.benchmark_common import (  # noqa: E402
    N_ITER_RV,
    PERPLEXITY,
    Q,
    SEED,
    SOFTENING,
    exp_indices_dir,
    get_device,
    pca_init,
    rv_embed,
)
from src.datasets import load_all  # noqa: E402
from src.rv_kernels import (  # noqa: E402
    default_weights,
    gaussian_affinity_base,
    rv_ceiling,
    soften_and_center,
)

EXP = "01_spectral"
CSV_FIELDS = ["dataset", "n", "perplexity", "gamma", "rv_gradient", "rv_max", "gap"]


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")
    datasets = load_all(random_state=SEED)
    rows: list[dict[str, object]] = []

    for ds in datasets.values():
        t0 = time.time()
        w = default_weights(ds.n, device)
        base = gaussian_affinity_base(ds.X, perplexity=PERPLEXITY)
        K_in = soften_and_center(base, SOFTENING, w, device)

        rv_max = rv_ceiling(K_in, q=Q)
        _, rv_grad = rv_embed(
            K_in,
            pca_init(ds.X, Q),
            device,
            output_kernel="linear",
            weights=w,
            n_iter=N_ITER_RV,
        )
        rows.append(
            {
                "dataset": ds.name,
                "n": ds.n,
                "perplexity": PERPLEXITY,
                "gamma": SOFTENING,
                "rv_gradient": rv_grad,
                "rv_max": rv_max,
                "gap": rv_max - rv_grad,
            }
        )
        print(
            f"  {ds.name:<11} n={ds.n}  rv_gradient={rv_grad:.4f}  "
            f"rv_max={rv_max:.4f}  gap={rv_max - rv_grad:+.2e}  "
            f"({time.time() - t0:.1f}s)"
        )

    out = exp_indices_dir(EXP) / "ceiling_check.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
