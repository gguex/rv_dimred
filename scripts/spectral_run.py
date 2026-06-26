"""
spectral_run.py  —  §5.3.1  compute & save spectral embeddings (closed-form solver)
===================================================================================

For every (dataset × spectral method) this computes two embeddings:
  * reference         — the named library implementation,
  * framework_linear  — closed-form RV optimum with a LINEAR output kernel
                        (√λ-scaled axes; Theorem 2 / classical MDS).

The framework embedding comes from the *spectral solver* in src/rv_kernels
(``spectral_embed_linear``): a single eigendecomposition of the centred input
kernel K_X, modular in the input kernel. No gradient ascent, no initialisation —
the eigh gives the global RV optimum.

It only *saves coordinates* to results/coordinates/spectral/ plus the final RV of
each run in run_meta.csv. Indices and figures are built from these coordinates by
the two companion scripts: spectral_indices.py and spectral_figures.py.
"""

from __future__ import annotations

import csv
import time

import numpy as np

from src.benchmark_common import (
    Q,
    SEED,
    SPECTRAL_METHODS,
    coord_path,
    coords_dir,
    get_device,
    meta_path,
    to_tensor,
)
from src.datasets import load_all
from src.rv_kernels import default_weights, spectral_embed_linear

FAMILY = "spectral"
META_FIELDS = ["dataset", "method_key", "method", "output_kernel", "rv_final"]


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")
    datasets = load_all(random_state=SEED)
    meta_rows: list[dict[str, object]] = []

    for ds in datasets.values():
        print("=" * 64)
        print(f"dataset = {ds.name}  (n={ds.n}, d={ds.X.shape[1]})")
        print("=" * 64)
        X_t = to_tensor(ds.X, device)
        w = default_weights(ds.n, device)

        for m in SPECTRAL_METHODS:
            t0 = time.time()
            Y_ref = np.asarray(m.reference(ds.X, ds), dtype=np.float32)
            np.save(coord_path(FAMILY, ds.name, m.key, "reference"), Y_ref)

            K_in = m.input_kernel(X_t, ds, device, w)

            Y, rv = spectral_embed_linear(K_in, q=Q, weights=w, device=device)
            np.save(
                coord_path(FAMILY, ds.name, m.key, "framework_linear"),
                Y.cpu().numpy().astype(np.float32),
            )
            meta_rows.append(
                {
                    "dataset": ds.name,
                    "method_key": m.key,
                    "method": m.name,
                    "output_kernel": "linear",
                    "rv_final": rv,
                }
            )
            print(f"  {m.name:<22} rv_linear={rv:.4f}  ({time.time() - t0:.1f}s)")

    with meta_path(FAMILY).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"\nSaved coordinates + run_meta.csv → {coords_dir(FAMILY)}")


if __name__ == "__main__":
    main()
