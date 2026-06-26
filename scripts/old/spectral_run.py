"""
spectral_run.py  —  §5.3.1  compute & save spectral embeddings (coordinates)
============================================================================

For every (dataset × spectral method) this computes three embeddings:
  * reference          — the named library implementation,
  * framework_linear   — RV-maximised with a LINEAR output kernel (√λ-scaled axes),
  * framework_projector— RV-maximised with the PROJECTOR output kernel (balanced
                         axes; the projector loss only fixes span(Y), so the
                         embedding is canonicalised by orthonormalising Ỹ = Q Y).

It only *saves coordinates* to results/coordinates/spectral/ plus the final RV of
each run in run_meta.csv. Indices and figures are built from these coordinates by
the two companion scripts: spectral_indices.py and spectral_figures.py.
"""

from __future__ import annotations

import csv
import time

import numpy as np
import torch

from src.benchmark_common import (
    SEED,
    SPECTRAL_METHODS,
    coord_path,
    coords_dir,
    get_device,
    meta_path,
    pca_init,
    rv_embed,
    to_tensor,
)
from src.datasets import load_all
from src.rv_kernels import centering_operator, default_weights

SECTION = "5.3.1"
FAMILY = "spectral"
META_FIELDS = ["dataset", "method_key", "method", "output_kernel", "rv_final"]


def canonicalize_projector(
    Y: np.ndarray, weights: torch.Tensor, device: str
) -> np.ndarray:
    """The projector loss is invariant to Y → Y M, so only span(Y) is fixed.
    Return a balanced representative: an orthonormal basis of Ỹ = Q Y."""
    Q = centering_operator(weights, device=device)
    Ytil = Q @ to_tensor(Y, device)
    U, _ = torch.linalg.qr(Ytil)
    return U.cpu().numpy().astype(np.float32)


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")
    datasets = load_all(random_state=SEED)
    meta_rows: list[dict[str, object]] = []

    for ds in datasets.values():
        print("=" * 64)
        print(f"§{SECTION}  dataset = {ds.name}  (n={ds.n}, d={ds.X.shape[1]})")
        print("=" * 64)
        X_t = to_tensor(ds.X, device)
        w = default_weights(ds.n, device)
        init = pca_init(ds.X)

        for m in SPECTRAL_METHODS:
            t0 = time.time()
            Y_ref = np.asarray(m.reference(ds.X, ds), dtype=np.float32)
            np.save(coord_path(FAMILY, ds.name, m.key, "reference"), Y_ref)

            K_in = m.input_kernel(X_t, ds, device, w)

            Y_lin, rv_lin = rv_embed(K_in, init, device, "linear", weights=w)
            np.save(
                coord_path(FAMILY, ds.name, m.key, "framework_linear"),
                Y_lin.astype(np.float32),
            )

            Y_proj_raw, rv_proj = rv_embed(K_in, init, device, "projector", weights=w)
            Y_proj = canonicalize_projector(Y_proj_raw, w, device)
            np.save(coord_path(FAMILY, ds.name, m.key, "framework_projector"), Y_proj)

            for ok, rv in (("linear", rv_lin), ("projector", rv_proj)):
                meta_rows.append(
                    {
                        "dataset": ds.name,
                        "method_key": m.key,
                        "method": m.name,
                        "output_kernel": ok,
                        "rv_final": rv,
                    }
                )
            print(
                f"  {m.name:<22} rv_linear={rv_lin:.4f}  rv_projector={rv_proj:.4f}  "
                f"({time.time() - t0:.1f}s)"
            )

    with meta_path(FAMILY).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"\nSaved coordinates + run_meta.csv → {coords_dir(FAMILY)}")


if __name__ == "__main__":
    main()
