"""
spectral_run.py  —  art. §7.1  compute & save spectral embeddings (closed-form solver)
=======================================================================================

For every (dataset × spectral method) this computes two embeddings:
  * reference  — the named library implementation,
  * framework  — closed-form RV optimum from a single eigendecomposition of the
                 centred input kernel K_X (src/rv_kernels spectral solvers). The
                 readout matches the reference's axis convention, set per method by
                 SpectralMethod.readout: ``linear`` (√λ scaling; PCA / MDS / Isomap /
                 KPCA / diffusion) or ``orthonormal`` (balanced unit axes; Laplacian
                 Eigenmaps, LLE).

No gradient ascent, no initialisation — the eigh gives the global RV optimum.

It only *saves coordinates* to results/01_spectral/coordinates/, plus, per run, the
final RV, the alignment ceiling RV_max(q) of Prop. 1 (computed from the spectrum of
K_X) and their ratio in run_meta.csv — for the linear readout, rv_final = rv_max by
Theorem 2, which the ratio column verifies. Indices are built from these coordinates
by the companion script spectral_indices.py; the Test A gradient-ascent check lives
in ceiling_check.py; scatter figures are the showcase's job (showcase/).
"""

# ruff: noqa: E402, I001  (imports follow the sys.path bootstrap)
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmark_common import (  # noqa: E402
    Q,
    SEED,
    SPECTRAL_METHODS,
    exp_coord_path,
    exp_coords_dir,
    exp_meta_path,
    get_device,
    to_tensor,
)
from src.datasets import load_all  # noqa: E402
from src.rv_kernels import (  # noqa: E402
    default_weights,
    rv_ceiling,
    spectral_embed_linear,
    spectral_embed_orthonormal,
)

EXP = "01_spectral"
META_FIELDS = [
    "dataset",
    "method_key",
    "method",
    "readout",
    "rv_final",
    "rv_max",
    "rv_ratio",
]

# readout name (SpectralMethod.readout) -> closed-form spectral solver
READOUTS = {
    "linear": spectral_embed_linear,
    "orthonormal": spectral_embed_orthonormal,
}


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
            np.save(exp_coord_path(EXP, ds.name, m.key, "reference"), Y_ref)

            K_in = m.input_kernel(X_t, ds, device, w)
            rv_max = rv_ceiling(K_in, q=Q)  # alignment ceiling (Prop. 1)

            Y, rv = READOUTS[m.readout](K_in, q=Q, weights=w, device=device)
            np.save(
                exp_coord_path(EXP, ds.name, m.key, "framework"),
                Y.cpu().numpy().astype(np.float32),
            )
            meta_rows.append(
                {
                    "dataset": ds.name,
                    "method_key": m.key,
                    "method": m.name,
                    "readout": m.readout,
                    "rv_final": rv,
                    "rv_max": rv_max,
                    "rv_ratio": rv / rv_max if rv_max > 0 else float("nan"),
                }
            )
            print(
                f"  {m.name:<22} readout={m.readout:<11} rv={rv:.4f}  "
                f"rv_max={rv_max:.4f}  ratio={rv / rv_max:.4f}  "
                f"({time.time() - t0:.1f}s)"
            )

    with exp_meta_path(EXP).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"\nSaved coordinates + run_meta.csv → {exp_coords_dir(EXP)}")


if __name__ == "__main__":
    main()
