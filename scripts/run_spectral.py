"""
run_spectral.py  —  §5.3.1  Spectral methods (exact recovery)
=============================================================

For every (spectral method × dataset):
  * reference embedding (named library implementation, output = linear),
  * framework embedding (input kernel maximised against a LINEAR output kernel),
  * identity indices  (Procrustes disparity, kNN overlap)  → expect ~0 / ~1,
  * quality indices   (trustworthiness, ARI for labelled datasets),
  * two separate scatter figures (framework / reference) in a shared Procrustes
    frame, for side-by-side placement in the paper.

Indices append to results/results_indices.csv; this driver first drops any old
§5.3.1 rows, so re-running is idempotent and leaves §5.3.2 / §5.3.3 untouched.
"""

from __future__ import annotations

import time

from src import indices as ix
from src.datasets import load_all
from src.rv_kernels import default_weights
from src.section5_common import (
    SEED,
    SPECTRAL_METHODS,
    TRUST_K,
    IndexLog,
    drop_sections,
    get_device,
    pca_init,
    rv_embed,
    save_pair_figures,
    to_tensor,
)

SECTION = "5.3.1"


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")
    log = IndexLog()
    datasets = load_all(random_state=SEED)

    for ds in datasets.values():
        print("=" * 64)
        print(f"§{SECTION}  dataset = {ds.name}  (n={ds.n}, d={ds.X.shape[1]})")
        print("=" * 64)
        X_t = to_tensor(ds.X, device)
        w = default_weights(ds.n, device)
        init = pca_init(ds.X)

        for m in SPECTRAL_METHODS:
            t0 = time.time()
            Y_ref = m.reference(ds.X, ds)
            K_in = m.input_kernel(X_t, ds, device, w)
            Y_fw, rv = rv_embed(K_in, init, device, output_kernel="linear", weights=w)

            disparity = ix.procrustes_disparity(Y_ref, Y_fw)
            knn = ix.knn_overlap(Y_ref, Y_fw, k=TRUST_K)

            # identity indices (framework vs reference)
            log.add(SECTION, ds.name, m.name, "procrustes", disparity, k=TRUST_K)
            log.add(SECTION, ds.name, m.name, "knn_overlap", knn, k=TRUST_K)
            log.add(SECTION, ds.name, m.name, "rv_final", rv)

            # quality indices for BOTH embeddings
            for variant, Y in (("framework", Y_fw), ("reference", Y_ref)):
                log.add(
                    SECTION,
                    ds.name,
                    m.name,
                    "trustworthiness",
                    ix.trustworthiness(ds.X, Y, k=TRUST_K),
                    variant=variant,
                    k=TRUST_K,
                )
                if ds.has_labels:
                    log.add(
                        SECTION,
                        ds.name,
                        m.name,
                        "ari",
                        ix.ari(Y, ds.labels),
                        variant=variant,
                    )

            save_pair_figures(
                SECTION,
                ds,
                m.key,
                m.name,
                Y_fw,
                Y_ref,
                fw_note=f"Procrustes={disparity:.4f}  RV={rv:.4f}",
            )
            print(
                f"  {m.name:<22} proc={disparity:.4f} knn={knn:.3f} "
                f"rv={rv:.4f}  ({time.time() - t0:.1f}s)"
            )

    drop_sections({SECTION})  # idempotent re-run; keep §5.3.2 / §5.3.3 rows
    path = log.write(append=True)
    print(f"\nSaved indices → {path}")


if __name__ == "__main__":
    main()
