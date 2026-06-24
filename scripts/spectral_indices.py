"""
spectral_indices.py  —  §5.3.1  indices from saved coordinates
==============================================================

Reads the embeddings saved by run_spectral.py (results/coordinates/spectral/*.npy
+ run_meta.csv) and computes, for each (dataset × method × output kernel):
  * identity vs the reference: Procrustes disparity, kNN overlap,
  * quality: trustworthiness, ARI (labelled datasets only),
  * rv_final (read from run_meta.csv).

Reference-only quality (trustworthiness, ARI) is logged once per method. Rows are
appended to results/results_indices.csv after dropping old §5.3.1 rows (so the
§5.3.2 / §5.3.3 results stay untouched). Variants: framework_linear,
framework_projector, reference.
"""

from __future__ import annotations

import csv

import numpy as np

from src import indices as ix
from src.datasets import load_all
from src.benchmark_common import (
    COORDS_DIR,
    SEED,
    SPECTRAL_METHODS,
    TRUST_K,
    IndexLog,
    coord_path,
    drop_sections,
)

SECTION = "5.3.1"
OUTPUT_KERNELS = ["linear", "projector"]


def load_rv_meta() -> dict[tuple[str, str, str], float]:
    meta: dict[tuple[str, str, str], float] = {}
    with (COORDS_DIR / "run_meta.csv").open() as f:
        for r in csv.DictReader(f):
            meta[(r["dataset"], r["method_key"], r["output_kernel"])] = float(
                r["rv_final"]
            )
    return meta


def main() -> None:
    datasets = load_all(random_state=SEED)
    meta = load_rv_meta()
    log = IndexLog()

    for ds in datasets.values():
        for m in SPECTRAL_METHODS:
            ref = np.load(coord_path(ds.name, m.key, "reference"))
            for ok in OUTPUT_KERNELS:
                fw = np.load(coord_path(ds.name, m.key, f"framework_{ok}"))
                variant = f"framework_{ok}"
                log.add(
                    SECTION,
                    ds.name,
                    m.name,
                    "procrustes",
                    ix.procrustes_disparity(fw, ref),
                    variant=variant,
                    k=TRUST_K,
                )
                log.add(
                    SECTION,
                    ds.name,
                    m.name,
                    "knn_overlap",
                    ix.knn_overlap(fw, ref, k=TRUST_K),
                    variant=variant,
                    k=TRUST_K,
                )
                log.add(
                    SECTION,
                    ds.name,
                    m.name,
                    "trustworthiness",
                    ix.trustworthiness(ds.X, fw, k=TRUST_K),
                    variant=variant,
                    k=TRUST_K,
                )
                if ds.has_labels:
                    log.add(
                        SECTION,
                        ds.name,
                        m.name,
                        "ari",
                        ix.ari(fw, ds.labels),
                        variant=variant,
                    )
                log.add(
                    SECTION,
                    ds.name,
                    m.name,
                    "rv_final",
                    meta[(ds.name, m.key, ok)],
                    variant=variant,
                )

            # reference quality (output-kernel independent), logged once
            log.add(
                SECTION,
                ds.name,
                m.name,
                "trustworthiness",
                ix.trustworthiness(ds.X, ref, k=TRUST_K),
                variant="reference",
                k=TRUST_K,
            )
            if ds.has_labels:
                log.add(
                    SECTION,
                    ds.name,
                    m.name,
                    "ari",
                    ix.ari(ref, ds.labels),
                    variant="reference",
                )
            print(f"  {ds.name:<11} {m.name}")

    drop_sections({SECTION})  # idempotent; keep §5.3.2 / §5.3.3 rows
    path = log.write(append=True)
    print(f"\nAppended indices → {path}")


if __name__ == "__main__":
    main()
