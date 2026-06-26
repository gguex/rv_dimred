"""
spectral_indices.py  —  indices from saved spectral coordinates
===============================================================

Reads the embeddings saved by spectral_run.py (results/coordinates/spectral/*.npy
+ run_meta.csv) and computes, for each (dataset × method):
  * identity vs the reference: Procrustes disparity, kNN overlap,
  * quality: trustworthiness, ARI (labelled datasets only),
  * rv_final (read from run_meta.csv).

Reference-only quality (trustworthiness, ARI) is logged once per method. The whole
results/results_indices.csv is rewritten on each run. Variants: framework_linear,
reference.
"""

from __future__ import annotations

import csv

import numpy as np

from src import indices as ix
from src.benchmark_common import (
    SEED,
    SPECTRAL_METHODS,
    TRUST_K,
    IndexLog,
    coord_path,
    meta_path,
)
from src.datasets import load_all

FAMILY = "spectral"
OUTPUT_KERNELS = ["linear"]


def load_rv_meta() -> dict[tuple[str, str, str], float]:
    meta: dict[tuple[str, str, str], float] = {}
    with meta_path(FAMILY).open() as f:
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
            ref = np.load(coord_path(FAMILY, ds.name, m.key, "reference"))
            for ok in OUTPUT_KERNELS:
                fw = np.load(coord_path(FAMILY, ds.name, m.key, f"framework_{ok}"))
                variant = f"framework_{ok}"
                log.add(
                    FAMILY,
                    ds.name,
                    m.name,
                    "procrustes",
                    ix.procrustes_disparity(fw, ref),
                    variant=variant,
                    k=TRUST_K,
                )
                log.add(
                    FAMILY,
                    ds.name,
                    m.name,
                    "knn_overlap",
                    ix.knn_overlap(fw, ref, k=TRUST_K),
                    variant=variant,
                    k=TRUST_K,
                )
                log.add(
                    FAMILY,
                    ds.name,
                    m.name,
                    "trustworthiness",
                    ix.trustworthiness(ds.X, fw, k=TRUST_K),
                    variant=variant,
                    k=TRUST_K,
                )
                if ds.has_labels:
                    log.add(
                        FAMILY,
                        ds.name,
                        m.name,
                        "ari",
                        ix.ari(fw, ds.labels),
                        variant=variant,
                    )
                log.add(
                    FAMILY,
                    ds.name,
                    m.name,
                    "rv_final",
                    meta[(ds.name, m.key, ok)],
                    variant=variant,
                )

            # reference quality (output-kernel independent), logged once
            log.add(
                FAMILY,
                ds.name,
                m.name,
                "trustworthiness",
                ix.trustworthiness(ds.X, ref, k=TRUST_K),
                variant="reference",
                k=TRUST_K,
            )
            if ds.has_labels:
                log.add(
                    FAMILY,
                    ds.name,
                    m.name,
                    "ari",
                    ix.ari(ref, ds.labels),
                    variant="reference",
                )
            print(f"  {ds.name:<11} {m.name}")

    path = log.write(append=False)
    print(f"\nWrote indices → {path}")


if __name__ == "__main__":
    main()
