"""
approximations_indices.py  —  §5.3.2  indices from saved coordinates
====================================================================

Reads results/coordinates/approximations/*.npy + run_meta.csv and computes, for
each (dataset × method):
  * identity vs the reference: Procrustes disparity, kNN overlap,
  * quality: trustworthiness, ARI (labelled datasets), for framework and reference,
  * rv_final (framework, from run_meta.csv).

Rows are appended to results/results_indices.csv after dropping old §5.3.2 rows
(so §5.3.1 / §5.3.3 stay untouched).
"""

from __future__ import annotations

import csv

import numpy as np

from src import indices as ix
from src.benchmark_common import (
    SEED,
    TRUST_K,
    IndexLog,
    coord_path,
    drop_sections,
    meta_path,
)
from src.datasets import Dataset, load_all

SECTION = "5.3.2"
FAMILY = "approximations"
METHODS = [("t-SNE", "tsne"), ("UMAP", "umap")]


def load_meta() -> dict[tuple[str, str], dict[str, str]]:
    meta: dict[tuple[str, str], dict[str, str]] = {}
    with meta_path(FAMILY).open() as f:
        for r in csv.DictReader(f):
            meta[(r["dataset"], r["method_key"])] = r
    return meta


def log_quality(
    log: IndexLog, ds: Dataset, method: str, variant: str, Y: np.ndarray
) -> None:
    log.add(
        SECTION,
        ds.name,
        method,
        "trustworthiness",
        ix.trustworthiness(ds.X, Y, k=TRUST_K),
        variant=variant,
        k=TRUST_K,
    )
    if ds.has_labels:
        log.add(SECTION, ds.name, method, "ari", ix.ari(Y, ds.labels), variant=variant)


def main() -> None:
    datasets = load_all(random_state=SEED)
    meta = load_meta()
    log = IndexLog()

    for ds in datasets.values():
        for name, key in METHODS:
            fw = np.load(coord_path(FAMILY, ds.name, key, "framework"))
            ref = np.load(coord_path(FAMILY, ds.name, key, "reference"))
            log.add(
                SECTION,
                ds.name,
                name,
                "procrustes",
                ix.procrustes_disparity(fw, ref),
                k=TRUST_K,
            )
            log.add(
                SECTION,
                ds.name,
                name,
                "knn_overlap",
                ix.knn_overlap(fw, ref, k=TRUST_K),
                k=TRUST_K,
            )
            log.add(
                SECTION,
                ds.name,
                name,
                "hyperparam",
                float(meta[(ds.name, key)]["hyperparam"]),
                variant="framework",
            )
            log.add(
                SECTION,
                ds.name,
                name,
                "rv_final",
                float(meta[(ds.name, key)]["rv_final"]),
                variant="framework",
            )
            log_quality(log, ds, name, "framework", fw)
            log_quality(log, ds, name, "reference", ref)
            print(f"  {ds.name:<11} {name}")

    drop_sections({SECTION})  # idempotent; keep §5.3.1 / §5.3.3 rows
    path = log.write(append=True)
    print(f"\nAppended indices → {path}")


if __name__ == "__main__":
    main()
