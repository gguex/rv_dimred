"""
hybrids_indices.py  —  §5.3.3  indices from saved coordinates
=============================================================

Reads results/coordinates/hybrids/*.npy + run_meta.csv and logs, for every hybrid
embedding (global-local α×ν grid, and unsup-sup α-sweep on labelled datasets):
trustworthiness, ARI (labelled datasets) and rv_final. The hybrids have no library
reference, so there is no Procrustes / kNN-overlap here.

Rows are appended to results/results_indices.csv after dropping old §5.3.3 rows
(so §5.3.1 / §5.3.2 stay untouched). The `method` field carries the sweep key
(e.g. ``globallocal_a0.5_nu1.0`` / ``unsupsup_a0.25``).
"""

from __future__ import annotations

import csv

import numpy as np

from src import indices as ix
from src.benchmark_common import (
    HYBRID_ALPHAS,
    HYBRID_NUS,
    SEED,
    TRUST_K,
    IndexLog,
    coord_path,
    drop_sections,
    meta_path,
)
from src.datasets import Dataset, load_all

SECTION = "5.3.3"
FAMILY = "hybrids"


def load_rv_meta() -> dict[tuple[str, str], float]:
    meta: dict[tuple[str, str], float] = {}
    with meta_path(FAMILY).open() as f:
        for r in csv.DictReader(f):
            meta[(r["dataset"], r["key"])] = float(r["rv_final"])
    return meta


def dataset_keys(ds: Dataset) -> list[str]:
    keys = [
        f"globallocal_a{a}_nu{nk}" for a in HYBRID_ALPHAS for nk, _, _ in HYBRID_NUS
    ]
    if ds.has_labels:
        keys += [f"unsupsup_a{a}" for a in HYBRID_ALPHAS]
    return keys


def main() -> None:
    datasets = load_all(random_state=SEED)
    meta = load_rv_meta()
    log = IndexLog()

    for ds in datasets.values():
        for key in dataset_keys(ds):
            Y = np.load(coord_path(FAMILY, ds.name, key, "embedding"))
            log.add(
                SECTION,
                ds.name,
                key,
                "trustworthiness",
                ix.trustworthiness(ds.X, Y, k=TRUST_K),
                k=TRUST_K,
            )
            log.add(SECTION, ds.name, key, "rv_final", meta[(ds.name, key)])
            if ds.has_labels:
                log.add(SECTION, ds.name, key, "ari", ix.ari(Y, ds.labels))
        print(f"  {ds.name:<11} {len(dataset_keys(ds))} embeddings")

    drop_sections({SECTION})  # idempotent; keep §5.3.1 / §5.3.2 rows
    path = log.write(append=True)
    print(f"\nAppended indices → {path}")


if __name__ == "__main__":
    main()
