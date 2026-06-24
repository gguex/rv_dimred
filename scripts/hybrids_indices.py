"""
hybrids_indices.py  —  §5.3.3  indices from saved coordinates
=============================================================

Reads results/coordinates/hybrids/*.npy + run_meta.csv and logs, for every β of
the supervised interpolation (class-kernel ↔ t-SNE), on each labelled dataset:
trustworthiness and ARI computed BOTH on the train embedding and on the projected
test embedding (variant ``train`` / ``test``), plus rv_final. The train/test split
is re-derived deterministically so the saved test coordinates line up with labels.

Rows are appended to results/results_indices.csv after dropping old §5.3.3 rows
(so §5.3.1 / §5.3.2 stay untouched). The ``method`` field carries the sweep key
(e.g. ``supervised_b0.5``).
"""

from __future__ import annotations

import csv

import numpy as np

from src import indices as ix
from src.benchmark_common import (
    SEED,
    SUPERVISED_BETAS,
    TRUST_K,
    IndexLog,
    coord_path,
    drop_sections,
    meta_path,
    supervised_split,
)
from src.datasets import load_all

SECTION = "5.3.3"
FAMILY = "hybrids"


def load_rv_meta() -> dict[tuple[str, str], float]:
    meta: dict[tuple[str, str], float] = {}
    with meta_path(FAMILY).open() as f:
        for r in csv.DictReader(f):
            meta[(r["dataset"], r["key"])] = float(r["rv_final"])
    return meta


def main() -> None:
    datasets = load_all(random_state=SEED)
    meta = load_rv_meta()
    log = IndexLog()

    for ds in datasets.values():
        if not ds.has_labels:
            continue
        X_tr, X_te, y_tr, y_te = supervised_split(ds.X, ds.labels)
        for beta in SUPERVISED_BETAS:
            key = f"supervised_b{beta}"
            Y_tr = np.load(coord_path(FAMILY, ds.name, key, "train"))
            Y_te = np.load(coord_path(FAMILY, ds.name, key, "test"))
            log.add(SECTION, ds.name, key, "rv_final", meta[(ds.name, key)])
            for variant, X, Y, y in (
                ("train", X_tr, Y_tr, y_tr),
                ("test", X_te, Y_te, y_te),
            ):
                log.add(
                    SECTION, ds.name, key, "trustworthiness",
                    ix.trustworthiness(X, Y, k=TRUST_K), variant=variant, k=TRUST_K,
                )
                log.add(
                    SECTION, ds.name, key, "ari",
                    ix.ari(Y, y), variant=variant,
                )
        print(f"  {ds.name:<11} {len(SUPERVISED_BETAS)} β × (train+test)")

    drop_sections({SECTION})  # idempotent; keep §5.3.1 / §5.3.2 rows
    path = log.write(append=True)
    print(f"\nAppended indices → {path}")


if __name__ == "__main__":
    main()
