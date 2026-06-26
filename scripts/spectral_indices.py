"""
spectral_indices.py  —  indices from saved spectral coordinates
===============================================================

Reads the embeddings saved by spectral_run.py (results/coordinates/spectral/*.npy
+ run_meta.csv) and computes, for each (dataset × method):
  * identity vs the reference: Procrustes disparity, kNN overlap,
  * quality: trustworthiness, ARI (labelled datasets only),
  * rv_final (read from run_meta.csv).

Reference-only quality (trustworthiness, ARI) is logged once per method. Two files
are written to results/indices/spectral/ on each run:
  * spectral_indices_tidy.csv  — long/tidy form (one row per index value),
  * spectral_indices_table.csv — wide methods × indices table (framework variant).
Variants in the tidy file: framework, reference.
"""

from __future__ import annotations

import csv

import numpy as np

from pathlib import Path

from src import indices as ix
from src.benchmark_common import (
    SEED,
    SPECTRAL_METHODS,
    TRUST_K,
    IndexLog,
    coord_path,
    indices_dir,
    meta_path,
)
from src.datasets import load_all

FAMILY = "spectral"
VARIANT = "framework"  # the single framework embedding (readout set per method)

# columns of the methods × indices table (spectral_indices_table.csv), in order
TABLE_INDICES = ["rv_final", "procrustes", "knn_overlap", "trustworthiness", "ari"]
TABLE_DECIMALS = 4


def write_table(
    rows: list[dict[str, object]],
    dataset_names: list[str],
    method_names: list[str],
    path: Path,
) -> Path:
    """Pivot the tidy rows into a wide methods × indices table: one row per
    (dataset, method) for the framework_linear variant, columns = TABLE_INDICES.
    Missing cells (e.g. ARI on unlabelled Swiss-roll) are left blank."""
    cell: dict[tuple[str, str], dict[str, object]] = {}
    for r in rows:
        if r["variant"] != VARIANT:
            continue
        cell.setdefault((str(r["dataset"]), str(r["method"])), {})[
            str(r["index_name"])
        ] = r["value"]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", *TABLE_INDICES])
        for ds_name in dataset_names:
            for m_name in method_names:
                vals = cell.get((ds_name, m_name), {})
                out = [ds_name, m_name]
                for idx in TABLE_INDICES:
                    v = vals.get(idx, "")
                    out.append(round(float(v), TABLE_DECIMALS) if v != "" else "")
                writer.writerow(out)
    return path


def load_rv_meta() -> dict[tuple[str, str], float]:
    meta: dict[tuple[str, str], float] = {}
    with meta_path(FAMILY).open() as f:
        for r in csv.DictReader(f):
            meta[(r["dataset"], r["method_key"])] = float(r["rv_final"])
    return meta


def main() -> None:
    datasets = load_all(random_state=SEED)
    meta = load_rv_meta()
    log = IndexLog()

    for ds in datasets.values():
        for m in SPECTRAL_METHODS:
            ref = np.load(coord_path(FAMILY, ds.name, m.key, "reference"))
            fw = np.load(coord_path(FAMILY, ds.name, m.key, "framework"))

            log.add(
                FAMILY,
                ds.name,
                m.name,
                "procrustes",
                ix.procrustes_disparity(fw, ref),
                variant=VARIANT,
                k=TRUST_K,
            )
            log.add(
                FAMILY,
                ds.name,
                m.name,
                "knn_overlap",
                ix.knn_overlap(fw, ref, k=TRUST_K),
                variant=VARIANT,
                k=TRUST_K,
            )
            log.add(
                FAMILY,
                ds.name,
                m.name,
                "trustworthiness",
                ix.trustworthiness(ds.X, fw, k=TRUST_K),
                variant=VARIANT,
                k=TRUST_K,
            )
            if ds.has_labels:
                log.add(
                    FAMILY,
                    ds.name,
                    m.name,
                    "ari",
                    ix.ari(fw, ds.labels),
                    variant=VARIANT,
                )
            log.add(
                FAMILY,
                ds.name,
                m.name,
                "rv_final",
                meta[(ds.name, m.key)],
                variant=VARIANT,
            )

            # reference quality (framework-independent), logged once
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

    out_dir = indices_dir(FAMILY)
    tidy = log.write(path=out_dir / "spectral_indices_tidy.csv", append=False)
    table = write_table(
        log.rows,
        list(datasets.keys()),
        [m.name for m in SPECTRAL_METHODS],
        out_dir / "spectral_indices_table.csv",
    )
    print(f"\nWrote tidy  → {tidy}")
    print(f"Wrote table → {table}")


if __name__ == "__main__":
    main()
