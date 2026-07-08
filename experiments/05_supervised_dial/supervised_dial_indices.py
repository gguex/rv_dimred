"""
supervised_dial_indices.py  —  art. §6.6  train/test metrics of the supervised dial
===================================================================================

Reads the embeddings saved by supervised_dial_run.py and computes, per
(dataset x beta x split in {train, test}):
  * ARI            (KMeans on the embedding vs true labels; test labels used for
                    evaluation only, never for building the embedding),
  * trustworthiness (feature-space neighbourhood preservation, k = TRUST_K).

The point (art. §6.6): under the double dial (input + output moving together), as
beta rises TEST ARI climbs on held-out points whose labels were never used --
supervision generalizes (singlecell 0.50 -> 0.92, mnist 0.35 -> 0.54) -- while test
trustworthiness holds near its t-SNE level through the intermediate regime; beta = 1
overfits (train ARI -> 1) so the test metric correctly rewards the intermediate /
near-full regime. Writes results/05_supervised_dial/indices/supervised_dial.csv.
"""

# ruff: noqa: E402, I001  (imports follow the sys.path bootstrap)
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (for `src`)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src import indices as ix  # noqa: E402
from src.benchmark_common import (  # noqa: E402
    SEED,
    SUPERVISED_BETAS,
    TRUST_K,
    exp_coord_path,
    exp_indices_dir,
    supervised_split,
)
from src.datasets import load_all  # noqa: E402

EXP = "05_supervised_dial"
DATASETS = ("mnist", "singlecell")
CSV_FIELDS = ["dataset", "beta", "split", "ari", "trustworthiness"]


def ari(Y: np.ndarray, labels: np.ndarray) -> float:
    km = KMeans(n_clusters=len(np.unique(labels)), n_init=10, random_state=SEED)
    return float(adjusted_rand_score(labels, km.fit_predict(Y)))


def main() -> None:
    datasets = load_all(random_state=SEED)
    rows: list[dict[str, object]] = []

    for name in DATASETS:
        ds = datasets[name]
        X_tr, X_te, y_tr, y_te = supervised_split(ds.X, ds.labels)
        feats = {"train": X_tr, "test": X_te}
        labs = {"train": y_tr, "test": y_te}

        for beta in SUPERVISED_BETAS:
            key = f"dial_b{beta}"
            for split in ("train", "test"):
                Y = np.load(exp_coord_path(EXP, ds.name, key, split))
                rows.append(
                    {
                        "dataset": ds.name,
                        "beta": beta,
                        "split": split,
                        "ari": ari(Y, labs[split]),
                        "trustworthiness": ix.trustworthiness(
                            feats[split], Y, k=TRUST_K
                        ),
                    }
                )
            tr, te = rows[-2], rows[-1]
            tru, teu = tr["trustworthiness"], te["trustworthiness"]
            print(f"  {ds.name:<11} beta={beta:<4}  "
                  f"ARI tr/te={tr['ari']:.3f}/{te['ari']:.3f}  "
                  f"trust tr/te={tru:.3f}/{teu:.3f}")

    out = exp_indices_dir(EXP) / "supervised_dial.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
