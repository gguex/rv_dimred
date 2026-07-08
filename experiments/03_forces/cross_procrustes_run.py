"""
cross_procrustes_run.py  —  art. §6.4(ii)  Table 4: the objective dominates the kernel
======================================================================================

The payoff prediction of the §5.4 push taxonomy: the RV-cosine objective can
approach t-SNE but NOT UMAP (UMAP's per-pair push on the raw Gram is out of reach
of the pure cosine). We test it by swapping only the *kernels* while keeping the
*objective* (hollow-RV) fixed, and measuring who ends up close to whom.

For MNIST and single-cell (n=2000) at the canonical operating point of §6.4-6.5
(neighbour hp = 30, softening gamma = 0.5) we build four 2-D embeddings:

  framework_tsne  hollow-RV on adaptive-Gaussian input + Student-t output
  framework_umap  hollow-RV on fuzzy-topological input + UMAP output
  reference_tsne  sklearn.manifold.TSNE   (perplexity 30)
  reference_umap  umap-learn UMAP         (n_neighbors 30)

and report the full 4x4 Procrustes-disparity matrix. The prediction: the two
framework variants are closer to *each other* than to their own references
(swapping in UMAP's kernels does not move the result toward UMAP) -- the objective
shapes the embedding, the kernel swap is secondary. Writes
results/03_forces/indices/cross_procrustes.csv (long form, one row per dataset x
ordered pair) plus one 4x4 matrix CSV per dataset.
"""

# ruff: noqa: E402, I001  (imports follow the sys.path bootstrap)
from __future__ import annotations

import csv
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (for `src`)

import numpy as np
import umap
from sklearn.manifold import TSNE

from src import indices as ix  # noqa: E402
from src.benchmark_common import (  # noqa: E402
    Q,
    SEED,
    exp_indices_dir,
    get_device,
    pca_init,
    pca_init_sklearn,
    rv_embed,
    to_tensor,
)
from src.datasets import Dataset, load_all  # noqa: E402
from src.rv_kernels import (  # noqa: E402
    default_weights,
    fuzzy_topological_base,
    gaussian_affinity_base,
    soften_and_center,
)

EXP = "03_forces"
DATASETS = ("mnist", "singlecell")   # labelled neighbour-embedding datasets
PERPLEXITY = 30                       # canonical neighbour hp (t-SNE side)
N_NEIGHBORS = 30                      # canonical neighbour hp (UMAP side)
GAMMA = 0.5                           # input-affinity softening (art. §6.1)
LABELS = ["framework_tsne", "framework_umap", "reference_tsne", "reference_umap"]


def embeddings(ds: Dataset, device: str) -> dict[str, np.ndarray]:
    """The four 2-D embeddings for one dataset at the canonical operating point."""
    X_t = to_tensor(ds.X, device)
    w = default_weights(ds.n, device)
    init_fw = pca_init(ds.X)
    init_ref = pca_init_sklearn(ds.X)

    base_tsne = gaussian_affinity_base(X_t, PERPLEXITY)
    base_umap = fuzzy_topological_base(X_t, N_NEIGHBORS)
    k_tsne = soften_and_center(base_tsne, GAMMA, w, device)
    k_umap = soften_and_center(base_umap, GAMMA, w, device)
    fw_tsne, _ = rv_embed(
        k_tsne, init_fw, device, "student_t", 1.0, weights=w, hollow=True
    )
    fw_umap, _ = rv_embed(
        k_umap, init_fw, device, "umap", None, weights=w, hollow=True
    )

    ref_tsne = TSNE(
        n_components=Q, perplexity=PERPLEXITY, init=init_ref, random_state=SEED
    ).fit_transform(ds.X)
    ref_umap = np.asarray(
        umap.UMAP(
            n_neighbors=N_NEIGHBORS, n_components=Q, init=init_ref, random_state=SEED
        ).fit_transform(ds.X)
    )
    return {
        "framework_tsne": np.asarray(fw_tsne, dtype=np.float32),
        "framework_umap": np.asarray(fw_umap, dtype=np.float32),
        "reference_tsne": np.asarray(ref_tsne, dtype=np.float32),
        "reference_umap": np.asarray(ref_umap, dtype=np.float32),
    }


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")
    datasets = load_all(random_state=SEED)
    out_dir = exp_indices_dir(EXP)
    long_rows: list[dict[str, object]] = []

    for name in DATASETS:
        ds = datasets[name]
        t0 = time.time()
        emb = embeddings(ds, device)

        # full symmetric 4x4 Procrustes-disparity matrix
        mat = np.zeros((len(LABELS), len(LABELS)))
        for i, j in combinations(range(len(LABELS)), 2):
            d = ix.procrustes_disparity(emb[LABELS[i]], emb[LABELS[j]])
            mat[i, j] = mat[j, i] = d
            long_rows.append(
                {"dataset": ds.name, "a": LABELS[i], "b": LABELS[j], "procrustes": d}
            )

        mpath = out_dir / f"cross_procrustes_matrix_{ds.name}.csv"
        with mpath.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["", *LABELS])
            for i, lab in enumerate(LABELS):
                writer.writerow([lab, *[round(float(v), 4) for v in mat[i]]])

        print(f"  {ds.name}  (n={ds.n})  ({time.time() - t0:.1f}s)")
        for i, lab in enumerate(LABELS):
            cells = "  ".join(f"{mat[i, j]:.4f}" for j in range(len(LABELS)))
            print(f"    {lab:<16} {cells}")
        # the payoff comparison
        fw_pair = mat[LABELS.index("framework_tsne"), LABELS.index("framework_umap")]
        fw_ref_t = mat[LABELS.index("framework_tsne"), LABELS.index("reference_tsne")]
        fw_ref_u = mat[LABELS.index("framework_umap"), LABELS.index("reference_umap")]
        print(
            f"    -> fw_tsne~fw_umap = {fw_pair:.4f}  <  "
            f"fw_tsne~ref_tsne = {fw_ref_t:.4f},  fw_umap~ref_umap = {fw_ref_u:.4f}\n"
        )

    lpath = out_dir / "cross_procrustes.csv"
    with lpath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "a", "b", "procrustes"])
        writer.writeheader()
        writer.writerows(long_rows)
    print(f"Wrote {lpath}")


if __name__ == "__main__":
    main()
