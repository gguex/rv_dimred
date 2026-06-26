"""
approximations_finetune_run.py  —  hollow-RV t-SNE / UMAP hyperparameter sweep
==============================================================================

New objective: the *hollow-RV* (both kernel diagonals zeroed), which frees the
embedding to spread into the neighbour-embedding regime. For every dataset this
sweeps the JOINT grid of

  * t-SNE : perplexity   × lambda   (lambda = the input-affinity softening exponent
                                     gamma, formerly fixed at 0.5)
  * UMAP  : n_neighbors  × lambda

The neighbour hyperparameter (perplexity / n_neighbors) is shared identically by
the framework input kernel AND the library reference (sklearn t-SNE / umap-learn);
lambda is framework-only (the reference has no softening), so the reference is
computed once per neighbour value and reused across lambda.

For every grid point it records Procrustes, kNN overlap, trustworthiness and ARI
(the last two for both framework and reference). No optimum is selected — the full
sweep is written for inspection:

  results/coordinates/approximations_finetune/   all embeddings (framework + ref)
  results/indices/approximations_finetune/       tidy CSV + one wide table / (dataset,method)

Figures are built from these by approximations_finetune_figures.py.
"""

from __future__ import annotations

import csv
import time

import numpy as np
import umap
from sklearn.manifold import TSNE

from src import indices as ix
from src.benchmark_common import (
    Q,
    SEED,
    TRUST_K,
    coord_path,
    coords_dir,
    get_device,
    indices_dir,
    pca_init,
    pca_init_sklearn,
    rv_embed,
    to_tensor,
)
from src.datasets import Dataset, load_all
from src.rv_kernels import (
    default_weights,
    fuzzy_topological_base,
    gaussian_affinity_base,
    soften_and_center,
)

FAMILY = "approximations_finetune"

# ── Sweep grids ───────────────────────────────────────────────────────────────
PERPLEXITIES = [5, 15, 30, 50, 100]
N_NEIGHBORS = [5, 15, 30, 50, 100]
LAMBDAS = [0.25, 0.5, 0.75, 1.0]  # input-affinity softening exponent gamma

TIDY_FIELDS = [
    "dataset",
    "method",
    "method_key",
    "hp_name",
    "hp_value",
    "lambda",
    "variant",
    "index_name",
    "value",
]


def tsne_ref(X: np.ndarray, init: np.ndarray, perplexity: int) -> np.ndarray:
    return TSNE(
        n_components=Q, perplexity=perplexity, init=init, random_state=SEED
    ).fit_transform(X)


def umap_ref(X: np.ndarray, init: np.ndarray, n_neighbors: int) -> np.ndarray:
    return np.asarray(
        umap.UMAP(
            n_neighbors=n_neighbors, n_components=Q, init=init, random_state=SEED
        ).fit_transform(X)
    )


# method spec: (name, key, hp_name, hp_values, base_fn, out_kernel, out_param, ref_fn)
METHODS = [
    (
        "t-SNE",
        "tsne",
        "perplexity",
        PERPLEXITIES,
        gaussian_affinity_base,
        "student_t",
        1.0,
        tsne_ref,
    ),
    (
        "UMAP",
        "umap",
        "n_neighbors",
        N_NEIGHBORS,
        fuzzy_topological_base,
        "umap",
        None,
        umap_ref,
    ),
]


def fw_key(mkey: str, hp: int, lam: float) -> str:
    return f"{mkey}_hp{hp}_g{lam:.2f}"


def ref_key(mkey: str, hp: int) -> str:
    return f"{mkey}_hp{hp}"


def main() -> None:
    device = get_device()
    print(f"device: {device}\n")
    datasets = load_all(random_state=SEED)
    rows: list[dict[str, object]] = []

    for ds in datasets.values():
        print("=" * 70)
        print(f"dataset = {ds.name}  (n={ds.n}, d={ds.X.shape[1]})")
        print("=" * 70)
        X_t = to_tensor(ds.X, device)
        w = default_weights(ds.n, device)
        init_fw = pca_init(ds.X)
        init_ref = pca_init_sklearn(ds.X)

        for name, mkey, hp_name, hp_vals, base_fn, out_k, out_p, ref_fn in METHODS:
            for hp in hp_vals:
                t0 = time.time()
                base = base_fn(X_t, hp)  # raw affinity, computed once per hp

                # reference (shared neighbour hp; independent of lambda)
                ref = np.asarray(ref_fn(ds.X, init_ref, hp), dtype=np.float32)
                np.save(coord_path(FAMILY, ds.name, ref_key(mkey, hp), "reference"), ref)
                _add_quality(rows, ds, name, mkey, hp_name, hp, "", "reference", ref)

                for lam in LAMBDAS:
                    K_in = soften_and_center(base, lam, w, device)
                    fw, rv = rv_embed(
                        K_in, init_fw, device, out_k, out_p, weights=w, hollow=True
                    )
                    fw = fw.astype(np.float32)
                    np.save(
                        coord_path(FAMILY, ds.name, fw_key(mkey, hp, lam), "framework"),
                        fw,
                    )
                    _add(rows, ds, name, mkey, hp_name, hp, lam, "framework",
                         "procrustes", ix.procrustes_disparity(fw, ref))
                    _add(rows, ds, name, mkey, hp_name, hp, lam, "framework",
                         "knn_overlap", ix.knn_overlap(fw, ref, k=TRUST_K))
                    _add(rows, ds, name, mkey, hp_name, hp, lam, "framework",
                         "rv_final", rv)
                    _add_quality(rows, ds, name, mkey, hp_name, hp, lam, "framework", fw)
                print(
                    f"  {name:<6} {hp_name}={hp:<4}  swept {len(LAMBDAS)} lambdas  "
                    f"({time.time() - t0:.1f}s)"
                )

    out_dir = indices_dir(FAMILY)
    tidy = out_dir / "approximations_finetune_tidy.csv"
    with tidy.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIDY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote tidy → {tidy}")
    write_tables(rows, out_dir)
    print(f"Saved coordinates → {coords_dir(FAMILY)}")


def _add(rows, ds, method, mkey, hp_name, hp, lam, variant, index_name, value) -> None:
    rows.append({
        "dataset": ds.name, "method": method, "method_key": mkey, "hp_name": hp_name,
        "hp_value": hp, "lambda": lam, "variant": variant, "index_name": index_name,
        "value": value,
    })


def _add_quality(rows, ds: Dataset, method, mkey, hp_name, hp, lam, variant, Y) -> None:
    _add(rows, ds, method, mkey, hp_name, hp, lam, variant,
         "trustworthiness", ix.trustworthiness(ds.X, Y, k=TRUST_K))
    if ds.has_labels:
        _add(rows, ds, method, mkey, hp_name, hp, lam, variant,
             "ari", ix.ari(Y, ds.labels))


# wide table per (dataset, method): one row per (hp, lambda), framework indices +
# the (lambda-independent) reference quality broadcast alongside.
TABLE_COLUMNS = [
    "hp_value", "lambda", "rv_final", "procrustes", "knn_overlap",
    "trustworthiness_fw", "ari_fw", "trustworthiness_ref", "ari_ref",
]
DEC = 4


def write_tables(rows: list[dict[str, object]], out_dir) -> None:
    fw: dict[tuple, dict[str, object]] = {}
    ref: dict[tuple, dict[str, object]] = {}
    keys: dict[tuple[str, str], set] = {}
    for r in rows:
        dm = (str(r["dataset"]), str(r["method_key"]))
        keys.setdefault(dm, set())
        if r["variant"] == "framework":
            cell = fw.setdefault((*dm, r["hp_value"], r["lambda"]), {})
            cell[str(r["index_name"])] = r["value"]
            keys[dm].add((r["hp_value"], r["lambda"]))
        else:  # reference: keyed by hp only
            ref.setdefault((*dm, r["hp_value"]), {})[str(r["index_name"])] = r["value"]

    def fmt(v) -> object:
        return round(float(v), DEC) if v not in ("", None) else ""

    for (dataset, method), hp_lams in keys.items():
        path = out_dir / f"table_{dataset}_{method}.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(TABLE_COLUMNS)
            for hp, lam in sorted(hp_lams, key=lambda t: (t[0], t[1])):
                c = fw.get((dataset, method, hp, lam), {})
                rq = ref.get((dataset, method, hp), {})
                writer.writerow([
                    hp, lam,
                    fmt(c.get("rv_final", "")),
                    fmt(c.get("procrustes", "")),
                    fmt(c.get("knn_overlap", "")),
                    fmt(c.get("trustworthiness", "")),
                    fmt(c.get("ari", "")),
                    fmt(rq.get("trustworthiness", "")),
                    fmt(rq.get("ari", "")),
                ])
        print(f"Wrote table → {path}")


if __name__ == "__main__":
    main()
