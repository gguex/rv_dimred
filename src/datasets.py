"""
datasets.py
===========

Loaders for the three §5 case-study datasets, with the preprocessing fixed by
the global protocol (§0.1) and a single ``random_state`` for reproducibility.

Each loader returns a :class:`Dataset`:

    name        short id for figure/CSV names ("singlecell", "mnist", "swissroll")
    X           (n, d) float32 feature matrix fed to every input kernel
                (PCA->50 for MNIST / single-cell, raw 3D for Swiss-roll)
    labels      (n,) int class labels, or ``None`` for Swiss-roll
    label_names list of human-readable class names (labelled datasets only)
    color       (n,) array used to colour scatterplots (class id or roll angle)
    label_type  "discrete" (labelled) or "continuous" (Swiss-roll)

``has_labels`` is a convenience property gating ARI, the class kernel and the
unsupervised-supervised hybrid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PCA_DIMS = 50  # working representation before any kNN (§0.1)


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    color: np.ndarray
    label_type: str  # "discrete" | "continuous"
    labels: np.ndarray | None = None
    label_names: list[str] | None = field(default=None)

    @property
    def has_labels(self) -> bool:
        return self.labels is not None

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def cmap(self) -> str:
        return "tab10" if self.label_type == "discrete" else "viridis"


# ---------------------------------------------------------------------------
# MNIST  (balanced subsample, flatten, PCA -> 50)
# ---------------------------------------------------------------------------
def load_mnist(
    n_per_digit: int = 200, pca_dims: int = PCA_DIMS, random_state: int = 0
) -> Dataset:
    raw = np.genfromtxt(DATA_DIR / "mnist_test.csv", delimiter=",", skip_header=1)
    raw = np.vstack([raw[raw[:, 0] == d][:n_per_digit] for d in range(10)])
    images = (raw[:, 1:] / 255.0).astype(np.float32)
    labels = raw[:, 0].astype(int)
    X = PCA(n_components=pca_dims, random_state=random_state).fit_transform(images)
    X = X.astype(np.float32)
    return Dataset(
        name="mnist",
        X=X,
        labels=labels,
        label_names=[str(d) for d in range(10)],
        color=labels.astype(float),
        label_type="discrete",
    )


# ---------------------------------------------------------------------------
# Swiss-roll  (raw 3D, continuous roll-angle ground truth, no discrete labels)
# ---------------------------------------------------------------------------
def load_swiss_roll(
    n: int = 2000, noise: float = 0.0, random_state: int = 0
) -> Dataset:
    X, t = make_swiss_roll(n_samples=n, noise=noise, random_state=random_state)
    return Dataset(
        name="swissroll",
        X=X.astype(np.float32),
        labels=None,
        label_names=None,
        color=t.astype(float),  # roll angle (continuous manifold parameter)
        label_type="continuous",
    )


# ---------------------------------------------------------------------------
# Single-cell  (10x PBMC3k, scanpy-processed: log-norm + HVG + PCA->50 already
# done by scanpy; cell-type labels from the louvain annotation)
# ---------------------------------------------------------------------------
def load_single_cell(pca_dims: int = PCA_DIMS, random_state: int = 0) -> Dataset:
    import scanpy as sc

    sc.settings.datasetdir = str(DATA_DIR / "scanpy")
    sc.settings.verbosity = 0
    ad = sc.datasets.pbmc3k_processed()

    X = np.asarray(ad.obsm["X_pca"][:, :pca_dims], dtype=np.float32)
    cell_types = ad.obs["louvain"].astype("category")
    label_names = list(cell_types.cat.categories)
    labels = cell_types.cat.codes.to_numpy().astype(int)
    return Dataset(
        name="singlecell",
        X=X,
        labels=labels,
        label_names=label_names,
        color=labels.astype(float),
        label_type="discrete",
    )


LOADERS = {
    "singlecell": load_single_cell,
    "mnist": load_mnist,
    "swissroll": load_swiss_roll,
}


def load_all(random_state: int = 0) -> dict[str, Dataset]:
    return {name: fn(random_state=random_state) for name, fn in LOADERS.items()}
