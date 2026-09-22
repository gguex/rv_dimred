# Results map

This directory contains the coordinates, scalar indices, and figures produced by
the current experiment scripts. The main project description and theoretical
scope are in the [repository README](../README.md).

## Primary manuscript artifacts

| Directory | Contents | Reproduction |
|---|---|---|
| [`01_spectral/`](01_spectral/) | spectral coordinates, recovery indices, and RV ceiling checks | `uv run python scripts/experiments/01_spectral/spectral_run.py` |
| [`02_manifold_dim/`](02_manifold_dim/) | numerical differential-rank checks | `uv run python scripts/experiments/02_manifold_dim/manifold_dim_run.py` |
| [`03_forces/`](03_forces/) | force-identity checks and cross-Procrustes comparisons | `uv run python scripts/experiments/03_forces/forces_check.py` |
| [`05_supervised_dial/`](05_supervised_dial/) | supervised interpolation coordinates, train/test metrics, and curves | `uv run python scripts/experiments/05_supervised_dial/supervised_dial_run.py` |
| [`06_regularization/`](06_regularization/) | mathematical checks, controlled trials, and the retained MNIST comparison | `uv run python scripts/experiments/06_regularization/mnist_hollow_regularization.py` |
| [`07_geometry/`](07_geometry/) | schematic view of the linear cone and nonlinear local image | `uv run python scripts/figures/kernel_geometry_overview.py` |

The main regularization panel used in the revised manuscript is
[`06_regularization/mnist_hollow_regularization/mnist_hollow_regularization.pdf`](06_regularization/mnist_hollow_regularization/mnist_hollow_regularization.pdf).
The other regularization subdirectories document mathematical checks, controlled
examples, and parameter sweeps; their README files state which results are
exploratory.

## Historical and supplementary artifacts

- [`04_tether/`](04_tether/) is retained because the current regularization
  comparison reuses its exact MNIST sample, saved t-SNE reference, and full-RV
  starting coordinates. The revised manuscript does not use its former general
  diagonal-tether interpretation.
- [`figures/`](figures/) and the older top-level `coordinates/`, `indices/`, and
  `showcase/` trees contain supplementary gallery and parameter-sweep artifacts.
  They are not the authoritative map of the revised manuscript experiments.

Some older CSV files use names inherited from the original scripts:

- `framework_tsne` means the RV objective with a Student-t output kernel;
- `framework_umap` means the RV objective with a UMAP-shaped output profile;
- `reference_tsne` and `reference_umap` are the corresponding library methods.

These labels identify configurations. They do not claim equality with the t-SNE
or UMAP objectives.

## Shared protocol

- random seed: `0`;
- output dimension: `q = 2`;
- neighborhood size: `15` where applicable;
- t-SNE perplexity and adaptive-Gaussian perplexity: `30`;
- standard RV optimization: 500 Adam steps with learning rate `0.1`;
- datasets: balanced MNIST sample, processed PBMC3k single-cell data, and Swiss
  roll, depending on the experiment.

Each experiment records any deviations from these defaults in its script or
local metadata. The regularization experiments additionally save JSON
configuration files and full trajectories.

## Complete reproduction sequence

Run commands from the repository root after `uv sync`:

```bash
uv run python scripts/experiments/01_spectral/spectral_run.py
uv run python scripts/experiments/01_spectral/spectral_indices.py
uv run python scripts/experiments/01_spectral/ceiling_check.py
uv run python scripts/experiments/02_manifold_dim/manifold_dim_run.py
uv run python scripts/experiments/03_forces/forces_check.py
uv run python scripts/experiments/03_forces/cross_procrustes_run.py
uv run python scripts/experiments/04_tether/tether_run.py
uv run python scripts/experiments/04_tether/tether_figure.py
uv run python scripts/experiments/05_supervised_dial/supervised_dial_run.py
uv run python scripts/experiments/05_supervised_dial/supervised_dial_indices.py
uv run python scripts/experiments/05_supervised_dial/supervised_dial_figure.py
uv run python scripts/experiments/05_supervised_dial/dial_scatter_figure.py
uv run python scripts/exploratory/verify_kernel_regularization.py
uv run python scripts/exploratory/verify_regularized_solver.py
uv run python scripts/experiments/06_regularization/mnist_hollow_regularization.py
uv run python scripts/figures/kernel_geometry_overview.py
```

Large data dependencies may be downloaded by their dataset loaders on a first
run. The committed outputs allow the figures and reported indices to be inspected
without rerunning the expensive embeddings.
