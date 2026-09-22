# Hollow RV and regularization on the same MNIST sample

This is the experiment retained in the revised manuscript. It reuses exactly
the 2,000 observations and artifacts from the earlier `04_tether` figure: a
balanced MNIST sample with 200 images per digit, seed 0, perplexity 30,
softening 0.5, and PCA initialization.

## Comparisons

The figure reuses the historical full-RV and t-SNE coordinates without
modification. The full-RV solution after 500 steps is the unique common state
before any branch-specific update.

Four independent branches start from this identical coordinate array and each
run 500 Adam steps: hollow RV, full RV with `eta=100`, full RV with `eta=1000`,
and hollow RV with `eta=100000`. The target is `rho0=0.99`. The two evolution
panels compare hollow RV, strong regularization, and their combination. They
retain the two directly relevant quantities: normalized kernel-space inertia
and the coordinate RMS radius. RV curves are omitted because the branches do
not all optimize the same RV, and their final values already appear in the
titles.

| Configuration | Optimized RV | rho | RMS radius | Trustworthiness (k=15) |
|---|---:|---:|---:|---:|
| Common full-RV state | 0.67061 | 0.98260 | 12.88 | 0.95505 |
| Hollow RV | 0.75271 | 0.99395 | 24.16 | 0.95260 |
| Full RV, eta=100 | 0.67346 | 0.98631 | 16.80 | 0.95647 |
| Full RV, eta=1000 | 0.67188 | 0.98839 | 20.04 | 0.95502 |
| Hollow RV, eta=100000 | 0.73962 | 0.99000 | 17.13 | 0.94994 |

Hollow RV alone pushes inertia above the target. The penalty alone approaches
it from below; a stronger penalty combined with hollow RV moderates expansion
and brings inertia back toward the target. This experiment illustrates both
effects from the same state. It does not establish general superiority or a
comparison at equal penalty strength.

## Files

- `mnist_hollow_regularization.pdf` and `.png`: complete figure;
- `summary.csv`: metrics for all configurations;
- `trajectories.csv`: hollow and strongly regularized trajectories;
- `coordinates.npz`: data, input kernel, and coordinates;
- `config.json`: protocol, versions, and hashes of the historical artifacts.

Reproduce the experiment from the repository root with:

```sh
.venv/bin/python scripts/experiments/06_regularization/mnist_hollow_regularization.py
```
