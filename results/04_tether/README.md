# Historical full-RV and hollow-RV baseline

This directory is retained for reproducibility. The current regularization
experiment reuses its exact MNIST sample, saved t-SNE reference, and full-RV
coordinates as a common initial state.

The comparison records full-RV and hollow-RV trajectories and their diagonal and
hollow kernel energies. The observed coordinate spreads belong to this numerical
setup. The revised manuscript does not present them as a general diagonal-tether
mechanism, and hollow RV is not identified with the t-SNE objective.

To regenerate the historical artifacts:

```bash
uv run python scripts/experiments/04_tether/tether_run.py
uv run python scripts/experiments/04_tether/tether_figure.py
```
