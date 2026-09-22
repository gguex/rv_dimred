# Exploratory sweep: hollow RV with neutral regularization

This experiment is not included in the manuscript. It uses exactly the same
MNIST sample (`n=2000`), input kernel, and initial full-RV state as the paper's
figure. Each run restarts independently from this state and maximizes hollow RV
with `eta = 0, 100, 1000, 3000, 10000, 100000, 1000000` for 500 Adam steps. The
target is `rho0 = 0.99`.

## Results

| eta | rho | Error to target | RMS radius | Hollow RV | Trustworthiness |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9939467 | 0.0039467 | 24.16 | 0.75271 | 0.95260 |
| 100 | 0.9937761 | 0.0037761 | 23.78 | 0.75262 | 0.95214 |
| 1,000 | 0.9922495 | 0.0022495 | 20.96 | 0.74919 | 0.95199 |
| 3,000 | 0.9909756 | 0.0009756 | 19.03 | 0.74541 | 0.95320 |
| 10,000 | 0.9902966 | 0.0002966 | 18.06 | 0.74282 | 0.95293 |
| 100,000 | 0.9900045 | 0.0000045 | 17.13 | 0.73959 | 0.94973 |
| 1,000,000 | 0.9899878 | 0.0000122 | 16.34 | 0.72136 | 0.94003 |

`eta=100` has little visible effect: its values are close to those of
unregularized hollow RV. The effect becomes clear around `eta=3000`. Between
`eta=10000` and `eta=100000`, the penalty places inertia very near the target
while retaining a visual structure close to hollow RV. At `eta=1000000`, the
penalty dominates, hollow RV and trustworthiness decrease further, and
optimization becomes stiffer; this value offers no practical benefit here.

Maps on a common scale primarily show contraction. Separately rescaled maps
show that the shape changes gradually and then becomes more compact under very
strong regularization.

## Figures

- `embeddings_common_scale.pdf`: seven embeddings on a common scale;
- `embeddings_autoscaled.pdf`: the same embeddings, each filling its panel;
- `final_metrics.pdf`: inertia, target error, radius, and hollow RV by `eta`;
- `trajectories.pdf`: evolution of the seven optimization runs.

The PNG versions, `summary.csv`, `trajectories.csv`, `coordinates.npz`, and
`config.json` are stored in the same directory.

## Reproduction

```sh
.venv/bin/python scripts/experiments/06_regularization/hollow_regularization_sweep.py
```
