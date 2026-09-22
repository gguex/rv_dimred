# Kernel t-SNE regularization path on MNIST

This experiment shows the progressive effect of the L² penalty on the neutral
component of the output kernel. It uses 500 MNIST images, namely the first 50
available images of each digit in the test file.

## Protocol

The construction called "Kernel t-SNE" here combines the manuscript's t-SNE
input kernel (adaptive Gaussian affinities, perplexity 30, softening 0.5) with a
Student-t output kernel. The alignment objective is full RV. It is therefore a
method within the proposed framework rather than the KL loss of classical
t-SNE.

The dimensionless penalty is

`eta/2 * (rho - rho0)^2`, with `rho = Tr(K_Y)/(1-sum(f_i^2))`,

and the interior target is `rho0 = 0.99`. The five levels are
`eta = 0, 1, 10, 100, 1000`.

Three slightly perturbed PCA initializations are first optimized without
regularization for 2,000 steps. Seed 1, which gives the highest RV, is selected
without consulting the labels and then refined for another 3,000 steps. Each
positive level starts from the previous level's solution and uses 1,000 Adam
steps. The learning rate is 0.1. Computation uses float32 on CPU.

## Results

| eta | RV | rho | RMS radius | ARI | Trustworthiness k=15 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.7008 | 0.9564 | 7.83 | 0.435 | 0.932 |
| 1 | 0.7016 | 0.9573 | 8.07 | 0.447 | 0.932 |
| 10 | 0.7010 | 0.9617 | 9.70 | 0.402 | 0.931 |
| 100 | 0.6924 | 0.9734 | 17.71 | 0.388 | 0.925 |
| 1000 | 0.6611 | 0.9838 | 27.94 | 0.425 | 0.919 |

The main figure uses the same scale for all five maps. It shows that increasing
kernel inertia produces progressive group expansion. Small values change the
geometry very little. From `eta=100`, expansion becomes clear and RV begins to
decrease noticeably.

The slightly higher RV at `eta=1` is not evidence of a general improvement due
to the penalty. The objective is non-convex: the small perturbation can move the
continuation path into a basin with a slightly better unpenalized RV. ARI does
not vary monotonically and was not used to choose parameters. The supported
conclusion is that the penalty controls inertia and spread, with a growing
trade-off in alignment and neighborhood preservation.

## Files

- `mnist_regularization_path.pdf/png`: five maps on a common scale and the
  inertia-alignment trade-off;
- `mnist_regularization_diagnostics.pdf/png`: internal trajectories of inertia,
  RV, and radius at each level;
- `summary.csv`: final metrics;
- `trajectories.csv`: measurements every ten iterations;
- `baseline_candidates.csv`: selection of the unregularized starting point;
- `coordinates.npz`: data, labels, input kernel, and coordinates;
- `config.json`: complete protocol, versions, and source hashes.

Reproduce the experiment from the repository root with:

```sh
.venv/bin/python scripts/experiments/06_regularization/mnist_regularization_path.py
```
