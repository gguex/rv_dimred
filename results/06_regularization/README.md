# Kernel-space regularization experiments

This directory contains the results of the checks and experiments concerning
regularization through control of the neutral component.

## Available results

The `math_checks/` subdirectory contains the output of
[verify_kernel_regularization.py](../../scripts/exploratory/verify_kernel_regularization.py):

- [checks.json](math_checks/checks.json): numerical errors for the identities,
  gradients, spectral ceiling, and two-group separation counterexample;
- [trace_regularization.pdf](math_checks/trace_regularization.pdf) and
  [trace_regularization.png](math_checks/trace_regularization.png): illustration
  of the trace and penalty along dilation paths.

Regenerate them from the repository root with:

```sh
.venv/bin/python scripts/exploratory/verify_kernel_regularization.py
```

These results are mathematical checks rather than an evaluation of a
dimensionality-reduction algorithm. The script also retains checks for
exploratory variants that were not selected for the revision.

## Integration and controlled example

The planned experiment was run on September 18, 2026: three penalty strengths,
including zero, applied to the same three initializations. The
[report](controlled_example/README.md) describes the protocol, results, and
limitations. All nine runs are retained without selecting the best one.

The penalty moves the kernel inertia toward its target at the cost of lower RV.
The strongest penalty also produces distant points and sensitivity to
initialization. This example illustrates inertia control; it does not support a
general improvement in visualization quality or a convergence guarantee.

All five solver integration checks pass: Frobenius projection and forces,
unchanged trajectories at zero penalty, optimization of the penalized
objective, consistency of the final score and tracking, and parameter
validation. See the [test results](solver_checks.json).

```sh
.venv/bin/python scripts/exploratory/verify_regularized_solver.py
.venv/bin/python scripts/experiments/06_regularization/regularization_run.py
```

## MNIST comparison retained in the paper

The [mnist_hollow_regularization](mnist_hollow_regularization/) directory uses
exactly the same 2,000 observations, full-RV, hollow-RV, and t-SNE coordinates,
and trajectories as the earlier `04_tether` experiment. Starting from the saved
full-RV solution, four matched 500-step runs compare hollow RV, full RV at
`eta=100` and `eta=1000`, and regularized hollow RV at `eta=100000`.

```sh
.venv/bin/python scripts/experiments/06_regularization/mnist_hollow_regularization.py
```

## Preliminary MNIST path

The [mnist_path](mnist_path/README.md) directory retains the preliminary
visualization on a balanced sample of 500 MNIST images. It starts from the best
of three unregularized solutions, selected by RV, and then follows the
continuation path `eta = 1, 10, 100, 1000`. Maps on a common scale show the
progressive expansion and the trade-off with alignment. This experiment is no
longer used in the manuscript.

```sh
.venv/bin/python scripts/experiments/06_regularization/mnist_regularization_path.py
```

The exploratory volume/UMAP script is archived as
`archive/exploratory/verify_volume_repulsion.py`; its previous results were
removed and are not part of this directory.

## Hollow sweep and strong regularization

The [hollow_regularization_sweep](hollow_regularization_sweep/) directory
contains an experiment outside the manuscript that applies the regularized
hollow objective up to `eta=1000000`. All branches start from the same full-RV
state. Maps on common and individual scales, final metrics, and trajectories
show that the effect becomes visible around `eta=3000`, the target is reached
almost exactly at `eta=100000`, and `eta=1000000` further degrades alignment and
local fidelity.
