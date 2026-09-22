# Neutral-component control: synthetic example

Experiment run on September 18, 2026. Parameters were fixed before the first
run, without hyperparameter search or initialization selection.

## Objective and protocol

The solver maximizes full RV minus
`lambda * (Tr(K_Y) - t0)^2 / (2*(n-1))`. This penalty is half the squared
Frobenius distance between the neutral projection of `K_Y` and
`t0*K0/(n-1)`, where `K0 = I - sqrt(f)*sqrt(f)^T`.

For interpretable parameters, define `T = 1 - sum(f_i^2)`,
`rho = Tr(K_Y)/T`, `t0 = rho0*T`, and `lambda = eta*(n-1)/T^2`.
The objective becomes `RV - eta*(rho-rho0)^2/2`.

- 60 objects: three Gaussian groups of 20 points in R³, seed 20260918.
  Centers (-2,0,0), (2,0,1), and (0,3,-1), with isotropic standard deviation
  0.6. The data are centered and divided by the median distance between
  distinct pairs.
- Student-t input and output kernels with parameter nu=1, weighted centering,
  and uniform weights. The output dimension is 2. Full RV includes the diagonal.
- Illustrative target `rho0 = 0.65`, hence `t0 = 0.6391667`. This control target
  is a modeling choice rather than a learned optimum.
- Strengths `eta = 0, 1, 10`, corresponding to
  `lambda = 0, 61.01695, 610.16949`.
- Initializations `0.1*randn(60,2)` with seeds 0, 1, and 2. For each seed, all
  three strengths start from exactly the same coordinates.
- Adam with learning rate 0.03 for 1,500 iterations, without early stopping or
  iterate selection. Computation uses one CPU thread and float64. Measurements
  are recorded every 25 iterations.
- Labels are used only to color the figures.

Parameters, versions, and code hashes are recorded in [config.json](config.json).
The final score is recomputed from the coordinates returned by the solver. The
RMS radius is `sqrt(sum_i f_i * ||y_i - mean_f(Y)||²)`.

## Results

Empirical means ± standard deviations over the three initializations:

| eta | Unpenalized RV | Normalized inertia rho | Absolute error to rho0 | RMS radius |
|---|---:|---:|---:|---:|
| 0 | 0.99734 ± <0.00001 | 0.42542 ± <0.00001 | 0.22458 ± <0.00001 | 0.72415 ± <0.00001 |
| 1 | 0.99161 ± <0.00001 | 0.57576 ± <0.00001 | 0.07424 ± <0.00001 | 1.11908 ± <0.00001 |
| 10 | 0.96296 ± 0.01435 | 0.64895 ± 0.00640 | 0.00527 ± 0.00093 | 2.04396 ± 0.43318 |

The per-run absolute error should be examined because the mean rho can hide
deviations on opposite sides of the target.

The moderate penalty moves inertia toward the target with a small decrease in
RV. The three initializations yield nearly identical metrics. The strong
penalty approaches the target more closely, but lowers RV further and produces
configurations that vary with initialization. The figures contain distant
points; for seed 1, one group spreads substantially.

For eta=0 and eta=1, the final gradient norm is at most 2.5e-8. For eta=10, it
remains between 1.8e-4 and 4.7e-4; the objective still increases by 3.6e-5 to
2.4e-4 over the last 100 iterations, and the radii continue to grow. These are
therefore fixed-budget results rather than certified optima. The experiment
establishes neither divergence to infinity nor final convergence of these three
trajectories.

**Conclusion retained for the paper:** the construction controls kernel-space
inertia and exposes its trade-off with alignment. A trace near the target is
insufficient to guarantee a good Euclidean configuration. The penalty should
not be presented as a general visualization improvement or as a guarantee that
a finite optimum exists.

## Files and reproduction

- [summary.csv](summary.csv): nine results with traces, RV values, radii, and
  diagnostics.
- [aggregate.csv](aggregate.csv): means and standard deviations across seeds.
- [trajectories.csv](trajectories.csv): evolution of every run.
- [coordinates.npz](coordinates.npz): data, labels, weights, input kernel, three
  initializations, and nine final configurations.
- [trajectories.pdf](trajectories.pdf): inertia control, RV, and spread.
- [embeddings.pdf](embeddings.pdf): nine configurations on a common scale,
  centered for display without aligning rotations.

```sh
.venv/bin/python scripts/experiments/06_regularization/regularization_run.py
```

The script regenerates the result files in this directory with the parameters
above. RV uses the solver's existing numerical stabilization (`+1e-12` in the
denominator). The existing dense centering implementation is retained.
