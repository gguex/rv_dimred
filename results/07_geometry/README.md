# Conceptual kernel-geometry figure

This figure provides the overview added to the revised manuscript. It is fully
schematic and does not represent a numerical experiment.

The two panels show:

1. a cross-section of the cone of realizable linear kernels, with the target,
   the nearest rank-constrained kernel, and the angle associated with RV;
2. a local sheet in the image of a nonlinear readout, with the ambient gradient
   and the tangent velocity actually induced by a coordinate update.

The separate rays in the first panel recall that the rank-constrained cone is
generally non-convex. The second panel deliberately avoids drawing the
parametric update as an orthogonal projection.

Generated files:

- `kernel_geometry_overview.pdf`, the vector version used in the paper;
- `kernel_geometry_overview.png`, the visual-check version.

Reproduce the figure from the repository root with:

```sh
.venv/bin/python scripts/figures/kernel_geometry_overview.py
```
