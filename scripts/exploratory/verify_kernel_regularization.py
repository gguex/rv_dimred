"""Check intrinsic trace/Frobenius regularization and its limits.

Run: .venv/bin/python scripts/exploratory/verify_kernel_regularization.py
Numerical algebra checks and a counterexample; no performance benchmark.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "06_regularization" / "math_checks"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="kernel-reg-mpl-"))
torch.set_default_dtype(torch.float64)
errors: dict[str, float] = {}


def check(name: str, x: torch.Tensor, y: torch.Tensor) -> None:
    err = float((x - y).detach().abs().max())
    errors[name] = max(errors.get(name, 0.0), err)
    assert torch.allclose(x, y, atol=1e-9, rtol=1e-9), (name, err)


def grad(value: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(value, y, retain_graph=True)[0]


def rv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum() / (a.norm() * b.norm())


for seed in range(5):
    torch.manual_seed(seed)
    n = 7
    for uniform in (True, False):
        f = torch.ones(n) if uniform else torch.rand(n) + .1
        f = f / f.sum()
        h = torch.eye(n) - torch.ones(n, 1) * f
        q = f.sqrt()[:, None] * h
        k0 = torch.eye(n) - torch.outer(f.sqrt(), f.sqrt())
        t0 = .4 * (1 - f.square().sum())
        target_neutral = t0 / (n - 1) * k0
        lam = 2.5

        for scale in (.1, 2.):
            y = (scale * torch.randn(n, 2)).requires_grad_()
            delta = y[:, None] - y[None, :]
            g = 1 / (1 + delta.square().sum(-1))
            k = q @ g @ q.T
            t = k.trace()
            p0k = t / (n - 1) * k0
            k_perp = k - p0k
            r2 = .5 * (p0k - target_neutral).square().sum()
            check("projected_L2_trace", r2, (t - t0).square() / (2 * (n - 1)))
            nuclear = torch.linalg.svdvals(p0k - target_neutral).sum()
            check("projected_nuclear_L1_trace", nuclear, (t - t0).abs())
            check("full_L2_decomposition", .5 * (k - target_neutral).square().sum(),
                  r2 + .5 * k_perp.square().sum())
            manual_trace = 4 * f[:, None] * (
                f[None, :, None] * g.square()[:, :, None] * delta
            ).sum(1)
            check("trace_gradient", grad(t, y), manual_trace)
            check("L2_reg_force", grad(-lam * r2, y),
                  lam * (t0 - t) / (n - 1) * manual_trace)
            check("L1_reg_force", grad(-lam * (t - t0).abs(), y),
                  lam * torch.sign(t0 - t) * manual_trace)

            x = torch.randn(n, 4)
            target = q @ (x @ x.T) @ q.T
            check("RV_as_normalized_Frobenius_distance", 1 - rv(target, k),
                  .5 * (target / target.norm() - k / k.norm()).square().sum())

            b = k / torch.sqrt(torch.outer(f, f))
            diagonal = b.diagonal()
            recovered_g = 1 + b - (diagonal[:, None] + diagonal[None, :]) / 2
            d2 = 1 / recovered_g - 1
            decoded_linear = -.5 * q @ d2 @ q.T
            centered_y = h @ y
            expected_linear = q @ (y @ y.T) @ q.T
            check("decoded_linear_kernel", decoded_linear, expected_linear)
            check("decoded_nuclear_equals_radius", decoded_linear.trace(),
                  (f[:, None] * centered_y.square()).sum())
            check("decoded_trace_gradient", grad(decoded_linear.trace(), y),
                  2 * f[:, None] * centered_y)
            scatter = centered_y.T @ (f[:, None] * centered_y)
            check("decoded_Frobenius_gradient", grad(decoded_linear.square().sum(), y),
                  4 * f[:, None] * (centered_y @ scatter))

        # The proof covers positive and indefinite targets with positive eigenvalues.
        x = torch.randn(n, 5)
        base_target = q @ (x @ x.T) @ q.T
        for shift in (0., .05):
            target = base_target - shift * k0
            vals, vectors = torch.linalg.eigh(target)
            clipped = vals[-2:].clamp_min(0)
            assert clipped.sum() > 0
            kq = (vectors[:, -2:] * clipped) @ vectors[:, -2:].T
            kstar = t0 / kq.trace() * kq
            ceiling = (clipped.square().sum() / vals.square().sum()).sqrt()
            check("linear_trace_anchor", kstar.trace(), t0)
            check("linear_alignment_ceiling_preserved", rv(target, kstar), ceiling)

# Exact two-block counterexample to finite attainment for any finite trace penalty.
n = 4
h_np = np.eye(n) - np.ones((n, n)) / n
block_g = np.array([[1., 1., 0., 0.], [1., 1., 0., 0.],
                    [0., 0., 1., 1.], [0., 0., 1., 1.]])
boundary_k = h_np @ block_g @ h_np / n
t0_np = float(np.trace(boundary_k))
rows = []
for s in (.1, 1., 10., 100.):
    y = s * np.array([-1., -1., 1., 1.])
    g = 1 / (1 + (y[:, None] - y[None, :]) ** 2)
    k = h_np @ g @ h_np / n
    factor = 4 * s * s / (1 + 4 * s * s)
    check(
        "split_counterexample_kernel",
        torch.tensor(k),
        torch.tensor(factor * boundary_k),
    )
    alignment = float(
        (k * boundary_k).sum() / np.linalg.norm(k) / np.linalg.norm(boundary_k)
    )
    check("split_counterexample_RV", torch.tensor(alignment), torch.tensor(1.))
    trace = float(np.trace(k))
    rows.append({"s": s, "trace": trace, "RV": alignment,
                 "L2_regularized_objective_lambda_1":
                     alignment - (trace - t0_np)**2 / 6,
                 "linear_kernel_trace": s*s})

report = {"maximum_absolute_errors": errors, "split_counterexample": rows,
          "trace_target_counterexample": t0_np}
(OUT / "checks.json").write_text(json.dumps(report, indent=2) + "\n")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

scales = np.logspace(-2, 2, 301)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for label, raw_y in (("Four distinct points", [0., 1., 2., 4.]),
                     ("Two coincident pairs", [-1., -1., 1., 1.])):
    y = np.array(raw_y)
    d2 = (y[:, None] - y[None, :]) ** 2
    traces = np.array([np.trace(h_np @ (1 / (1 + s*s*d2)) @ h_np / n) for s in scales])
    axes[0].semilogx(scales, traces, label=label)
    axes[1].loglog(scales, (traces - t0_np)**2 / (2*(n-1)), label=label)
axes[0].axhline(t0_np, color="gray", ls="--", label="Target trace = 0.5")
axes[0].set(xlabel="Dilation scale", ylabel="Trace of the centered kernel",
            title="Trace depends on scale and configuration")
axes[1].set(xlabel="Dilation scale", ylabel="Projected Frobenius penalty",
            title="A trace penalty alone does not prevent escape")
for ax in axes:
    ax.legend(fontsize=8)
fig.savefig(OUT / "trace_regularization.pdf")
fig.savefig(OUT / "trace_regularization.png", dpi=170)
plt.close(fig)
print(json.dumps(report, indent=2))
