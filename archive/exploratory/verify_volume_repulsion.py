"""Numerical checks for the mathematical note on volume, RV and UMAP.

Archived exploration; not part of the active revision plan.
Run from the repository root: .venv/bin/python archive/exploratory/verify_volume_repulsion.py
These checks validate identities, not embedding performance or bibliographic novelty.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
torch.set_default_dtype(torch.float64)
errors: dict[str, float] = {}


def check(name: str, left: torch.Tensor, right: torch.Tensor) -> None:
    error = float((left - right).detach().abs().max())
    errors[name] = max(errors.get(name, 0.0), error)
    assert torch.allclose(left, right, atol=1e-9, rtol=1e-9), (name, error)


def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * (p.log() - q.log())).sum()


def bern(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return p * (p.log() - q.log()) + (1 - p) * (
        torch.log1p(-p) - torch.log1p(-q)
    )


def gradient(loss: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(loss, y, retain_graph=True)[0]


for seed in range(5):
    torch.manual_seed(seed)
    n = 7
    y = torch.randn(n, 2, requires_grad=True)
    delta = y[:, None, :] - y[None, :, :]
    g = 1 / (1 + delta.square().sum(-1))
    mask = ~torch.eye(n, dtype=torch.bool)
    one = torch.ones(n)
    for uniform in (True, False):
        f = one / n if uniform else torch.rand(n) + 0.1
        f = f / f.sum()
        h = torch.eye(n) - torch.outer(one, f)
        q = f.sqrt()[:, None] * h
        k = q @ g @ q.T
        centered_g = k / torch.sqrt(torch.outer(f, f))
        d = centered_g.diagonal()
        reconstructed = centered_g + 1 - (d[:, None] + d[None, :]) / 2
        check("affinity_reconstruction", g, reconstructed)
        mass_f = (torch.outer(f, f) * g)[mask].sum()
        check("weighted_mass_trace", mass_f, 1 - f.square().sum() - k.trace())
        check("diagonal_mass_variance", k.diagonal().square().sum(),
              k.trace().square() / n + (k.diagonal() - k.trace() / n).square().sum())
        k0 = torch.eye(n) - torch.outer(f.sqrt(), f.sqrt())
        k_perp = k - k.trace() / (n - 1) * k0
        check("neutral_trace_decomposition", k.square().sum(),
              k.trace().square() / (n - 1) + k_perp.square().sum())
        if uniform:
            z = g[mask].sum()
            check("unweighted_mass_trace", z, n * (n - 1) - n * n * k.trace())
            force = 4 / z * (g.square()[:, :, None] * delta).sum(1)
            check("tsne_repulsion_gradient", gradient(-z.log(), y), force)
            check("trace_repulsion_gradient", gradient(-z.log(), y),
                  n * n / z * gradient(k.trace(), y))

        # All pair sums are over ordered pairs, and coefficients are symmetric.
        a = torch.rand(n, n) + 0.1
        a = (a + a.T) / 2
        b = torch.rand(n, n) + 0.1
        b = (b + b.T) / 2
        av, bv = a[mask], b[mask]
        w = av + bv
        mu = av / w
        nu = g[mask]
        nu_reconstructed = reconstructed[mask]
        direct = (w * bern(mu, nu)).sum()
        via_k = (w * bern(mu, nu_reconstructed)).sum()
        check("umap_loss_via_centered_kernel", direct, via_k)
        check("umap_gradient_via_centered_kernel", gradient(direct, y), gradient(via_k, y))

        total = w.sum()
        pos = av.sum()
        fitted_pos = (w * nu).sum()
        p_pos = av / pos
        q_pos = w * nu / fitted_pos
        p_neg = bv / (total - pos)
        q_neg = w * (1 - nu) / (total - fitted_pos)
        decomposition = (
            pos * kl(p_pos, q_pos)
            + (total - pos) * kl(p_neg, q_neg)
            + total * bern(pos / total, fitted_pos / total)
        )
        check("bernoulli_mass_conditionals", direct, decomposition)
        check("bernoulli_mass_conditionals_gradient", gradient(direct, y),
              gradient(decomposition, y))

        entropy_mu = (w * (mu * mu.log() + (1 - mu) * torch.log1p(-mu))).sum()
        entropy_nu = (w * (nu * nu.log() + (1 - nu) * torch.log1p(-nu))).sum()
        bregman = entropy_mu - entropy_nu - (
            w * (nu.log() - torch.log1p(-nu)) * (mu - nu)
        ).sum()
        check("bernoulli_bregman", direct, bregman)

        base = bv / bv.sum()
        mean_affinity = (base * nu).sum()
        conditional_nonedge = base * (1 - nu) / (1 - mean_affinity)
        repulsion = -(base * torch.log1p(-nu)).sum()
        rep_decomposition = -torch.log1p(-mean_affinity) + kl(base, conditional_nonedge)
        check("repulsion_mass_heterogeneity", repulsion, rep_decomposition)

        edge_force = torch.zeros_like(g)
        edge_force[mask] = -av * nu + bv * nu.square() / (1 - nu)
        manual = 4 * (edge_force[:, :, None] * delta).sum(1)
        log_likelihood = (av * nu.log() + bv * torch.log1p(-nu)).sum()
        check("bernoulli_signed_forces", gradient(log_likelihood, y), manual)

# Same-volume, different-repulsion examples using actual 1-D configurations.
examples = []
for name, y_raw in (("evenly_spaced", [0., 1., 2.]), ("close_pair", [0., .01, 2.])):
    y = np.array(y_raw)
    d2 = (y[:, None] - y[None, :]) ** 2
    mask_np = ~np.eye(3, dtype=bool)
    def mass(scale: float) -> float:
        return float((1 / (1 + scale * scale * d2))[mask_np].mean())
    scale = brentq(lambda s: mass(s) - 0.4, 1e-6, 1e6)
    nu = (1 / (1 + scale * scale * d2))[mask_np]
    rep = float(-np.log1p(-nu).mean())
    examples.append({"shape": name, "scale": scale, "mean_affinity": mass(scale),
                     "repulsive_cost": rep, "global_mass_cost": float(-np.log(.6)),
                     "heterogeneity_cost": rep + float(np.log(.6))})

# A dilation counterexample: target is the neutral centered kernel.
y = np.array([0., 1., 3.])
d2 = (y[:, None] - y[None, :]) ** 2
h = np.eye(3) - np.ones((3, 3)) / 3
mask_np = ~np.eye(3, dtype=bool)
scales = np.logspace(-3, 3, 241)
curves = []
target_mass = 0.4
for scale in scales:
    g = 1 / (1 + scale * scale * d2)
    k = h @ g @ h / 3
    rv = float((h * k).sum() / np.linalg.norm(h) / np.linalg.norm(k))
    nu = g[mask_np]
    r = float(nu.mean())
    d_bern = target_mass * np.log(target_mass / r) + (
        1 - target_mass
    ) * np.log((1 - target_mass) / (1 - r))
    curves.append((rv - np.log(nu.sum()), rv + np.log1p(-nu).mean(), rv - d_bern))

out = ROOT / "archive" / "exploratory" / "results" / "volume_repulsion_math"
out.mkdir(parents=True, exist_ok=True)
report = {"maximum_absolute_errors": errors, "same_volume_examples": examples,
          "dilation_counterexample": {
              "scale_min": float(scales[0]), "objectives_at_scale_min": curves[0],
              "scale_max": float(scales[-1]), "objectives_at_scale_max": curves[-1],
              "columns": ["RV_minus_log_Z", "RV_plus_UMAP_repulsion", "RV_minus_mass_KL"]}}
(out / "checks.json").write_text(json.dumps(report, indent=2) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
values = np.array(curves)
for j, label in enumerate((r"RV $-\log Z$", "RV + mean log(1-affinity)", "RV - mass KL")):
    axes[0].semilogx(scales, values[:, j], label=label)
axes[0].axhline(1, color="gray", lw=.8, ls="--")
axes[0].set(xlabel="Dilation scale", ylabel="Objective to maximize",
            title="Bounded does not imply a finite maximizer")
axes[0].legend(fontsize=8)
labels = ["Evenly spaced", "Close pair"]
axes[1].bar(labels, [e["global_mass_cost"] for e in examples], label="Global mass term")
axes[1].bar(labels, [e["heterogeneity_cost"] for e in examples],
            bottom=[e["global_mass_cost"] for e in examples], label="Heterogeneity term")
axes[1].set(ylabel="Mean repulsive cost: -log(1-affinity)",
            title="Same mean affinity (0.4), different repulsion")
axes[1].legend(fontsize=8)
fig.savefig(out / "volume_repulsion.pdf")
fig.savefig(out / "volume_repulsion.png", dpi=170)
plt.close(fig)
print(json.dumps(report, indent=2))
