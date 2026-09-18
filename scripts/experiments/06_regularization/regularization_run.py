"""One fixed synthetic illustration of neutral-component L2 regularization.

Run: .venv/bin/python scripts/experiments/06_regularization/regularization_run.py
No parameter search: three strengths and three paired initializations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="rv-reg-mpl-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.rv_kernels import (  # noqa: E402
    compute_student_t_kernel_torch as student,
)
from src.rv_kernels import (  # noqa: E402
    kernel_trace_penalty,
    rv_coefficient,
    rv_dimred,
)

OUT = ROOT / "results/06_regularization/controlled_example"
CONFIG = {
    "dataset": "three isotropic Gaussian groups in R3, 20 objects per group",
    "data_seed": 20260918,
    "centers": [[-2, 0, 0], [2, 0, 1], [0, 3, -1]],
    "noise_std": 0.6,
    "input_scaling": "center, then divide by median pairwise Euclidean distance",
    "input_kernel": "weighted centered Student-t, nu=1",
    "output_kernel": "weighted centered Student-t, nu=1",
    "weights": "uniform",
    "q": 2,
    "rho_target": 0.65,
    "eta_values": [0.0, 1.0, 10.0],
    "initialization_seeds": [0, 1, 2],
    "initialization": "0.1 * torch.randn(n, q), shared across strengths within seed",
    "optimizer": "Adam, default betas and eps, no weight decay",
    "lr": 0.03,
    "n_iter": 1500,
    "stopping_rule": "fixed budget, no early stopping or best-iterate selection",
    "record_every": 25,
    "dtype": "float64",
    "device": "cpu",
    "torch_threads": 1,
    "parameter_conversion": "T=1-sum(f_i^2); t0=rho_target*T; lambda=eta*(n-1)/T^2",
    "scope": "illustration of inertia control, not a visualization quality benchmark",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(CONFIG["torch_threads"])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(CONFIG["data_seed"])
    labels = np.repeat(np.arange(3), 20)
    x = np.array(CONFIG["centers"])[labels] + CONFIG["noise_std"] * rng.normal(
        size=(60, 3)
    )
    x -= x.mean(axis=0)
    distances = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    scale = np.median(distances[np.triu_indices(len(x), k=1)])
    x /= scale
    f = torch.ones(len(x)) / len(x)
    kx = student(torch.tensor(x), weights=f)
    bound = float(1 - f.square().sum())
    target = CONFIG["rho_target"] * bound
    config = dict(
        CONFIG,
        n=len(x),
        input_scale=float(scale),
        trace_bound=bound,
        trace_target=target,
        input_trace=float(kx.trace()),
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        source_hashes={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), ROOT / "src/rv_kernels.py")
        },
    )
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    arrays = {"X": x, "labels": labels, "weights": f.numpy(), "K_X": kx.numpy()}
    trajectories, summary = [], []

    def metrics(y: torch.Tensor, strength: float) -> dict[str, float]:
        with torch.no_grad():
            k = student(y, weights=f)
            rv = float(rv_coefficient(kx, k))
            trace = float(k.trace())
            penalty = float(strength * kernel_trace_penalty(k, target))
            centered = y - (f[:, None] * y).sum(0)
            radius = float((f * centered.square().sum(1)).sum().sqrt())
        return {
            "rv": rv,
            "trace": trace,
            "rho": trace / bound,
            "abs_rho_error": abs(trace / bound - CONFIG["rho_target"]),
            "rms_radius": radius,
            "penalty": penalty,
            "objective": rv - penalty,
        }

    for seed in CONFIG["initialization_seeds"]:
        torch.manual_seed(seed)
        init = 0.1 * torch.randn(len(x), CONFIG["q"])
        arrays[f"init_seed{seed}"] = init.numpy()
        for eta in CONFIG["eta_values"]:
            strength = eta * (len(x) - 1) / bound**2
            run_rows = []

            def record(step: int, y: torch.Tensor) -> None:
                if step % CONFIG["record_every"] == 0 or step == CONFIG["n_iter"]:
                    row = {
                        "seed": seed,
                        "eta": eta,
                        "step": step,
                        **metrics(y, strength),
                    }
                    run_rows.append(row)
                    trajectories.append(row)

            y, reported_rv = rv_dimred(
                kx,
                weights=f,
                init=init,
                q=CONFIG["q"],
                n_iter=CONFIG["n_iter"],
                lr=CONFIG["lr"],
                trace_target=target,
                trace_strength=strength,
                callback=record,
            )
            final = metrics(y, strength)
            assert abs(reported_rv - final["rv"]) < 1e-12
            assert all(np.isfinite(value) for value in final.values())
            variable = y.clone().requires_grad_()
            k = student(variable, weights=f)
            objective = rv_coefficient(kx, k) - strength * kernel_trace_penalty(
                k, target
            )
            gradient = torch.autograd.grad(objective, variable)[0]
            final["gradient_norm"] = float(gradient.norm())
            final["objective_change_last_100"] = (
                final["objective"] - run_rows[-5]["objective"]
            )
            summary.append({"seed": seed, "eta": eta, "lambda": strength, **final})
            arrays[f"Y_seed{seed}_eta{eta:g}"] = y.numpy()
            print(
                f"seed={seed} eta={eta:g}: RV={final['rv']:.6f}, "
                f"rho={final['rho']:.6f}, radius={final['rms_radius']:.4f}, "
                f"|grad|={final['gradient_norm']:.2e}",
                flush=True,
            )

    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "trajectories.csv", trajectories)
    np.savez_compressed(OUT / "coordinates.npz", **arrays)
    aggregates = []
    for eta in CONFIG["eta_values"]:
        group = [r for r in summary if r["eta"] == eta]
        row = {"eta": eta}
        for name in ("rv", "rho", "abs_rho_error", "rms_radius", "gradient_norm"):
            values = [r[name] for r in group]
            row[f"{name}_mean"] = float(np.mean(values))
            row[f"{name}_std"] = float(np.std(values, ddof=1))
        aggregates.append(row)
    write_csv(OUT / "aggregate.csv", aggregates)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained")
    colors = ["#555555", "#167d9a", "#c66b24"]
    for eta, color in zip(CONFIG["eta_values"], colors):
        for seed in CONFIG["initialization_seeds"]:
            rows = [r for r in trajectories if r["eta"] == eta and r["seed"] == seed]
            for ax, name in zip(axes, ("rho", "rv", "rms_radius")):
                ax.plot(
                    [r["step"] for r in rows],
                    [r[name] for r in rows],
                    color=color,
                    alpha=0.7,
                    lw=1.3,
                    label=f"eta = {eta:g}" if seed == 0 else None,
                )
    axes[0].axhline(CONFIG["rho_target"], color="black", ls=":", lw=1, label="target")
    for ax, title in zip(
        axes, ("Normalized kernel inertia", "Unpenalized RV", "Euclidean RMS radius")
    ):
        ax.set(xlabel="Adam steps", title=title)
        ax.grid(alpha=0.15)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"trajectories.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        3, 3, figsize=(9, 8), sharex=True, sharey=True, layout="constrained"
    )
    for i, seed in enumerate(CONFIG["initialization_seeds"]):
        for j, eta in enumerate(CONFIG["eta_values"]):
            y = arrays[f"Y_seed{seed}_eta{eta:g}"]
            y = y - y.mean(axis=0)
            axes[i, j].scatter(*y.T, c=labels, cmap="viridis", s=13)
            axes[i, j].set_aspect("equal", adjustable="box")
            axes[i, j].set_title(f"seed {seed}, eta {eta:g}", fontsize=10)
    fig.suptitle("Shared coordinate scale; group labels used only for color")
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"embeddings.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
