"""Exploratory sweep of neutral regularization under the hollow-RV objective.

This experiment is deliberately separate from the manuscript figure. Every run
starts from the same saved full-RV MNIST embedding and changes only eta.

Run:
    .venv/bin/python \
        scripts/experiments/06_regularization/hollow_regularization_sweep.py
"""

# ruff: noqa: E402, I001
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="rv-hollow-sweep-"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.benchmark_common import PERPLEXITY, SEED, SOFTENING, normalize_kernel
from src.datasets import load_mnist
from src.rv_kernels import (
    default_weights,
    gaussian_affinity_base,
    rv_coefficient,
    rv_dimred,
)

OLD = ROOT / "results/04_tether"
OUT = ROOT / "results/06_regularization/hollow_regularization_sweep"
CONFIG: dict[str, Any] = {
    "scope": "exploratory; not included in the manuscript",
    "dataset": "same balanced MNIST sample as experiment 04",
    "n_per_digit": 200,
    "sample_size": 2000,
    "sample_seed": SEED,
    "perplexity": PERPLEXITY,
    "softening": SOFTENING,
    "common_start": "saved full-RV solution after 500 steps",
    "objective": "hollow RV minus neutral-component quadratic penalty",
    "rho_target": 0.99,
    "eta_values": [0.0, 100.0, 1000.0, 3000.0, 10000.0, 100000.0, 1000000.0],
    "iterations_per_eta": 500,
    "record_every": 10,
    "lr": 0.1,
    "dtype": "float32",
    "device": "cpu",
    "torch_threads": 4,
    "parameter_conversion": "T=1-1/n; t0=rho_target*T; lambda=eta*(n-1)/T^2",
}


def uniform_center(g: torch.Tensor) -> torch.Tensor:
    row_mean = g.mean(1, keepdim=True)
    return (g - row_mean - row_mean.T + g.mean()) / g.shape[0]


def uniform_student_kernel(
    coords: np.ndarray | torch.Tensor,
    param: Any = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    del param
    y = torch.as_tensor(coords, dtype=torch.float32, device=device)
    n = y.shape[0]
    if weights is not None:
        expected = torch.full_like(weights, 1.0 / n)
        if not torch.allclose(weights, expected, atol=1e-7, rtol=1e-6):
            raise ValueError("This experiment requires uniform weights")
    sq = (y * y).sum(1)
    d2 = (sq[:, None] + sq[None, :] - 2 * y @ y.T).clamp_min(0)
    return uniform_center(1 / (1 + d2))


def build_input_kernel(x: np.ndarray) -> torch.Tensor:
    base = gaussian_affinity_base(x, PERPLEXITY)
    g = (base / (base.max() + 1e-12)) ** SOFTENING
    return normalize_kernel(uniform_center(torch.as_tensor(g, dtype=torch.float32)))


def kernel_metrics(y: torch.Tensor, kx: torch.Tensor, eta: float) -> dict[str, float]:
    with torch.no_grad():
        ky = uniform_student_kernel(y)
        trace = float(ky.trace())
        rho = trace / (1 - 1 / len(y))
        rv_full = float(rv_coefficient(kx, ky))
        rv_hollow = float(rv_coefficient(kx, ky, hollow=True))
        radius = float((y - y.mean(0)).square().sum(1).mean().sqrt())
        penalty = eta * (rho - CONFIG["rho_target"]) ** 2 / 2
    return {
        "rv_full": rv_full,
        "rv_hollow": rv_hollow,
        "rho": rho,
        "rho_error": abs(rho - CONFIG["rho_target"]),
        "rms_radius": radius,
        "dimensionless_penalty": penalty,
        "objective": rv_hollow - penalty,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_embeddings(
    coordinates: dict[float, np.ndarray],
    labels: np.ndarray,
    summary: dict[float, dict[str, Any]],
    common_scale: bool,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.2))
    fig.subplots_adjust(
        left=0.035, right=0.99, top=0.88, bottom=0.105, hspace=0.32, wspace=0.16
    )
    values = np.concatenate(
        [coords - coords.mean(0) for coords in coordinates.values()], axis=0
    )
    shared_bound = float(np.quantile(np.abs(values), 0.998) * 1.04)
    for ax, eta in zip(axes.flat, CONFIG["eta_values"]):
        y = coordinates[eta] - coordinates[eta].mean(0)
        bound = (
            shared_bound
            if common_scale
            else float(np.quantile(np.abs(y), 0.998) * 1.04)
        )
        row = summary[eta]
        ax.scatter(
            y[:, 0], y[:, 1], c=labels, cmap="tab10", s=6,
            alpha=0.82, linewidths=0, rasterized=True,
        )
        ax.set_title(
            rf"$\eta={eta:g}$" + "\n" +
            rf"radius={row['rms_radius']:.1f}, $\rho$={row['rho']:.5f}, "
            rf"RV$_{{hol}}$={row['rv_hollow']:.3f}",
            fontsize=9.2,
        )
        ax.set(xlim=(-bound, bound), ylim=(-bound, bound))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    axes.flat[-1].axis("off")
    digit_handles = [
        Line2D(
            [], [], marker="o", linestyle="", markersize=4.5,
            color=plt.cm.tab10(digit), label=str(digit),
        )
        for digit in range(10)
    ]
    fig.legend(
        handles=digit_handles, title="digit", loc="lower center", ncol=10,
        bbox_to_anchor=(0.5, 0.005), frameon=False, columnspacing=0.9,
    )
    scale_name = "common coordinate scale" if common_scale else "individual scales"
    fig.suptitle(
        f"Hollow RV + neutral regularization: {scale_name}", fontsize=13
    )
    stem = "embeddings_common_scale" if common_scale else "embeddings_autoscaled"
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_final_metrics(summary: dict[float, dict[str, Any]]) -> None:
    eta_values = CONFIG["eta_values"]
    positions = np.arange(len(eta_values))
    labels = [f"{eta:g}" for eta in eta_values]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.2), layout="constrained")
    series = [
        ("rho", r"normalized inertia $\rho$"),
        ("rho_error", r"$|\rho-\rho_0|$"),
        ("rms_radius", "RMS radius"),
        ("rv_hollow", "hollow RV"),
    ]
    for ax, (key, title) in zip(axes, series):
        ax.plot(positions, [float(summary[e][key]) for e in eta_values], "o-")
        if key == "rho":
            ax.axhline(CONFIG["rho_target"], color="black", ls=":", lw=1)
        ax.set(title=title, xlabel=r"regularization strength $\eta$")
        ax.set_xticks(positions, labels, rotation=35, ha="right")
        ax.grid(alpha=0.18)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"final_metrics.{suffix}", dpi=200)
    plt.close(fig)


def plot_trajectories(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4), layout="constrained")
    colors = plt.cm.viridis(np.linspace(0, 1, len(CONFIG["eta_values"])))
    for eta, color in zip(CONFIG["eta_values"], colors):
        selected = [row for row in rows if float(row["eta"]) == eta]
        steps = [int(row["step"]) for row in selected]
        for ax, key in zip(axes, ("rho", "rms_radius", "rv_hollow")):
            ax.plot(
                steps, [float(row[key]) for row in selected], color=color,
                lw=1.5, label=rf"$\eta={eta:g}$",
            )
    axes[0].axhline(CONFIG["rho_target"], color="black", ls=":", lw=1)
    for ax, title in zip(
        axes, (r"normalized inertia $\rho$", "RMS radius", "hollow RV")
    ):
        ax.set(xlabel="Adam steps", title=title)
        ax.grid(alpha=0.18)
    axes[0].legend(fontsize=7.2, ncol=2)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"trajectories.{suffix}", dpi=200)
    plt.close(fig)


def main() -> None:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(CONFIG["torch_threads"])
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset = load_mnist(n_per_digit=CONFIG["n_per_digit"], random_state=SEED)
    labels = np.load(OLD / "coordinates/labels.npy")
    common_start = np.load(OLD / "coordinates/full_rv.npy")
    if dataset.n != CONFIG["sample_size"] or not np.array_equal(dataset.labels, labels):
        raise RuntimeError("The reconstructed MNIST sample differs from experiment 04")

    kx = build_input_kernel(dataset.X)
    weights = default_weights(dataset.n)
    trace_bound = float(1 - weights.square().sum())
    target = CONFIG["rho_target"] * trace_bound
    coordinates: dict[float, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []

    for eta in CONFIG["eta_values"]:
        strength = eta * (dataset.n - 1) / trace_bound**2

        def callback(step: int, y: torch.Tensor, eta_value: float = eta) -> None:
            if step % CONFIG["record_every"] == 0:
                trajectories.append(
                    {"eta": eta_value, "step": step, **kernel_metrics(y, kx, eta_value)}
                )

        fitted, _ = rv_dimred(
            kx,
            output_kernel=uniform_student_kernel,
            weights=weights,
            init=common_start,
            q=2,
            n_iter=CONFIG["iterations_per_eta"],
            lr=CONFIG["lr"],
            hollow=True,
            trace_target=target,
            trace_strength=strength,
            callback=callback,
        )
        y = fitted.numpy()
        coordinates[eta] = y
        row = {"eta": eta, **kernel_metrics(fitted, kx, eta)}
        prediction = KMeans(
            n_clusters=10, n_init=20, random_state=SEED
        ).fit_predict(y)
        row.update(
            {
                "ari": float(adjusted_rand_score(labels, prediction)),
                "trustworthiness_15": float(
                    trustworthiness(dataset.X, y, n_neighbors=15)
                ),
            }
        )
        summary_rows.append(row)
        print(
            f"eta={eta:g}: rho={row['rho']:.7f}, radius={row['rms_radius']:.3f}, "
            f"RV_hol={row['rv_hollow']:.6f}, objective={row['objective']:.6f}",
            flush=True,
        )

    # Every eta branch must start from exactly the same kernel metrics.
    starts = [row for row in trajectories if int(row["step"]) == 0]
    for key in ("rv_full", "rv_hollow", "rho", "rms_radius"):
        if len({float(row[key]) for row in starts}) != 1:
            raise RuntimeError(f"Branches do not share the same step-zero {key}")

    summary = {float(row["eta"]): row for row in summary_rows}
    write_csv(OUT / "summary.csv", summary_rows)
    write_csv(OUT / "trajectories.csv", trajectories)
    np.savez_compressed(
        OUT / "coordinates.npz",
        X=dataset.X,
        labels=labels,
        K_X=kx.numpy(),
        common_start=common_start,
        **{f"eta_{eta:g}": y for eta, y in coordinates.items()},
    )
    plot_embeddings(coordinates, labels, summary, common_scale=True)
    plot_embeddings(coordinates, labels, summary, common_scale=False)
    plot_final_metrics(summary)
    plot_trajectories(trajectories)

    script = Path(__file__).resolve()
    config = dict(
        CONFIG,
        trace_bound=trace_bound,
        trace_target=target,
        common_start_sha256=hashlib.sha256(common_start.tobytes()).hexdigest(),
        wall_time_seconds=time.time() - start,
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        source_hashes={
            str(script.relative_to(ROOT)): hashlib.sha256(
                script.read_bytes()
            ).hexdigest(),
            "src/rv_kernels.py": hashlib.sha256(
                (ROOT / "src/rv_kernels.py").read_bytes()
            ).hexdigest(),
        },
    )
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {OUT} in {config['wall_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
