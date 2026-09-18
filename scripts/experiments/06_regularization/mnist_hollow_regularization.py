"""Hollow and neutral regularization from one common MNIST embedding.

The saved full-RV embedding from experiment 04 is the exact common state at
step zero. Four 500-step branches change only the objective: hollow RV,
intermediate neutral regularization, strong neutral regularization, and hollow
RV combined with strong neutral regularization.

Run:
    .venv/bin/python \
        scripts/experiments/06_regularization/mnist_hollow_regularization.py
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

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="rv-mnist-comparison-"))
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
OUT = ROOT / "results/06_regularization/mnist_hollow_regularization"
CONFIG: dict[str, Any] = {
    "dataset": "MNIST test set, balanced prefix sample",
    "n_per_digit": 200,
    "sample_size": 2000,
    "sample_seed": SEED,
    "perplexity": PERPLEXITY,
    "softening": SOFTENING,
    "q": 2,
    "weights": "uniform",
    "input_kernel": "normalized centered adaptive Gaussian affinity",
    "output_kernel": "centered Student-t, nu=1",
    "common_start": "saved full-RV solution from experiment 04 after 500 steps",
    "reference": "unchanged saved t-SNE coordinates from experiment 04",
    "rho_target": 0.99,
    "intermediate_eta": 100.0,
    "strong_eta": 1000.0,
    "combined_eta": 100000.0,
    "branches": [
        "hollow RV",
        "full RV plus eta=100",
        "full RV plus eta=1000",
        "hollow RV plus eta=100000",
    ],
    "branch_protocol": "independent; exact same coordinates at step zero",
    "iterations_per_branch": 500,
    "record_every": 10,
    "lr": 0.1,
    "dtype": "float32",
    "device": "cpu",
    "torch_threads": 4,
    "parameter_conversion": "T=1-1/n; t0=rho_target*T; lambda=eta*(n-1)/T^2",
}


def uniform_center(g: torch.Tensor) -> torch.Tensor:
    """Q G Q^T for uniform weights, evaluated in O(n^2) memory and time."""
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def kernel_metrics(y: torch.Tensor, kx: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        ky = uniform_student_kernel(y)
        trace = float(ky.trace())
        centered = y - y.mean(0)
        return {
            "rv_full": float(rv_coefficient(kx, ky)),
            "rv_hollow": float(rv_coefficient(kx, ky, hollow=True)),
            "rho": trace / (1 - 1 / len(y)),
            "rms_radius": float(centered.square().sum(1).mean().sqrt()),
        }


def final_metrics(
    name: str,
    y: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
    kx: torch.Tensor,
    target: float,
    eta: float,
    hollow: bool,
) -> dict[str, Any]:
    values = kernel_metrics(torch.as_tensor(y, dtype=torch.float32), kx)
    prediction = KMeans(n_clusters=10, n_init=20, random_state=SEED).fit_predict(y)
    return {
        "configuration": name,
        "hollow_objective": hollow,
        "eta": eta,
        **values,
        "objective_rv": values["rv_hollow" if hollow else "rv_full"],
        "ari": float(adjusted_rand_score(labels, prediction)),
        "trustworthiness_15": float(trustworthiness(x, y, n_neighbors=15)),
        "rho_target": CONFIG["rho_target"],
        "trace_target": target,
    }


def run_branch(
    name: str,
    initial: np.ndarray,
    kx: torch.Tensor,
    weights: torch.Tensor,
    target: float,
    eta: float,
    hollow: bool,
    record: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    strength = eta * (len(initial) - 1) / (1 - 1 / len(initial)) ** 2
    trajectory: list[dict[str, Any]] = []

    def callback(step: int, y: torch.Tensor) -> None:
        if record and step % CONFIG["record_every"] == 0:
            trajectory.append(
                {"configuration": name, "step": step, **kernel_metrics(y, kx)}
            )

    fitted, _ = rv_dimred(
        kx,
        output_kernel=uniform_student_kernel,
        weights=weights,
        init=initial,
        q=2,
        n_iter=CONFIG["iterations_per_branch"],
        lr=CONFIG["lr"],
        hollow=hollow,
        trace_target=target,
        trace_strength=strength,
        callback=callback,
    )
    return fitted.numpy(), trajectory


def scatter_panel(
    ax: Any,
    y_raw: np.ndarray,
    labels: np.ndarray,
    title: str,
    row: dict[str, Any],
    bound: float,
) -> None:
    y = y_raw - y_raw.mean(0)
    ax.scatter(
        y[:, 0], y[:, 1], c=labels, cmap="tab10", s=6,
        alpha=0.82, linewidths=0, rasterized=True,
    )
    ax.set_title(
        title + "\n" +
        rf"radius={row['rms_radius']:.1f}, $\rho$={row['rho']:.3f}, "
        rf"RV={row['objective_rv']:.3f}",
        fontsize=9.2,
    )
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def evolution_panel(
    ax: Any,
    trajectories: dict[str, list[dict[str, Any]]],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    styles = [
        ("hollow_rv", "hollow RV", "#d62728"),
        ("regularized_eta1000", r"full RV, $\eta=1000$", "#1f77b4"),
        (
            "hollow_regularized_eta100000",
            r"hollow RV, $\eta=10^5$",
            "#9467bd",
        ),
    ]
    for configuration, label, color in styles:
        rows = trajectories[configuration]
        ax.plot(
            [int(row["step"]) for row in rows],
            [float(row[metric]) for row in rows],
            color=color,
            lw=1.8,
            label=label,
        )
    if metric == "rho":
        ax.axhline(
            CONFIG["rho_target"], color="black", ls=":", lw=1, label="target"
        )
    ax.set(xlabel="Adam steps", ylabel=ylabel, title=title)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=7.3, loc="best")


def plot(
    labels: np.ndarray,
    coordinates: dict[str, np.ndarray],
    rows: dict[str, dict[str, Any]],
    trajectories: dict[str, list[dict[str, Any]]],
) -> None:
    fig = plt.figure(figsize=(14.4, 7.3))
    grid = fig.add_gridspec(2, 4)
    fig.subplots_adjust(
        left=0.035, right=0.965, top=0.88, bottom=0.105, hspace=0.38, wspace=0.28
    )

    panels = [
        (0, 0, "tsne", "t-SNE reference"),
        (0, 1, "common_start", "common start: unmodified full RV"),
        (0, 2, "hollow_rv", "hollow RV"),
        (
            0, 3, "regularized_eta100",
            r"intermediate regularization, $\eta=100$",
        ),
        (1, 0, "regularized_eta1000", r"regularized, $\eta=1000$"),
        (
            1, 1, "hollow_regularized_eta100000",
            r"hollow + regularization, $\eta=10^5$",
        ),
    ]
    for row_index, column, key, title in panels:
        centered = coordinates[key] - coordinates[key].mean(0)
        bound = float(np.quantile(np.abs(centered), 0.998) * 1.04)
        scatter_panel(
            fig.add_subplot(grid[row_index, column]), coordinates[key], labels,
            title, rows[key], bound,
        )

    evolution_panel(
        fig.add_subplot(grid[1, 2]), trajectories, "rho",
        "Kernel-space evolution", r"normalized inertia $\rho$",
    )
    evolution_panel(
        fig.add_subplot(grid[1, 3]), trajectories, "rms_radius",
        "Coordinate-space evolution", "RMS radius",
    )

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
    fig.suptitle(
        "One common MNIST embedding: hollow and neutral regularization branches",
        fontsize=13,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(
            OUT / f"mnist_hollow_regularization.{suffix}", dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(CONFIG["torch_threads"])
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset = load_mnist(n_per_digit=CONFIG["n_per_digit"], random_state=SEED)
    labels = np.load(OLD / "coordinates/labels.npy")
    if dataset.n != CONFIG["sample_size"] or not np.array_equal(dataset.labels, labels):
        raise RuntimeError("The reconstructed MNIST sample differs from experiment 04")

    common_start = np.load(OLD / "coordinates/full_rv.npy")
    tsne = np.load(OLD / "coordinates/tsne.npy")
    if common_start.shape != (dataset.n, 2) or tsne.shape != (dataset.n, 2):
        raise RuntimeError("Unexpected historical-coordinate shape")

    kx = build_input_kernel(dataset.X)
    weights = default_weights(dataset.n)
    trace_bound = float(1 - weights.square().sum())
    target = CONFIG["rho_target"] * trace_bound
    coordinates = {"tsne": tsne, "common_start": common_start}
    trajectories: dict[str, list[dict[str, Any]]] = {}

    specifications = [
        ("hollow_rv", 0.0, True, True),
        ("regularized_eta100", CONFIG["intermediate_eta"], False, False),
        ("regularized_eta1000", CONFIG["strong_eta"], False, True),
        (
            "hollow_regularized_eta100000",
            CONFIG["combined_eta"],
            True,
            True,
        ),
    ]
    for name, eta, hollow, record in specifications:
        fitted, trajectory = run_branch(
            name, common_start, kx, weights, target, eta, hollow, record
        )
        coordinates[name] = fitted
        trajectories[name] = trajectory

    baseline_values = kernel_metrics(
        torch.as_tensor(common_start, dtype=torch.float32), kx
    )
    for name in (
        "hollow_rv",
        "regularized_eta1000",
        "hollow_regularized_eta100000",
    ):
        first = trajectories[name][0]
        for key in ("rv_full", "rv_hollow", "rho", "rms_radius"):
            if abs(float(first[key]) - baseline_values[key]) > 1e-12:
                raise RuntimeError(f"{name} does not start from the common state")

    summary = [
        final_metrics("tsne", tsne, dataset.X, labels, kx, target, 0.0, False),
        final_metrics(
            "common_start", common_start, dataset.X, labels, kx, target, 0.0, False
        ),
    ]
    for name, eta, hollow, _ in specifications:
        row = final_metrics(
            name, coordinates[name], dataset.X, labels, kx, target, eta, hollow
        )
        summary.append(row)
        print(
            f"{name}: objective_RV={row['objective_rv']:.6f}, "
            f"rho={row['rho']:.6f}, radius={row['rms_radius']:.3f}, "
            f"trust={row['trustworthiness_15']:.4f}",
            flush=True,
        )

    by_name = {str(row["configuration"]): row for row in summary}
    write_csv(OUT / "summary.csv", summary)
    write_csv(
        OUT / "trajectories.csv",
        trajectories["hollow_rv"]
        + trajectories["regularized_eta1000"]
        + trajectories["hollow_regularized_eta100000"],
    )
    np.savez_compressed(
        OUT / "coordinates.npz",
        X=dataset.X,
        labels=labels,
        K_X=kx.numpy(),
        **coordinates,
    )
    plot(labels, coordinates, by_name, trajectories)

    source_paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/experiments/04_tether/tether_run.py",
        ROOT / "src/rv_kernels.py",
    ]
    artifact_paths = [
        OLD / "coordinates/full_rv.npy",
        OLD / "coordinates/tsne.npy",
        OLD / "coordinates/labels.npy",
    ]
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
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        historical_artifact_hashes={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in artifact_paths
        },
    )
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {OUT} in {config['wall_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
