"""Regularization path for the kernel t-SNE construction on sampled MNIST.

The unregularized starting point is selected by full RV among three perturbed
PCA initializations. Increasing eta values are then followed by continuation.

Run: .venv/bin/python scripts/experiments/06_regularization/mnist_regularization_path.py
"""

# ruff: noqa: E402, I001  (imports follow the repository-path bootstrap)
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

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="rv-mnist-reg-mpl-"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.spatial import procrustes
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.benchmark_common import normalize_kernel, pca_init, to_tensor
from src.datasets import load_mnist
from src.rv_kernels import (
    default_weights,
    gaussian_affinity_base,
    kernel_trace_penalty,
    rv_coefficient,
    rv_dimred,
    soften_and_center,
)

OUT = ROOT / "results/06_regularization/mnist_path"
CONFIG: dict[str, Any] = {
    "dataset": "MNIST test set, balanced prefix sample",
    "n_per_digit": 50,
    "sample_size": 500,
    "sample_seed": 0,
    "perplexity": 30,
    "softening": 0.5,
    "input_kernel": "normalized centered adaptive Gaussian affinity",
    "output_kernel": "centered Student-t, nu=1",
    "rv_variant": "full",
    "weights": "uniform",
    "q": 2,
    "rho_target": 0.99,
    "eta_path": [0.0, 1.0, 10.0, 100.0, 1000.0],
    "baseline_seeds": [0, 1, 2],
    "baseline_initialization": "PCA plus Gaussian noise of 5% PCA coordinate SD",
    "baseline_selection": "largest final unpenalized full RV; labels unused",
    "baseline_refinement": "additional unpenalized steps after seed selection",
    "continuation": "each positive eta starts from the preceding path solution",
    "optimizer": "Adam, default betas and eps, no weight decay",
    "lr": 0.1,
    "baseline_iterations": 2000,
    "baseline_refinement_iterations": 3000,
    "iterations_per_positive_eta": 1000,
    "record_every": 10,
    "dtype": "float32",
    "device": "cpu",
    "torch_threads": 4,
    "parameter_conversion": "T=1-sum(f_i^2); t0=rho_target*T; lambda=eta*(n-1)/T^2",
    "interpretation": "kernel-inertia path, not classical t-SNE or a quality benchmark",
}


def uniform_student_kernel(
    coords: np.ndarray | torch.Tensor,
    param: Any = None,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Student kernel with O(n^2) uniform double centering.

    This is algebraically the same as Q G Q^T for uniform f, while avoiding the
    dense matrix multiplications in the generic centering implementation.
    """
    del param
    y = coords if isinstance(coords, torch.Tensor) else to_tensor(coords, str(device))
    y = y.to(device)
    n = y.shape[0]
    if weights is not None:
        expected = torch.full_like(weights, 1.0 / n)
        if not torch.allclose(weights, expected, atol=1e-7, rtol=1e-6):
            raise ValueError("uniform_student_kernel requires uniform weights")
    sq = (y * y).sum(1)
    d2 = (sq[:, None] + sq[None, :] - 2 * y @ y.T).clamp_min(0)
    g = 1 / (1 + d2)
    row_mean = g.mean(1, keepdim=True)
    return (g - row_mean - row_mean.T + g.mean()) / n


def metrics(
    y: torch.Tensor,
    kx: torch.Tensor,
    weights: torch.Tensor,
    labels: np.ndarray,
    x: np.ndarray,
    target: float,
    strength: float,
    baseline: np.ndarray,
) -> dict[str, float]:
    with torch.no_grad():
        ky = uniform_student_kernel(y, weights=weights)
        trace = float(ky.trace())
        rv = float(rv_coefficient(kx, ky))
        penalty = float(strength * kernel_trace_penalty(ky, target))
        yc = y - y.mean(0)
        radius = float(yc.square().sum(1).mean().sqrt())
    y_np = y.detach().cpu().numpy()
    predicted = KMeans(n_clusters=10, n_init=20, random_state=0).fit_predict(y_np)
    _, _, disparity = procrustes(baseline, y_np)
    return {
        "rv": rv,
        "trace": trace,
        "rho": trace / (1 - 1 / len(y)),
        "rho_error": abs(trace / (1 - 1 / len(y)) - CONFIG["rho_target"]),
        "penalty": penalty,
        "objective": rv - penalty,
        "rms_radius": radius,
        "ari": float(adjusted_rand_score(labels, predicted)),
        "trustworthiness_15": float(trustworthiness(x, y_np, n_neighbors=15)),
        "procrustes_to_baseline": float(disparity),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    labels: np.ndarray,
    coordinates: dict[float, np.ndarray],
    summary: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
) -> None:
    # A shared coordinate range makes expansion along the path visible.
    values = np.concatenate(list(coordinates.values()), axis=0)
    bound = float(np.quantile(np.abs(values), 0.997) * 1.05)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), layout="constrained")
    for ax, row in zip(axes.flat[:5], summary):
        eta = float(row["eta"])
        y = coordinates[eta]
        ax.scatter(
            y[:, 0],
            y[:, 1],
            c=labels,
            cmap="tab10",
            s=7,
            alpha=0.82,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(
            rf"$\eta={eta:g}$  |  $\rho={row['rho']:.3f}$  |  RV={row['rv']:.3f}",
            fontsize=10,
        )
        ax.set(xlim=(-bound, bound), ylim=(-bound, bound))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes.flat[5]
    eta = np.array([float(row["eta"]) for row in summary])
    ax.semilogx(
        eta[1:], [float(row["rho"]) for row in summary[1:]], "o-", label=r"$\rho$"
    )
    ax.axhline(CONFIG["rho_target"], color="black", ls=":", lw=1, label="target")
    ax.semilogx(eta[1:], [float(row["rv"]) for row in summary[1:]], "s-", label="RV")
    ax.set(
        xlabel=r"regularization strength $\eta$", ylim=(0, 1.02), title="Kernel metrics"
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    digit_handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5,
               color=plt.cm.tab10(digit), label=str(digit))
        for digit in range(10)
    ]
    fig.legend(handles=digit_handles, title="digit", loc="lower center", ncol=10,
               bbox_to_anchor=(0.5, -0.025), frameon=False, columnspacing=1.0)
    fig.suptitle(
        "Kernel t-SNE on sampled MNIST: increasing neutral-component regularization"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"mnist_regularization_path.{suffix}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), layout="constrained")
    for eta_value, color in zip(
        CONFIG["eta_path"], plt.cm.viridis(np.linspace(0, 1, 5))
    ):
        rows = [row for row in trajectories if float(row["eta"]) == eta_value]
        step = [int(row["step"]) for row in rows]
        for ax, key in zip(axes, ("rho", "rv", "rms_radius")):
            ax.plot(
                step,
                [float(row[key]) for row in rows],
                color=color,
                label=rf"$\eta={eta_value:g}$",
            )
    axes[0].axhline(CONFIG["rho_target"], color="black", ls=":", lw=1)
    for ax, title in zip(
        axes, ("Normalized kernel inertia", "Unpenalized RV", "RMS radius")
    ):
        ax.set(xlabel="steps at current strength", title=title)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"mnist_regularization_diagnostics.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(CONFIG["torch_threads"])
    torch.manual_seed(0)
    np.random.seed(0)
    ds = load_mnist(n_per_digit=CONFIG["n_per_digit"], random_state=0)
    assert ds.labels is not None
    x = ds.X
    labels = ds.labels
    weights = default_weights(ds.n, "cpu")
    base = gaussian_affinity_base(to_tensor(x, "cpu"), CONFIG["perplexity"])
    kx = normalize_kernel(soften_and_center(base, CONFIG["softening"], weights, "cpu"))
    trace_bound = float(1 - weights.square().sum())
    target = CONFIG["rho_target"] * trace_bound
    pca = pca_init(x)
    noise_scale = 0.05 * float(pca.std())

    baseline_rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, torch.Tensor]] = []
    for seed in CONFIG["baseline_seeds"]:
        rng = np.random.default_rng(seed)
        init = pca + rng.normal(0, noise_scale, size=pca.shape).astype(np.float32)
        y, rv = rv_dimred(
            kx,
            output_kernel=uniform_student_kernel,
            weights=weights,
            init=init,
            q=2,
            n_iter=CONFIG["baseline_iterations"],
            lr=CONFIG["lr"],
        )
        candidates.append((rv, seed, y))
        baseline_rows.append({"seed": seed, "rv": rv})
        print(f"baseline seed={seed}: RV={rv:.7f}", flush=True)

    _, best_seed, current = max(candidates, key=lambda item: item[0])
    current, best_rv = rv_dimred(
        kx,
        output_kernel=uniform_student_kernel,
        weights=weights,
        init=current,
        q=2,
        n_iter=CONFIG["baseline_refinement_iterations"],
        lr=CONFIG["lr"],
    )
    for row in baseline_rows:
        row["selected"] = row["seed"] == best_seed
        row["refined_rv"] = best_rv if row["selected"] else ""
    print(f"refined baseline seed={best_seed}: RV={best_rv:.7f}", flush=True)
    baseline = current.detach().cpu().numpy()
    coordinates = {0.0: baseline.copy()}
    summary: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []

    for eta in CONFIG["eta_path"]:
        strength = eta * (ds.n - 1) / trace_bound**2
        if eta > 0:
            rows: list[dict[str, Any]] = []

            def record(step: int, y: torch.Tensor) -> None:
                if step % CONFIG["record_every"] == 0:
                    with torch.no_grad():
                        ky = uniform_student_kernel(y, weights=weights)
                        trace = float(ky.trace())
                        rv = float(rv_coefficient(kx, ky))
                        radius = float((y - y.mean(0)).square().sum(1).mean().sqrt())
                    rows.append(
                        {
                            "eta": eta,
                            "step": step,
                            "rho": trace / trace_bound,
                            "rv": rv,
                            "rms_radius": radius,
                        }
                    )

            current, _ = rv_dimred(
                kx,
                output_kernel=uniform_student_kernel,
                weights=weights,
                init=current,
                q=2,
                n_iter=CONFIG["iterations_per_positive_eta"],
                lr=CONFIG["lr"],
                trace_target=target,
                trace_strength=strength,
                callback=record,
            )
            trajectories.extend(rows)
            coordinates[eta] = current.detach().cpu().numpy()
        else:
            ky = uniform_student_kernel(current, weights=weights)
            trajectories.append(
                {
                    "eta": eta,
                    "step": 0,
                    "rho": float(ky.trace()) / trace_bound,
                    "rv": float(rv_coefficient(kx, ky)),
                    "rms_radius": float(
                        (current - current.mean(0)).square().sum(1).mean().sqrt()
                    ),
                }
            )

        row = metrics(current, kx, weights, labels, x, target, strength, baseline)
        variable = current.clone().requires_grad_()
        ky = uniform_student_kernel(variable, weights=weights)
        objective = rv_coefficient(kx, ky) - strength * kernel_trace_penalty(ky, target)
        row.update(
            {
                "eta": eta,
                "lambda": strength,
                "gradient_norm": float(
                    torch.autograd.grad(objective, variable)[0].norm()
                ),
            }
        )
        summary.append(row)
        print(
            f"path eta={eta:g}: RV={row['rv']:.6f}, rho={row['rho']:.6f}, "
            f"radius={row['rms_radius']:.3f}, ARI={row['ari']:.3f}",
            flush=True,
        )

    arrays = {
        "X": x,
        "labels": labels,
        "K_X": kx.numpy(),
        "baseline_seed": np.array(best_seed),
    }
    arrays.update({f"Y_eta{eta:g}": y for eta, y in coordinates.items()})
    np.savez_compressed(OUT / "coordinates.npz", **arrays)
    write_csv(OUT / "baseline_candidates.csv", baseline_rows)
    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "trajectories.csv", trajectories)
    plot_results(labels, coordinates, summary, trajectories)

    config = dict(
        CONFIG,
        selected_baseline_seed=best_seed,
        selected_baseline_rv=best_rv,
        trace_bound=trace_bound,
        trace_target=target,
        wall_time_seconds=time.time() - start,
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        source_hashes={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), ROOT / "src/rv_kernels.py")
        },
    )
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {OUT} in {config['wall_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
