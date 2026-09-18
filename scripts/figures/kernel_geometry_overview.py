"""Draw the conceptual geometry figure used in the revised manuscript."""

# ruff: noqa: E402, I001
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="rv-geometry-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/07_geometry"

BLUE = "#2878B5"
LIGHT_BLUE = "#A9D6E5"
RED = "#C93C3C"
GREEN = "#2A9D55"
GRAY = "#646A73"


def arrow2d(
    ax: Any,
    start: tuple[float, float] | np.ndarray,
    end: tuple[float, float] | np.ndarray,
    *,
    color: str,
    lw: float = 1.8,
    style: str = "-",
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": lw,
            "linestyle": style,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 10,
        },
    )


def linear_cone_panel(ax: Any) -> None:
    ax.set_title("(a) Linear output: conic feasible set", fontsize=10.5, pad=6)
    origin = np.array([0.0, 0.0])
    angles = np.deg2rad([8, 20, 34, 48])
    lengths = [2.75, 2.55, 2.65, 2.45]
    for angle, length in zip(angles, lengths, strict=True):
        end = length * np.array([np.cos(angle), np.sin(angle)])
        ax.plot(
            [0, end[0]],
            [0, end[1]],
            color=LIGHT_BLUE,
            lw=8,
            alpha=0.48,
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot([0, end[0]], [0, end[1]], color=BLUE, lw=1.0, alpha=0.75)

    projection_angle = angles[-1]
    direction = np.array([np.cos(projection_angle), np.sin(projection_angle)])
    target = np.array([0.95, 2.55])
    projection = np.dot(target, direction) * direction

    arrow2d(ax, origin, target, color=RED, lw=2.0)
    arrow2d(ax, origin, projection, color=BLUE, lw=2.2)
    ax.plot(
        [target[0], projection[0]],
        [target[1], projection[1]],
        color=GRAY,
        lw=1.3,
        ls=(0, (2, 2)),
    )
    ax.scatter(*target, s=25, color=RED, zorder=5)
    ax.scatter(*projection, s=25, color=BLUE, zorder=5)
    ax.scatter(*origin, s=13, color="#222222", zorder=5)

    target_angle = np.arctan2(target[1], target[0])
    theta = np.linspace(projection_angle, target_angle, 35)
    ax.plot(0.68 * np.cos(theta), 0.68 * np.sin(theta), color=GRAY, lw=1.0)
    theta_mid = (projection_angle + target_angle) / 2
    ax.text(
        0.88 * np.cos(theta_mid),
        0.88 * np.sin(theta_mid),
        r"$\theta$",
        color=GRAY,
        fontsize=9,
        ha="center",
        va="center",
    )
    ax.text(
        target[0] - 0.20,
        target[1] + 0.12,
        r"target $\mathbf{K}_X$",
        color=RED,
        fontsize=9,
    )
    ax.text(
        projection[0] + 0.08,
        projection[1] - 0.20,
        r"nearest $\mathbf{K}_Y^\star$",
        color=BLUE,
        fontsize=9,
    )
    ax.text(
        0.02,
        -0.35,
        r"schematic slice of $\mathcal{S}_q$",
        color=GRAY,
        fontsize=8.3,
    )
    ax.set_xlim(-0.2, 3.0)
    ax.set_ylim(-0.48, 2.95)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def nonlinear_sheet_panel(ax: Any) -> None:
    ax.set_title("(b) Non-linear output: local sheet", fontsize=10.5, pad=6)
    x = np.linspace(-1.55, 1.55, 180)
    y = 0.42 + 0.18 * x**2 + 0.10 * x**3
    ax.fill_between(x, y - 0.13, y + 0.13, color=LIGHT_BLUE, alpha=0.58)
    ax.plot(x, y, color=BLUE, lw=1.2, alpha=0.78)

    x0 = -0.42
    point = np.array([x0, 0.42 + 0.18 * x0**2 + 0.10 * x0**3])
    target = np.array([0.98, 2.25])
    origin = np.array([-1.48, -0.02])
    tangent = np.array([1.0, 0.36 * x0 + 0.30 * x0**2])
    tangent *= 1.05 / np.linalg.norm(tangent)

    arrow2d(ax, origin, target, color=RED, lw=2.0)
    arrow2d(ax, origin, point, color=BLUE, lw=2.2)
    ax.scatter(*point, color=BLUE, s=30, zorder=5)
    ax.scatter(*target, color=RED, s=30, zorder=5)
    ax.scatter(*origin, color="#222222", s=13, zorder=5)
    ax.plot(
        [point[0], target[0]],
        [point[1], target[1]],
        color=GRAY,
        ls=(0, (2, 2)),
        lw=1.3,
    )
    arrow2d(ax, point, point + tangent, color=GREEN, lw=2.2)
    ax.text(
        *(target + [0.07, 0.02]),
        r"target $\mathbf{K}_X$",
        color=RED,
        fontsize=9,
    )
    ax.text(
        *(point + [0.05, -0.28]),
        r"$\mathbf{K}_Y$",
        color=BLUE,
        fontsize=8.8,
    )
    ax.text(
        *(point + tangent + [-0.04, 0.10]),
        r"induced velocity $JJ^*g$",
        color=GREEN,
        fontsize=8.5,
    )
    gradient_label = point + 0.56 * (target - point) + np.array([0.18, 0.02])
    ax.text(*gradient_label, r"gradient $g$", color=GRAY, fontsize=8.3)
    ax.set(xlim=(-1.62, 1.65), ylim=(-0.18, 2.72))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.2, 3.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.16)
    linear_cone_panel(fig.add_subplot(grid[0, 0]))
    nonlinear_sheet_panel(fig.add_subplot(grid[0, 1]))
    fig.subplots_adjust(left=0.02, right=0.985, top=0.91, bottom=0.04)
    for suffix in ("pdf", "png"):
        fig.savefig(
            OUT / f"kernel_geometry_overview.{suffix}",
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
