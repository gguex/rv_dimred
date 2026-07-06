"""
supervised_dial_figure.py  —  art. §7.5  Figure 2: supervision generalizes
==========================================================================

From results/05_supervised_dial/indices/supervised_dial.csv, a 2x2 panel:
rows = {ARI, trustworthiness}, columns = {mnist, singlecell}; each panel plots the
train and test curves against the double dial beta, with beta = 0.5 marked as a
safe intermediate point. The story: test ARI (held-out points, labels never used)
rises with beta -- supervision generalizes -- while test trustworthiness holds near
its t-SNE level through the intermediate regime; beta = 1 overfits (train ARI -> 1),
so the test curve peaks at an intermediate / near-full beta, not necessarily beta=1.

Writes results/05_supervised_dial/supervised_dial_figure.{png,pdf}.
"""

# ruff: noqa: E402, I001  (imports follow the sys.path bootstrap)
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (for `src`)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.benchmark_common import exp_coords_dir, exp_indices_dir  # noqa: E402

EXP = "05_supervised_dial"
DATASETS = ("mnist", "singlecell")
METRICS = [("ari", "ARI"), ("trustworthiness", "trustworthiness")]


def load() -> dict[tuple[str, str, str], list[tuple[float, float]]]:
    """(dataset, metric, split) -> sorted [(beta, value)]."""
    rows = list(csv.DictReader((exp_indices_dir(EXP) / "supervised_dial.csv").open()))
    out: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for r in rows:
        for metric, _ in METRICS:
            key = (r["dataset"], metric, r["split"])
            out.setdefault(key, []).append((float(r["beta"]), float(r[metric])))
    for k in out:
        out[k].sort()
    return out


def main() -> None:
    data = load()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

    for col, ds_name in enumerate(DATASETS):
        for row, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            for split, style in (("train", "-o"), ("test", "--s")):
                pts = data[(ds_name, metric, split)]
                betas = [b for b, _ in pts]
                vals = [v for _, v in pts]
                ax.plot(betas, vals, style, label=split, markersize=5)
            ax.axvline(0.5, color="gray", ls=":", lw=1)
            ax.set_ylim(-0.02, 1.02)
            if row == 0:
                ax.set_title(ds_name, fontsize=12)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel(r"dial $\beta$  (0 = t-SNE, 1 = class kernel)")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper left", fontsize=9)

    out = exp_coords_dir(EXP).parent / "supervised_dial_figure"
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}.png and {out}.pdf")


if __name__ == "__main__":
    main()
