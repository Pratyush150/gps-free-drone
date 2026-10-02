"""Regenerate docs/*.png. The error figure needs results/benchmark.json."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from gpsfree import Sortie, fly, make_terrain  # noqa: E402

DOCS = ROOT / "docs"
INK, MUTED = "#1f2328", "#8c959f"
C = {"ins": "#8c959f", "flow": "#bf8700", "homography": "#0969da", "essential": "#cf222e", "epipolar": "#1a7f37"}
LABEL = {"ins": "IMU only", "flow": "optical flow", "homography": "homography",
         "essential": "essential (vokit, unaided)", "epipolar": "epipolar + gyro"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def paths(seed=0):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, relief in zip(axs, (False, True)):
        if relief:
            for x0, y0, x1, y1, h in make_terrain(seed, True).buildings:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#d0d7de", ec="none", alpha=0.35 + h / 40))
        ax.plot([0, 150, 150, 0, 0], [0, 0, 150, 150, 0], c=MUTED, lw=0.8, ls=":")
        for m in ("ins", "flow", "homography", "epipolar"):
            r = fly(Sortie(method=m, relief=relief, landing=False), seed, record=True)
            P = np.array([p for _, p, _ in r["log"]])
            ax.plot(P[:, 0], P[:, 1], c=C[m], lw=1.3, label=f"{LABEL[m]}: {r['handover_error_m']:.1f} m off")
            ax.plot(*P[-1], "o", c=C[m], ms=4)
        ax.plot(0, 0, "*", c=INK, ms=12)
        ax.set_xlim(-60, 220)
        ax.set_ylim(-60, 220)
        ax.set_aspect("equal")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        ax.set_title(("Buildings 5-20 m" if relief else "Flat ground") +
                     f": true paths flown on each estimate (seed {seed}). Star = pad.", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "paths.png", dpi=150)


def errors(rows):
    methods = ["ins", "flow", "homography", "essential", "epipolar"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, relief in zip(axs, (False, True)):
        for i, m in enumerate(methods):
            rs = [r for r in rows if r["method"] == m and r["relief"] == relief]
            e = np.array([r["handover_error_m"] for r in rs])
            ok = np.array([r["landed"] for r in rs])
            jit = (np.random.default_rng(i).random(len(e)) - 0.5) * 0.35
            ax.scatter(i + jit[ok], e[ok], s=14, c=C[m], label=None)
            ax.scatter(i + jit[~ok], e[~ok], s=14, facecolors="none", edgecolors=C[m])
            ax.text(i, 0.06, f"{ok.sum()}/{len(rs)}", ha="center", fontsize=8, color=INK)
        ax.set_yscale("log")
        ax.set_ylim(0.04, 5000)
        ax.set_xticks(range(len(methods)), [LABEL[m] for m in methods], rotation=15, fontsize=8)
        ax.set_title(("Buildings 5-20 m" if relief else "Flat ground") +
                     ": error at hand-over (filled = preland then landed)", loc="left", fontsize=9)
    axs[0].set_ylabel("true distance from pad at hand-over (m)")
    fig.tight_layout()
    fig.savefig(DOCS / "handover.png", dpi=150)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    errors(json.load(open(ROOT / "results" / "benchmark.json")))
    paths()
