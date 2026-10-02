"""Every navigation method on the same seeded sorties, flat ground and buildings.

    python examples/benchmark.py --seeds 20 --jobs 8

Each sortie: take off over the pad at 40 m, fly a 150 m square loop (600 m) on
IMU + visual odometry only, return to where the estimate says the pad is, and
hand the true position error to preland's concurrent lander.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from gpsfree import Sortie, fly  # noqa: E402

METHODS = ["ins", "flow", "homography", "essential", "epipolar"]


def job(a):
    method, relief, seed = a
    return fly(Sortie(method=method, relief=relief), seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    tasks = [(m, r, s) for r in (False, True) for m in METHODS for s in range(a.seeds)]
    with Pool(a.jobs) as p:
        rows = p.map(job, tasks, chunksize=1)
    Path(a.out).mkdir(exist_ok=True)
    json.dump(rows, open(Path(a.out) / "benchmark.json", "w"))
    for relief in (False, True):
        print(f"\n{'buildings 5-20 m' if relief else 'flat ground'}  ({a.seeds} sorties each, 600 m loop at 40 m)")
        print(f"  {'method':<12}{'returned':>9}{'handover err m (median / p90)':>32}{'drift %':>9}"
              f"{'landed':>8}{'VO vel err m/s':>16}")
        for m in METHODS:
            rs = [r for r in rows if r["method"] == m and r["relief"] == relief]
            he = np.array([r["handover_error_m"] for r in rs])
            ve = [r["vo_vel_err_med"] for r in rs if r["vo_vel_err_med"] is not None]
            print(f"  {m:<12}{sum(r['returned'] for r in rs):>6}/{len(rs)}"
                  f"{np.median(he):>18.1f} / {np.percentile(he, 90):<10.1f}"
                  f"{np.median([r['drift_pct'] for r in rs]):>9.2f}"
                  f"{sum(r['landed'] for r in rs):>5}/{len(rs)}"
                  f"{(np.median(ve) if ve else float('nan')):>14.2f}")


if __name__ == "__main__":
    main()
