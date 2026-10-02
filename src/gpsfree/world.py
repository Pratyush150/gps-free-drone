"""Terrain, landmarks, and a downward camera that tracks them.

World frame: x east, y north, z up, metres; the landing pad is at the origin.
Camera optical frame: x right, y down, z forward (pointing at the ground).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# body (x fwd, y left, z up) -> optical, for a camera looking straight down
R_BODY_CAM = np.diag([1.0, -1.0, -1.0])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def body_rotation(pitch_dctk, roll_dctk):
    """World <- body for the two tilt angles the dctk cascades produce.

    dctk's horizontal chain uses a = -g tan(theta) ("nose down to go forward"),
    so a positive dctk angle is a rotation that tilts thrust *backwards*.
    """
    return rot_y(-pitch_dctk) @ rot_x(roll_dctk)


@dataclass
class Terrain:
    """Ground landmarks, optionally on top of box buildings."""
    landmarks: np.ndarray                 # (N, 3)
    buildings: np.ndarray                 # (B, 5): x0, y0, x1, y1, height
    cell: float = 10.0
    _grid: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        keys = np.floor(self.landmarks[:, :2] / self.cell).astype(int)
        for i, k in enumerate(map(tuple, keys)):
            self._grid.setdefault(k, []).append(i)
        self._grid = {k: np.array(v) for k, v in self._grid.items()}

    def height_at(self, x, y):
        h = 0.0
        for x0, y0, x1, y1, hb in self.buildings:
            if x0 <= x <= x1 and y0 <= y <= y1:
                h = max(h, hb)
        return h

    def near(self, x, y, radius):
        r = int(np.ceil(radius / self.cell))
        cx, cy = int(np.floor(x / self.cell)), int(np.floor(y / self.cell))
        idx = [self._grid[(i, j)] for i in range(cx - r, cx + r + 1) for j in range(cy - r, cy + r + 1)
               if (i, j) in self._grid]
        return np.concatenate(idx) if idx else np.zeros(0, int)


def make_terrain(seed, relief=False, extent=(-60.0, 220.0), density=0.08, n_buildings=60):
    """Textured ground over the mission area. relief=True adds 5-20 m box buildings,
    never over the landing pad."""
    rng = np.random.default_rng(seed)
    lo, hi = extent
    n = int(density * (hi - lo) ** 2)
    pts = np.column_stack([rng.uniform(lo, hi, n), rng.uniform(lo, hi, n), np.zeros(n)])
    blds = []
    if relief:
        while len(blds) < n_buildings:
            w, d = rng.uniform(8, 25, 2)
            x0, y0 = rng.uniform(lo, hi - 25, 2)
            if x0 < 15 and y0 < 15 and x0 + w > -15 and y0 + d > -15:
                continue                                    # keep the pad area open
            blds.append([x0, y0, x0 + w, y0 + d, rng.uniform(5, 20)])
    blds = np.array(blds).reshape(-1, 5)
    t = Terrain(pts, blds)
    if relief:
        t.landmarks[:, 2] = [t.height_at(x, y) for x, y in t.landmarks[:, :2]]
        t = Terrain(t.landmarks, blds)
    return t


@dataclass
class DownCamera:
    f: float = 300.0
    width: int = 320
    height: int = 240
    pixel_noise: float = 0.5
    outlier_rate: float = 0.05
    max_features: int = 120

    @property
    def K(self):
        return np.array([[self.f, 0, (self.width - 1) / 2], [0, self.f, (self.height - 1) / 2], [0, 0, 1.0]])

    def R_wc(self, R_wb):
        return R_wb @ R_BODY_CAM

    def project(self, terrain: Terrain, C, R_wc):
        """Visible landmark ids and their pixel coordinates."""
        idx = terrain.near(C[0], C[1], radius=C[2])
        if len(idx) == 0:
            return idx, np.zeros((0, 2))
        Xc = (terrain.landmarks[idx] - C) @ R_wc
        ok = Xc[:, 2] > 1.0
        idx, Xc = idx[ok], Xc[ok]
        K = self.K
        uv = np.column_stack([K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]])
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < self.width) & (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
        return idx[inside], uv[inside]

    def track(self, terrain, C0, R0, C1, R1, rng):
        """Matched pixel pairs between two frames: noise plus a fraction of wrong matches."""
        i0, uv0 = self.project(terrain, C0, R0)
        i1, uv1 = self.project(terrain, C1, R1)
        common, a, b = np.intersect1d(i0, i1, return_indices=True)
        if len(common) > self.max_features:
            keep = rng.choice(len(common), self.max_features, replace=False)
            a, b = a[keep], b[keep]
        p0 = uv0[a] + rng.normal(0, self.pixel_noise, (len(a), 2))
        p1 = uv1[b] + rng.normal(0, self.pixel_noise, (len(b), 2))
        bad = rng.random(len(p1)) < self.outlier_rate
        p1[bad] = rng.uniform([0, 0], [self.width, self.height], (bad.sum(), 2))
        return p0, p1


def rangefinder(terrain: Terrain, C, R_wb, rng, sigma_rel=0.01, step=0.25):
    """Downward lidar along the body -z axis (the optical axis): ray-cast range to
    the first surface it hits, roof or ground. At 0.3 rad of tilt from 40 m that
    surface is 12 m away from the point directly below the vehicle."""
    down = -R_wb[:, 2]
    r_ground = C[2] / max(-down[2], 0.2)
    r = r_ground
    if len(terrain.buildings):
        for s in np.arange(0.0, r_ground, step):
            p = C + s * down
            if p[2] <= terrain.height_at(p[0], p[1]):
                r = s
                break
    return r * (1 + rng.normal(0, sigma_rel))
