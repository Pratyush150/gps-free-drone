"""One GPS-free sortie: fly a loop on VO + IMU, come back, hand over to the lander.

Flight control is dctk's MultirotorCascade, one instance per horizontal axis,
driving the same per-axis attitude plant its own tests use. The landing is
preland's run_episode with the "concurrent" strategy, started from the true
position error the navigation stage left behind.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import preland.episode as pe
from dctk.cascade import MultirotorCascade
from preland import Scenario, run_episode

from .nav import NavEKF
from .vo import displacement
from .world import DownCamera, body_rotation, make_terrain, rangefinder

G = 9.81
# Every method compares keyframes 1 s apart (10 frames at 10 Hz). Epipolar
# methods need the parallax -- 0.5 m of baseline from 40 m drowns in 0.5 px of
# noise -- and giving all methods the same baseline keeps the comparison fair.
KEYFRAME = {"ins": 10, "flow": 10, "homography": 10, "essential": 10, "epipolar": 10}
INERTIA, DAMPING, MASS, DRAG = 0.01, 0.002, 1.2, 0.15   # dctk's own test plant


@dataclass
class Sortie:
    method: str = "homography"         # "ins" | "flow" | "homography" | "essential" | "epipolar"
    relief: bool = False
    altitude: float = 40.0
    route: tuple = ((150.0, 0.0), (150.0, 150.0), (0.0, 150.0), (0.0, 0.0))
    waypoint_radius: float = 3.0
    wind_mps: float = 3.0
    acc_bias_sigma: float = 0.05
    acc_noise: float = 0.15
    gyro_noise_rad: float = 2e-4      # relative-rotation error per 0.1 s frame (~0.01 deg)
    dt: float = 0.004
    nav_every: int = 5                # EKF predict at 50 Hz
    cam_every: int = 25               # camera at 10 Hz
    keyframe: int | None = None       # frames between the two images VO compares; None = per method
    t_max: float = 300.0
    camera: DownCamera = field(default_factory=DownCamera)
    landing: bool = True


class Axis:
    """One horizontal axis: dctk cascade + attitude plant + translational dynamics."""

    def __init__(self):
        self.ctl = MultirotorCascade()
        self.p = self.v = self.th = self.q = 0.0

    def step(self, sp, p_est, v_est, wind, dt):
        tau = self.ctl.update_horizontal(sp, p_est, v_est, self.th, self.q, dt)
        self.q += (np.clip(tau, -0.5, 0.5) - DAMPING * self.q) / INERTIA * dt
        self.th += self.q * dt
        rel = self.v - wind
        a = -G * math.tan(np.clip(self.th, -0.6, 0.6)) - DRAG * rel * abs(rel) / MASS
        self.v += a * dt
        self.p += self.v * dt
        return a


def fly(cfg: Sortie, seed: int, record=False):
    rng = np.random.default_rng(seed)
    terrain = make_terrain(seed, cfg.relief)
    cam = cfg.camera
    ax, ay = Axis(), Axis()
    nav = NavEKF(np.zeros(2))
    bias = rng.normal(0, cfg.acc_bias_sigma, 2)
    wdir = rng.uniform(0, 2 * np.pi)
    wind = cfg.wind_mps * np.array([np.cos(wdir), np.sin(wdir)])
    gust = np.zeros(2)
    route = list(cfg.route)
    k, t, step = 0, 0.0, 0
    acc_sum = np.zeros(2)
    frames = []                       # recent (C, R_wc, R_wb), newest last
    gap = cfg.keyframe or KEYFRAME[cfg.method]
    settled, log = 0.0, []
    vo_err = []
    while t < cfg.t_max:
        sp = route[k]
        gust += (-gust / 3.0) * cfg.dt + rng.normal(0, 0.8 * np.sqrt(2 * cfg.dt / 3.0), 2)
        w = wind + gust
        a = np.array([ax.step(sp[0], nav.pos[0], nav.vel[0], w[0], cfg.dt),
                      ay.step(sp[1], nav.pos[1], nav.vel[1], w[1], cfg.dt)])
        acc_sum += a
        step += 1
        t += cfg.dt
        if step % cfg.nav_every == 0:
            n = cfg.nav_every
            meas = acc_sum / n + bias + rng.normal(0, cfg.acc_noise / np.sqrt(n), 2)
            nav.predict(meas, n * cfg.dt)
            acc_sum[:] = 0
        if step % cfg.cam_every == 0:
            C = np.array([ax.p, ay.p, cfg.altitude])
            R_wb = body_rotation(ax.th, ay.th)
            R_wc = cam.R_wc(R_wb)
            frames.append((C, R_wc, R_wb))
            frames = frames[-(gap + 1):]
            if len(frames) == gap + 1 and cfg.method != "ins" and (step // cfg.cam_every) % gap == 0:
                C0, R0, Rb0 = frames[0]
                p0, p1 = cam.track(terrain, C0, R0, C, R_wc, rng)
                # gyro gives the relative rotation, with a little error
                dR = body_rotation(*rng.normal(0, cfg.gyro_noise_rad, 2))
                r = rangefinder(terrain, C0, Rb0, rng)
                d = displacement(cfg.method, p0, p1, cam.K, R0, dR @ R_wc, r, rng)
                if d is not None and np.all(np.isfinite(d)):
                    dtc = gap * cfg.cam_every * cfg.dt
                    v_meas = d[:2] / dtc
                    vo_err.append(np.linalg.norm(v_meas - (C[:2] - C0[:2]) / dtc))
                    nav.update_velocity(v_meas)
            if record:
                log.append((t, np.array([ax.p, ay.p]), nav.pos.copy()))
        if np.linalg.norm(nav.pos - sp) < cfg.waypoint_radius and k < len(route) - 1:
            k += 1
        if k == len(route) - 1 and np.linalg.norm(nav.pos - sp) < 0.5 and np.linalg.norm(nav.vel) < 0.3:
            settled += cfg.dt
            if settled > 2.0:
                break
        else:
            settled = 0.0
    true = np.array([ax.p, ay.p])
    dist = 2 * (150 + 150)
    out = {"method": cfg.method, "relief": cfg.relief, "seed": seed, "time": t,
           "returned": settled > 2.0,
           "handover_error_m": float(np.linalg.norm(true)),
           "drift_pct": float(100 * np.linalg.norm(true) / dist),
           "vo_vel_err_med": float(np.median(vo_err)) if vo_err else None,
           "vo_accepted": nav.accepted, "vo_rejected": nav.rejected}
    if cfg.landing:
        res = land(true, seed)
        out.update({"landed": bool(res.success), "landing_outcome": res.outcome,
                    "landing_error_m": float(res.lateral_error_m)})
    if record:
        out["log"] = log
    return out


@contextmanager
def _handover_at(offset_xy):
    """Start preland's episode at a given true offset instead of its sampled one.

    preland's run_episode draws the handover offset from a Gaussian inside
    initial_state(); here the offset is whatever the navigation stage produced.
    Everything else -- speed, heading and yaw draws -- is kept as preland does it.
    """
    original = pe.initial_state

    def initial_state(scenario, rng):
        s = original(scenario, rng)
        s.position[:2] = offset_xy
        return s

    pe.initial_state = initial_state
    try:
        yield
    finally:
        pe.initial_state = original


def land(offset_xy, seed, strategy="concurrent"):
    with _handover_at(np.asarray(offset_xy, float)):
        return run_episode(strategy, Scenario(), seed)
