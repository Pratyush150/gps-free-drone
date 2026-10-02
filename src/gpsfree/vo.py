"""Three ways to turn two downward images into a camera displacement.

All three get the same matched pixels, the same gyro-derived relative rotation
(where they use one), and the same rangefinder reading. Each returns the
camera displacement C1 - C0 in the world frame, or None if it cannot.

flow        mean de-rotated pixel flow times range over focal length. What a
            PX4Flow-class sensor does. Assumes every feature is at the
            rangefinder distance.
homography  RANSAC homography, decomposed with the gyro rotation and the
            ground plane at the rangefinder distance. Exact for a flat floor.
essential   RANSAC essential matrix and cheirality decomposition from vokit
            (vendored), scale from triangulated depth versus the rangefinder.
            Needs a scene that is not a plane.
epipolar    the same epipolar constraint with the rotation taken from the gyro,
            so only the translation direction is estimated (linear, 2 points),
            then the same cheirality check and rangefinder scale.
"""
from __future__ import annotations

import numpy as np

from .vendor.vokit_geometry import (cheirality_counts, essential_8point, recover_pose, sampson_error,
                                    skew)

E_Z = np.array([0.0, 0.0, 1.0])


def _norm(p, K):
    return np.column_stack([(p[:, 0] - K[0, 2]) / K[0, 0], (p[:, 1] - K[1, 2]) / K[1, 1]])


def _ground(R_wc0, rng_m):
    """Plane n.X0 = d in camera-0 coordinates for the ground under a camera
    whose rangefinder reads rng_m along the body axis (the optical axis here)."""
    n = -R_wc0.T @ E_Z
    d = rng_m * n[2]                     # optical axis is camera z: vertical height = range * cos(tilt)
    return n, d


def vo_flow(p0, p1, K, R_wc0, R_wc1, rng_m, window=0.25):
    """PX4Flow-style: only the central window, where depth ~ the rangefinder reading."""
    c = np.hypot((p0[:, 0] - K[0, 2]) / K[0, 2], (p0[:, 1] - K[1, 2]) / K[1, 2])
    if (c < window).sum() >= 6:
        p0, p1 = p0[c < window], p1[c < window]
    if len(p0) < 6:
        return None
    R10 = R_wc1.T @ R_wc0
    x0 = np.column_stack([_norm(p0, K), np.ones(len(p0))])
    pred = x0 @ R10.T                                  # where each ray lands under rotation alone
    pred = pred[:, :2] / pred[:, 2:]
    resid = _norm(p1, K) - pred
    flow = np.median(resid, axis=0)                    # robust to the wrong matches
    n, d = _ground(R_wc0, rng_m)
    Z = d / max(n[2], 0.5)                             # depth at the image centre
    # Centre flow only sees the camera-frame x, y part of the motion; with a tilted
    # camera, horizontal motion also has an optical-axis part. Altitude is held, so
    # solve for the horizontal displacement D whose camera-frame x, y match the flow:
    # a camera moving +x sees the ground flow -x, i.e. (R_c1w D)_xy = -flow * Z.
    A = R_wc1.T[:2, :2]
    D = np.linalg.solve(A, -flow * Z)
    return np.array([D[0], D[1], 0.0])


def _homography(x0, x1):
    A = []
    for (u, v), (s, w) in zip(x0, x1):
        A.append([0, 0, 0, -u, -v, -1, w * u, w * v, w])
        A.append([u, v, 1, 0, 0, 0, -s * u, -s * v, -s])
    _, _, vt = np.linalg.svd(np.array(A))
    return vt[-1].reshape(3, 3)


def _h_err(H, x0, x1):
    q = np.column_stack([x0, np.ones(len(x0))]) @ H.T
    return np.linalg.norm(q[:, :2] / q[:, 2:] - x1, axis=1)


def vo_homography(p0, p1, K, R_wc0, R_wc1, rng_m, rng, iters=60, thresh_px=1.5):
    if len(p0) < 8:
        return None
    x0, x1 = _norm(p0, K), _norm(p1, K)
    th = thresh_px / K[0, 0]
    best = None
    for _ in range(iters):
        s = rng.choice(len(x0), 4, replace=False)
        H = _homography(x0[s], x1[s])
        inl = _h_err(H, x0, x1) < th
        if best is None or inl.sum() > best.sum():
            best = inl
    if best.sum() < 8:
        return None
    H = _homography(x0[best], x1[best])
    H = H / np.linalg.svd(H, compute_uv=False)[1]      # H = +-(R + t n^T / d) has sigma_2 = 1
    R10 = R_wc1.T @ R_wc0
    if np.trace(H @ R10.T) < 0:
        H = -H
    n, d = _ground(R_wc0, rng_m)
    t10 = d * (H - R10) @ n
    return -R_wc1 @ t10


def vo_essential(p0, p1, K, R_wc0, R_wc1, rng_m, rng, iters=60, thresh_px=1.0):
    """vokit's 8-point + recover_pose inside RANSAC; scale from the rangefinder."""
    if len(p0) < 12:
        return None
    best = None
    for _ in range(iters):
        s = rng.choice(len(p0), 8, replace=False)
        try:
            E = essential_8point(p0[s], p1[s], K)
        except np.linalg.LinAlgError:
            continue
        inl = sampson_error(E, p0, p1, K) * K[0, 0] < thresh_px
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 12:
        return None
    E = essential_8point(p0[best], p1[best], K)
    R10, t_unit, n_front, X = recover_pose(E, p0[best], p1[best], K)
    if n_front < 8:
        return None
    # Scale: the rangefinder measures depth along the optical axis to whatever it
    # hits -- ground or roof. The triangulated features nearest the image centre
    # sit on that same surface.
    t10 = _scale_from_centre(p0[best], X, K, rng_m, t_unit)
    return None if t10 is None else -R_wc1 @ t10


def _scale_from_centre(p0, X, K, rng_m, t_unit):
    """Rangefinder depth along the optical axis vs triangulated depth near the image centre."""
    c = np.hypot(p0[:, 0] - K[0, 2], p0[:, 1] - K[1, 2])
    z_unit = np.median(np.asarray(X).reshape(-1, 3)[np.argsort(c)[:6], 2])
    if not np.isfinite(z_unit) or z_unit <= 1e-9:
        return None
    return (rng_m / z_unit) * t_unit


def _t_given_R(q0, q1, R):
    """Unit t with q1^T [t]x R q0 = 0 for all rows: t is orthogonal to (R q0) x q1."""
    A = np.cross(q0 @ R.T, q1)
    _, _, vt = np.linalg.svd(A)
    return vt[-1]


def vo_epipolar(p0, p1, K, R_wc0, R_wc1, rng_m, rng, iters=60, thresh_px=1.0):
    """Gyro-aided: rotation from the IMU, translation direction from 2-point RANSAC."""
    if len(p0) < 12:
        return None
    R10 = R_wc1.T @ R_wc0
    Ki = np.linalg.inv(K)
    q0 = np.column_stack([p0, np.ones(len(p0))]) @ Ki.T
    q1 = np.column_stack([p1, np.ones(len(p1))]) @ Ki.T
    best = None
    for _ in range(iters):
        s = rng.choice(len(p0), 2, replace=False)
        t = _t_given_R(q0[s], q1[s], R10)
        inl = sampson_error(skew(t) @ R10, p0, p1, K) * K[0, 0] < thresh_px
        if best is None or inl.sum() > best.sum():
            best = inl
    if best.sum() < 12:
        return None
    t = _t_given_R(q0[best], q1[best], R10)
    (n_pos, X_pos), (n_neg, X_neg) = (cheirality_counts(R10, t, p0[best], p1[best], K),
                                      cheirality_counts(R10, -t, p0[best], p1[best], K))
    t, X = (t, X_pos) if n_pos >= n_neg else (-t, X_neg)
    t10 = _scale_from_centre(p0[best], X, K, rng_m, t)
    return None if t10 is None else -R_wc1 @ t10


METHODS = {"flow": vo_flow, "homography": vo_homography, "essential": vo_essential,
           "epipolar": vo_epipolar}


def displacement(method, p0, p1, K, R_wc0, R_wc1, rng_m, rng):
    f = METHODS[method]
    if method == "flow":
        return f(p0, p1, K, R_wc0, R_wc1, rng_m)
    return f(p0, p1, K, R_wc0, R_wc1, rng_m, rng)
