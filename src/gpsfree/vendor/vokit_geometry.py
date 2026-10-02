# Vendored unchanged from vokit (https://github.com/Pratyush150/monocular-visual-odometry,
# commit 7ab552a, src/vokit/vo/geometry.py), same author, MIT licence.
# Copied rather than depended on because the vokit package pins opencv-contrib,
# and only this pure-NumPy module is needed here.
"""Two-view geometry, derived: the essential matrix, its four poses, cheirality.

Everything here is implemented from the definitions and then checked against
OpenCV in ``tests/test_geometry.py``.  The point is not to replace OpenCV --
the pipeline in :mod:`vokit.vo.monocular` calls ``cv2.findEssentialMat``
because what you want from a library is the RANSAC-wrapped minimal solver, not
the algebra.  The point is that you can state what the library is doing.

**The epipolar constraint.**  A point ``X`` in the world projects to ``x0`` in
camera 0 and ``x1`` in camera 1.  In *calibrated* (normalised) coordinates,
``x = K^-1 * pixel``, the two rays and the baseline are coplanar.  Writing that
coplanarity as a scalar triple product::

    x1 . ( t x (R x0) ) = 0
    x1^T [t]_x R x0 = 0
    x1^T E x0 = 0,        E = [t]_x R

That is the whole derivation.  ``E`` is 3x3, defined up to scale (8 degrees of
freedom in the entries), and constrained: ``[t]_x`` has rank 2, so ``det E = 0``,
and the two non-zero singular values of ``E`` are **equal**.  Five degrees of
freedom survive -- 3 rotation + 3 translation - 1 scale -- which is why the
minimal solver is a five-point algorithm.

**The scale is not there.**  ``E`` is homogeneous.  Scaling ``t`` scales ``E``
and satisfies the same constraint.  No amount of algebra recovers ``|t|`` from
two images, because two images of a scene twice as large from twice as far away
are pixel-identical.  See :mod:`vokit.vo.scale`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalise_points",
    "skew",
    "essential_8point",
    "decompose_essential",
    "triangulate_dlt",
    "cheirality_counts",
    "recover_pose",
    "sampson_error",
]


def skew(t: np.ndarray) -> np.ndarray:
    """``[t]_x``, the matrix with ``[t]_x a = t x a``."""
    tx, ty, tz = np.asarray(t, dtype=float).ravel()
    return np.array([[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]])


def normalise_points(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: centroid at the origin, mean distance ``sqrt(2)``.

    Without it the entries of the 8-point design matrix span several orders of
    magnitude (pixel coordinates squared reach 1e6 while the homogeneous column
    is 1), the smallest singular vector is dominated by round-off, and the
    result is visibly wrong on real data while being algebraically correct.
    This is the single reason the 8-point algorithm has a reputation for being
    unusable and also the reason it is not.

    Returns the transformed points and the ``3x3`` transform ``T`` such that
    ``x_norm = T x``.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 2)
    mu = x.mean(axis=0)
    d = np.linalg.norm(x - mu, axis=1).mean()
    s = np.sqrt(2.0) / d if d > 1e-12 else 1.0
    T = np.array([[s, 0, -s * mu[0]], [0, s, -s * mu[1]], [0, 0, 1.0]])
    xh = np.column_stack([x, np.ones(len(x))]) @ T.T
    return xh[:, :2], T


def essential_8point(p0: np.ndarray, p1: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Essential matrix from >=8 correspondences, normalised 8-point.

    ``p0``, ``p1`` are pixel coordinates in views 0 and 1.  ``x1^T E x0 = 0``
    is linear in the nine entries of ``E``, so stack one row per match::

        [x1.x0, x1.y0, x1, y1.x0, y1.y0, y1, x0, y0, 1] . vec(E) = 0

    and take the right singular vector of the smallest singular value.  Then
    **project onto the essential manifold**: SVD ``E``, replace the singular
    values with ``(1, 1, 0)``, recompose.  Skipping that projection is the
    classic bug -- the estimate is a valid fundamental matrix but not a valid
    essential matrix, so its four-way decomposition produces rotations that are
    not rotations, and the failure appears downstream as a trajectory that is
    subtly wrong rather than as an exception here.
    """
    K = np.asarray(K, dtype=float)
    Kinv = np.linalg.inv(K)
    q0 = (np.column_stack([np.asarray(p0, float).reshape(-1, 2), np.ones(len(p0))]) @ Kinv.T)[:, :2]
    q1 = (np.column_stack([np.asarray(p1, float).reshape(-1, 2), np.ones(len(p1))]) @ Kinv.T)[:, :2]
    if len(q0) < 8:
        raise ValueError("the 8-point algorithm needs at least 8 correspondences")

    n0, T0 = normalise_points(q0)
    n1, T1 = normalise_points(q1)
    x0, y0 = n0[:, 0], n0[:, 1]
    x1, y1 = n1[:, 0], n1[:, 1]
    A = np.column_stack(
        [x1 * x0, x1 * y0, x1, y1 * x0, y1 * y0, y1, x0, y0, np.ones(len(x0))]
    )
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)

    E = _project_essential(E)
    E = T1.T @ E @ T0  # undo the normalisation
    # ...and project AGAIN.  The un-normalisation is a general 3x3 congruence,
    # which does not preserve "the two non-zero singular values are equal", so
    # a matrix that was a valid essential matrix in normalised coordinates is
    # not one in pixel coordinates.  Projecting only once -- the version in
    # most write-ups -- leaves a matrix whose four-way decomposition produces
    # rotations that are not quite rotations, and the damage appears far
    # downstream as a trajectory that is subtly wrong.
    E = _project_essential(E)
    return E / (np.linalg.norm(E) + 1e-300)


def _project_essential(E: np.ndarray) -> np.ndarray:
    """Nearest matrix (in Frobenius norm) with singular values ``(1, 1, 0)``."""
    U, _, Vt = np.linalg.svd(np.asarray(E, dtype=float))
    return U @ np.diag([1.0, 1.0, 0.0]) @ Vt


def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The four ``(R, t)`` pairs consistent with an essential matrix.

    Write ``E = U diag(1,1,0) V^T`` and ``W = [[0,-1,0],[1,0,0],[0,0,1]]``
    (a 90 degree rotation about z).  Then::

        R in {U W V^T, U W^T V^T}          t in {+u3, -u3}

    Four combinations.  Geometrically they are: the true one; the same with the
    baseline reversed; and both of those with the second camera rotated 180
    degrees about the baseline -- the "twisted pair".  **Only one puts the
    observed points in front of both cameras**, and that is what
    :func:`cheirality_counts` tests.  This four-way ambiguity is the same one
    that appears in stereo rectification, and it is a favourite interview
    question precisely because the resolution is a physical argument rather than
    an algebraic one.
    """
    U, _, Vt = np.linalg.svd(np.asarray(E, dtype=float))
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def triangulate_dlt(P0: np.ndarray, P1: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Linear triangulation.  ``(N, 3)`` from two ``3x4`` matrices and two ``(N,2)``.

    ``x = P X`` in homogeneous coordinates means ``x`` and ``P X`` are parallel,
    so their cross product vanishes.  Two of the three cross-product rows are
    independent, giving two equations per view::

        x * P[2] - P[0] = 0
        y * P[2] - P[1] = 0

    Four rows, four unknowns up to scale, smallest singular vector.  This
    minimises an algebraic residual rather than reprojection error, so it is
    biased for points near the epipoles and for wide baselines.  Good enough for
    a cheirality vote and for the scale-ratio estimate; not good enough to be
    the final structure in a system that cares.
    """
    P0 = np.asarray(P0, float)
    P1 = np.asarray(P1, float)
    x0 = np.asarray(x0, float).reshape(-1, 2)
    x1 = np.asarray(x1, float).reshape(-1, 2)
    n = len(x0)
    A = np.empty((n, 4, 4))
    A[:, 0] = x0[:, 0:1] * P0[2] - P0[0]
    A[:, 1] = x0[:, 1:2] * P0[2] - P0[1]
    A[:, 2] = x1[:, 0:1] * P1[2] - P1[0]
    A[:, 3] = x1[:, 1:2] * P1[2] - P1[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1, :]
    w = X[:, 3:4]
    return X[:, :3] / np.where(np.abs(w) < 1e-12, np.nan, w)


def cheirality_counts(
    R: np.ndarray, t: np.ndarray, p0: np.ndarray, p1: np.ndarray, K: np.ndarray
) -> tuple[int, np.ndarray]:
    """How many correspondences land in front of *both* cameras.

    "Cheirality" is the handedness constraint: a camera cannot see what is
    behind it.  It is the only thing that separates the four decompositions,
    and it is a statement about the physical world rather than about the
    algebra -- which is why no amount of least squares finds it for you.

    Returns the count and the ``(N, 3)`` triangulated points in camera 0.
    """
    K = np.asarray(K, float)
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([np.asarray(R, float), np.asarray(t, float).reshape(3, 1)])
    X0 = triangulate_dlt(P0, P1, p0, p1)
    X1 = (np.asarray(R, float) @ X0.T + np.asarray(t, float).reshape(3, 1)).T
    good = np.isfinite(X0).all(1) & (X0[:, 2] > 0) & (X1[:, 2] > 0)
    return int(good.sum()), X0


def recover_pose(E: np.ndarray, p0: np.ndarray, p1: np.ndarray, K: np.ndarray):
    """Pick the physically possible ``(R, t)`` by cheirality vote.

    Returns ``(R, t_unit, n_in_front, points)``.  ``t`` has unit norm, always,
    on every frame pair, and that is not a normalisation choice made here --
    it is what the data contains.
    """
    best = None
    for R, t in decompose_essential(E):
        n, X = cheirality_counts(R, t, p0, p1, K)
        if best is None or n > best[2]:
            best = (R, t / (np.linalg.norm(t) + 1e-300), n, X)
    return best


def sampson_error(E: np.ndarray, p0: np.ndarray, p1: np.ndarray, K: np.ndarray) -> np.ndarray:
    """First-order approximation to reprojection error, per correspondence.

    ``x1^T E x0`` is an *algebraic* residual with no units.  Dividing by the
    gradient norm turns it into an approximate distance in normalised image
    coordinates, which is what a RANSAC threshold should actually be measured
    in.  Multiply by ``fx`` for pixels.
    """
    K = np.asarray(K, float)
    Kinv = np.linalg.inv(K)
    q0 = np.column_stack([np.asarray(p0, float).reshape(-1, 2), np.ones(len(p0))]) @ Kinv.T
    q1 = np.column_stack([np.asarray(p1, float).reshape(-1, 2), np.ones(len(p1))]) @ Kinv.T
    Ex0 = q0 @ np.asarray(E, float).T
    Etx1 = q1 @ np.asarray(E, float)
    num = np.einsum("ij,ij->i", q1, Ex0)
    den = Ex0[:, 0] ** 2 + Ex0[:, 1] ** 2 + Etx1[:, 0] ** 2 + Etx1[:, 1] ** 2
    return np.abs(num) / np.sqrt(np.maximum(den, 1e-300))
