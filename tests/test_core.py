import numpy as np
import preland.episode as pe
import pytest
from preland import Scenario

from gpsfree import NavEKF, Sortie, body_rotation, displacement, fly, make_terrain, rangefinder
from gpsfree.mission import Axis, _handover_at
from gpsfree.world import DownCamera

RNG = np.random.default_rng
QUIET = DownCamera(pixel_noise=0.0, outlier_rate=0.0)


def frames(relief, tilt0=(0.2, -0.1), tilt1=(0.22, -0.08), move=(4.5, -2.0), seed=3, at=(60.0, 60.0)):
    T = make_terrain(seed, relief)
    C0 = np.array([*at, 40.0])
    C1 = C0 + [*move, 0.0]
    Rb0, Rb1 = body_rotation(*tilt0), body_rotation(*tilt1)
    R0, R1 = QUIET.R_wc(Rb0), QUIET.R_wc(Rb1)
    p0, p1 = QUIET.track(T, C0, R0, C1, R1, RNG(0))
    return T, p0, p1, R0, R1, rangefinder(T, C0, Rb0, RNG(0), sigma_rel=0.0), np.array(move)


# ------------------------------------------------------------ frames and sensors
def test_positive_dctk_angle_tilts_thrust_backwards():
    # dctk: a = -g tan(theta); thrust (body z) must lean towards -x for theta > 0
    assert body_rotation(0.2, 0.0)[0, 2] < 0
    assert body_rotation(0.0, 0.2)[1, 2] < 0


def test_rangefinder_ray_casts_along_the_tilted_axis():
    T = make_terrain(0)
    r = rangefinder(T, np.array([50.0, 50.0, 40.0]), body_rotation(0.3, 0.0), RNG(0), sigma_rel=0)
    assert r == pytest.approx(40.0 / np.cos(0.3))


def test_rangefinder_hits_a_roof_before_the_ground():
    T = make_terrain(3, relief=True)
    x0, y0, x1, y1, h = T.buildings[0]
    C = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 40.0])
    assert rangefinder(T, C, np.eye(3), RNG(0), sigma_rel=0) == pytest.approx(40.0 - h, abs=0.3)


# ------------------------------------------------------------ visual odometry
def test_homography_is_exact_over_flat_ground():
    T, p0, p1, R0, R1, r, move = frames(False)
    assert displacement("homography", p0, p1, QUIET.K, R0, R1, r, RNG(1))[:2] == pytest.approx(move, abs=1e-6)


def test_gyro_aided_epipolar_is_exact_on_flat_ground_and_buildings():
    for relief in (False, True):
        T, p0, p1, R0, R1, r, move = frames(relief)
        d = displacement("epipolar", p0, p1, QUIET.K, R0, R1, r, RNG(1))
        assert d[:2] == pytest.approx(move, abs=0.05 * np.linalg.norm(move))


def test_unaided_essential_is_degenerate_over_a_plane_even_without_noise():
    T, p0, p1, R0, R1, r, move = frames(False)
    flat = displacement("essential", p0, p1, QUIET.K, R0, R1, r, RNG(1))
    assert flat is None or np.linalg.norm(flat[:2] - move) > 0.5


def test_flow_tilt_compensation_on_flat_ground():
    T, p0, p1, R0, R1, r, move = frames(False, tilt0=(0.3, 0.0), tilt1=(0.3, 0.0), move=(0.5, 0.0))
    d = displacement("flow", p0, p1, QUIET.K, R0, R1, r, RNG(1))
    assert d[:2] == pytest.approx([0.5, 0.0], abs=0.03)     # uncompensated reads cos^2(0.3) = 9 % short


def test_gyro_aid_beats_unaided_essential_under_pixel_noise():
    cam = DownCamera(pixel_noise=0.5, outlier_rate=0.05)
    T = make_terrain(3, relief=True)
    rng = RNG(5)
    err = {"essential": [], "epipolar": []}
    for _ in range(12):
        C0 = np.array([*rng.uniform(30, 170, 2), 40.0])
        v = rng.normal(size=2)
        v = 4.8 * v / np.linalg.norm(v)
        Rb0, Rb1 = body_rotation(*rng.normal(0, 0.2, 2)), body_rotation(*rng.normal(0, 0.2, 2))
        R0, R1 = cam.R_wc(Rb0), cam.R_wc(Rb1)
        p0, p1 = cam.track(T, C0, R0, C0 + [*v, 0], R1, rng)
        r = rangefinder(T, C0, Rb0, rng)
        for m in err:
            d = displacement(m, p0, p1, cam.K, R0, R1, r, rng)
            err[m].append(np.nan if d is None else np.linalg.norm(d[:2] - v))
    assert np.nanmedian(err["epipolar"]) < 0.3 < 2.0 < np.nanmedian(err["essential"])


# ------------------------------------------------------------ navigation and control
def test_ins_only_drift_is_half_b_t_squared():
    nav = NavEKF(np.zeros(2), sigma_acc=0.0)
    for _ in range(1000):
        nav.predict(np.array([0.05, 0.0]), 0.02)              # pure bias, vehicle at rest
    assert nav.pos[0] == pytest.approx(0.5 * 0.05 * 20.0 ** 2, rel=0.01)


def test_velocity_updates_estimate_the_accelerometer_bias():
    nav = NavEKF(np.zeros(2))
    for _ in range(3000):
        nav.predict(np.array([0.05, -0.03]), 0.02)
        nav.update_velocity(np.zeros(2))
    assert nav.x[4:] == pytest.approx([0.05, -0.03], abs=0.01)


def test_velocity_gate_rejects_a_wild_measurement():
    nav = NavEKF(np.zeros(2))
    assert not nav.update_velocity(np.array([5.0, 0.0]))


def test_dctk_cascade_axis_reaches_its_setpoint():
    ax = Axis()
    for _ in range(int(15 / 0.004)):
        ax.step(20.0, ax.p, ax.v, 0.0, 0.004)
    assert ax.p == pytest.approx(20.0, abs=0.3)


# ------------------------------------------------------------ hand-over to preland
def test_handover_places_preland_vehicle_at_the_given_offset_and_restores():
    original = pe.initial_state
    with _handover_at(np.array([3.0, -4.0])):
        s = pe.initial_state(Scenario(), RNG(0))
        assert s.position[:2] == pytest.approx([3.0, -4.0])
        assert s.position[2] == Scenario().start_height_m
    assert pe.initial_state is original


def test_short_sortie_returns_and_lands_on_homography():
    r = fly(Sortie(method="homography", route=((40.0, 0.0), (40.0, 40.0), (0.0, 0.0))), seed=1)
    assert r["returned"] and r["handover_error_m"] < 3.0 and r["landed"]
