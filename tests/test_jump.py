"""Synthetic two-foot approach jump with known event times."""
import numpy as np
import pytest

from formtrace import jump
from formtrace.landmarks import SIDES
from formtrace.pipeline import Session
from formtrace.pose import PoseTrack
from formtrace.video import VideoInfo

FPS = 60.0
GROUND = 1500.0
PEN_TD, FINAL_TD, TAKEOFF, LANDING = 0.30, 0.80, 1.40, 1.90


def _foot(t, x_plant_a, x_plant_b, swing_start, swing_end, start_planted=True):
    """x and height of a foot that swings from a to b between swing_start and swing_end,
    and is airborne between TAKEOFF and LANDING."""
    x = np.where(t < swing_start, x_plant_a, x_plant_b).astype(float)
    h = np.zeros_like(t)
    m = (t >= swing_start) & (t < swing_end)
    u = (t[m] - swing_start) / (swing_end - swing_start)
    x[m] = x_plant_a + (x_plant_b - x_plant_a) * u
    h[m] = 60 * np.sin(np.pi * u)
    f = (t > TAKEOFF) & (t < LANDING)
    u = (t[f] - TAKEOFF) / (LANDING - TAKEOFF)
    h[f] = 250 * 4 * u * (1 - u)
    x[t >= LANDING] = x_plant_b + 50
    return x, h


def make_session():
    t = np.arange(0, 2.5, 1 / FPS)
    T = len(t)
    xy = np.full((T, 33, 2), np.nan)
    # right foot: penultimate step, swings 0.0-0.30, planted from 0.30
    xr, hr = _foot(t, 100, 500, 0.0, PEN_TD)
    # left foot: final step, swings 0.50-0.80, planted from 0.80
    xl, hl = _foot(t, 300, 700, 0.50, FINAL_TD)
    hip_x = np.interp(t, [0, 0.8, 1.4, 2.5], [200, 600, 680, 800])
    hip_y = GROUND - 400 + np.interp(t, [0, 0.8, 1.1, 1.4, 1.65, 1.9, 2.5],
                                     [0, 20, 100, 0, -250, 0, 30])
    for sd, fx, fh in (("left", xl, hl), ("right", xr, hr)):
        j = SIDES[sd]
        fy = GROUND - fh
        xy[:, j["heel"]] = np.c_[fx, fy]
        xy[:, j["toe"]] = np.c_[fx + 60, fy]
        xy[:, j["ankle"]] = np.c_[fx + 5, fy - 30]
        xy[:, j["hip"]] = np.c_[hip_x, hip_y]
        xy[:, j["knee"]] = np.c_[(hip_x + fx) / 2 + 40, (hip_y + fy) / 2]
        xy[:, j["shoulder"]] = np.c_[hip_x + 20, hip_y - 300]
    vis = np.ones((T, 33))
    info = VideoInfo("synthetic", 1080, 1920, FPS, T, t, "pts")
    raw = PoseTrack(xy, vis, np.ones(T, bool), t)
    return Session(info, raw, xy, vis, "right", {}, np.ones((T, 33), bool))


def test_jump_events_match_ground_truth():
    s = make_session()
    res = jump.analyze(s)
    e = res["events"]
    tol = 2.5 / FPS
    assert e["penultimate_touchdown"]["t_s"] == pytest.approx(PEN_TD, abs=tol)
    assert e["final_touchdown"]["t_s"] == pytest.approx(FINAL_TD, abs=tol)
    assert e["final_touchdown"]["foot"] == "left"
    assert e["takeoff"]["t_s"] == pytest.approx(TAKEOFF, abs=tol)
    assert e["landing"]["t_s"] == pytest.approx(LANDING, abs=tol)
    flight = res["summary"]["flight_time_s"]
    assert flight == pytest.approx(LANDING - TAKEOFF, abs=2 / FPS)
    # h = g t^2 / 8
    assert res["summary"]["jump_height_from_flight_cm"] == pytest.approx(100 * 9.81 * flight ** 2 / 8, abs=0.1)
    # hip lowest at 1.1 s inside the plant
    assert e["lowest_hip"]["t_s"] == pytest.approx(1.1, abs=tol)


def test_velocity_loss_sign():
    s = make_session()
    summ = jump.analyze(s)["summary"]
    # hip speed: 500 px/s before 0.8 s, then 133 px/s -> loss across the penultimate step
    assert summ["velocity_change_penultimate_pct"] > 0
    assert summ["approach_velocity"]["px_s"] == pytest.approx(500, rel=0.05)


def test_no_flight_reports_flag():
    s = make_session()
    s.xy[:, :, 1] = np.where(np.isfinite(s.xy[:, :, 1]), GROUND, np.nan)  # glue everything to the floor
    res = jump.analyze(s)
    assert not res["events"] and res["flags"]
