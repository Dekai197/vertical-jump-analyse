import numpy as np
import pytest

from formtrace import filters, geometry
from formtrace.video import resolve_times
from formtrace.landmarks import camera_side, SIDES


def test_joint_angle_basic():
    assert geometry.joint_angle([0, 0], [1, 0], [2, 0]) == pytest.approx(180)
    assert geometry.joint_angle([0, 1], [0, 0], [1, 0]) == pytest.approx(90)
    a = geometry.joint_angle(np.array([[1, 1]]), np.array([[0, 0]]), np.array([[1, 0]]))
    assert a[0] == pytest.approx(45)


def test_lean_from_vertical():
    # image coords: y grows downward, so "up" is negative y
    assert geometry.lean_from_vertical([0, 10], [0, 0]) == pytest.approx(0)
    assert geometry.lean_from_vertical([0, 0], [10, 0]) == pytest.approx(90)
    assert geometry.lean_from_vertical([0, 10], [-10, 0]) == pytest.approx(45)


def test_fill_gaps_interpolates_inside_only():
    x = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
    ok = np.isfinite(x)
    np.testing.assert_allclose(filters.fill_gaps(x, ok), [np.nan, 1, 2, 3, np.nan])


def test_fill_gaps_respects_max_gap_and_per_landmark_mask():
    x = np.arange(10, dtype=float)
    ok = np.ones(10, bool)
    ok[2:3] = False   # 1-sample gap -> filled
    ok[5:9] = False   # 4-sample gap -> left NaN
    out = filters.fill_gaps(x, ok, max_gap=2)
    assert out[2] == 2 and np.all(np.isnan(out[5:9])) and out[9] == 9
    xy = np.random.default_rng(0).normal(size=(6, 3, 2))
    m = np.ones((6, 3), bool)
    m[2, 1] = False
    filled = filters.fill_gaps(xy, m)
    np.testing.assert_allclose(filled[2, 1], (xy[1, 1] + xy[3, 1]) / 2)
    np.testing.assert_allclose(filled[2, 0], xy[2, 0])


def test_smooth_does_not_bridge_or_erode_nan_gaps():
    y = np.r_[np.linspace(0, 1, 20), [np.nan] * 5, np.linspace(1, 0, 20)]
    out = filters.smooth(y, 7)
    assert np.isnan(out[20:25]).all()
    assert np.isfinite(out[:20]).all() and np.isfinite(out[25:]).all()


def test_find_peaks_prominence():
    x = np.array([0, 5, 0, 1, 0.5, 1.2, 0, 8, 0], dtype=float)
    assert filters.find_peaks(x, 3) == [1, 7]
    assert 5 in filters.find_peaks(x, 0.5)


def test_savgol_preserves_quadratic_and_reduces_noise():
    t = np.linspace(-1, 1, 101)
    y = 3 * t ** 2 - t + 2
    np.testing.assert_allclose(filters.smooth(y, 9)[10:-10], y[10:-10], atol=1e-9)
    rng = np.random.default_rng(0)
    noisy = y + rng.normal(0, 0.1, len(y))
    assert np.std(filters.smooth(noisy, 9) - y) < np.std(noisy - y)


def test_window_is_odd():
    for fps in (24, 30, 59.94, 120, 240):
        assert filters.window_for(fps, 0.12) % 2 == 1


def test_derivative_uses_real_timestamps():
    t = np.array([0, 0.033, 0.066, 0.133, 0.166])  # one dropped frame
    x = 2.0 * t
    np.testing.assert_allclose(filters.derivative(x, t), 2.0)


def test_resolve_times_prefers_pts_and_falls_back():
    times, src = resolve_times([0, 33.3, 66.7, 100.0], 26.4)
    assert src == "pts" and times[-1] == pytest.approx(0.1)
    times, src = resolve_times([0, 0, 0, 0], 25)
    assert src == "nominal_fps" and times[-1] == pytest.approx(3 / 25)


def test_camera_side():
    vis = np.full((10, 33), 0.2)
    for j in (SIDES["right"]["hip"], SIDES["right"]["knee"], SIDES["right"]["ankle"]):
        vis[:, j] = 0.9
    assert camera_side(vis) == "right"
