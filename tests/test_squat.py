import numpy as np

from formtrace.squat import hip_rise_index, segment_reps


def _squat_signal(fps=60.0):
    """Top at y=0, 3 reps of depth 300 px, separated by pauses, plus a walk-out."""
    t = np.arange(0, 14, 1 / fps)
    y = np.zeros_like(t)
    x = np.full_like(t, 500.0)
    # walk-out: 1-2.5 s, bar moves 250 px horizontally and dips 90 px
    w = (t > 1) & (t < 2.5)
    x[w] = 500 + 250 * np.sin(np.pi * (t[w] - 1) / 1.5)
    y[w] = 90 * np.sin(np.pi * (t[w] - 1) / 1.5)
    for c in (5, 8, 11):  # each rep: 0.8 s down, 0.8 s up, cosine profile
        m = (t >= c - 0.8) & (t <= c + 0.8)
        y[m] = 300 * (1 - np.cos(np.pi * (t[m] - (c - 0.8)) / 0.8)) / 2
    return t, y, x


def test_segment_reps_finds_three_and_rejects_walkout():
    t, y, x = _squat_signal()
    reps = segment_reps(y, t, min_drop_px=60, x=x)
    assert len(reps) == 3
    for r, c in zip(reps, (5, 8, 11)):
        assert abs(t[r["bottom"]] - c) < 0.02
        assert abs(t[r["start"]] - (c - 0.8)) < 0.15
        assert abs(t[r["end"]] - (c + 0.8)) < 0.15


def test_segment_reps_ignores_gaps_and_spikes():
    t, y, x = _squat_signal()
    y = y.copy()
    y[385:405] = np.nan          # occlusion gap during the pause at ~6.5 s
    y[700] = 400                 # single-frame glitch
    reps = segment_reps(y, t, min_drop_px=60, x=x)
    assert len(reps) == 3


def test_hip_rise_index():
    hip = np.array([300, 250, 200, 150, 100, 50, 0], float)
    same = hip.copy()
    lag = np.array([300, 280, 260, 240, 160, 80, 0], float)
    r1, _ = hip_rise_index(hip, same, 0, 6)
    r2, _ = hip_rise_index(hip, lag, 0, 6)
    assert abs(r1 - 1.0) < 1e-9
    assert r2 < 0.7
