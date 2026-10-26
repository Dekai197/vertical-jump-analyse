"""Generic mode: skeleton + neon joint trails + live knee/hip angles."""

from __future__ import annotations

import numpy as np

from . import render as R
from .landmarks import SIDES
from .pipeline import Session

TRAIL_S = 1.5


def summarize(s: Session) -> dict:
    knee, hip = s.knee_angle(), s.hip_angle()
    i = int(np.nanargmin(knee))
    return {
        "min_knee_angle_deg": round(float(knee[i]), 1),
        "min_knee_angle_t_s": round(float(s.t[i]), 3),
        "min_hip_angle_deg": round(float(np.nanmin(hip)), 1),
    }


def make_drawer(s: Session):
    near = set(SIDES[s.side].values())
    hip, ankle = s.joint("hip"), s.joint("ankle")
    knee_a, hip_a = s.knee_angle(), s.hip_angle()

    def draw(frame: np.ndarray, i: int) -> None:
        j0 = int(np.searchsorted(s.t, s.t[i] - TRAIL_S))
        R.glow_polylines(frame, [(hip[j0:i + 1], R.CYAN), (ankle[j0:i + 1], R.MAGENTA)])
        if s.raw.detected[i]:
            R.draw_skeleton(frame, s.xy[i], s.vis[i], near)
        R.hud(frame, [(f"t {s.t[i]:.2f}s", R.WHITE),
                      (f"knee {knee_a[i]:.0f} deg", R.YELLOW),
                      (f"hip  {hip_a[i]:.0f} deg", R.YELLOW)],
              title=f"formtrace | {s.side} side")
    return draw
