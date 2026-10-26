"""Squat analysis: rep segmentation, depth, knee/hip angles, hip-rise timing, bar path."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import filters, geometry
from . import render as R
from .barbell import BarTrack
from .landmarks import SIDES
from .pipeline import Freeze, Session


@dataclass
class Calib:
    px_per_cm: float | None
    source: str  # "plate" | "two_points" | "none"


def make_calib(bar: BarTrack, plate_cm: float | None, two_points: tuple | None) -> Calib:
    if two_points:
        x1, y1, x2, y2, cm = two_points
        return Calib(float(np.hypot(x2 - x1, y2 - y1) / cm), "two_points")
    if plate_cm and bar.radius_px and bar.source.startswith("plate"):
        return Calib(2 * bar.radius_px / plate_cm, "plate")
    return Calib(None, "none")


def segment_reps(y: np.ndarray, t: np.ndarray, min_drop_px: float,
                 onset_frac: float = 0.1, min_phase_s: float = 0.2,
                 x: np.ndarray | None = None, max_sway_ratio: float = 0.5) -> list[dict]:
    """Find squat reps from a vertical trajectory (image y grows downward).

    - works per run of valid samples, so occlusion gaps never create fake reps;
    - needs a drop of ``min_drop_px`` that returns to the top on both sides;
    - start/end are where vertical speed falls below ``onset_frac`` of that
      phase's peak speed (a velocity onset, as in bar-velocity tools), so a
      slow sag while bracing at the top does not count as descent;
    - with ``x`` given, a "rep" whose horizontal travel exceeds
      ``max_sway_ratio`` x its vertical travel is rejected (walk-out / re-rack).
    """
    reps = []
    y = filters.medfilt(y, 5)  # single-frame tracking glitches must not create "bottoms"
    v_all = filters.derivative(y, t)
    for a0, b0 in filters._runs(np.isfinite(y)):
        yy, v = y[a0:b0], v_all[a0:b0]
        bottoms = filters.find_peaks(yy, min_drop_px)
        for k, p in enumerate(bottoms):
            lo = bottoms[k - 1] if k else 0
            hi = bottoms[k + 1] if k + 1 < len(bottoms) else len(yy) - 1
            lb = lo + int(np.argmin(yy[lo:p + 1]))
            rb = p + int(np.argmin(yy[p:hi + 1]))
            if yy[p] - yy[lb] < min_drop_px or yy[p] - yy[rb] < min_drop_px:
                continue  # the rep must start and finish near the top
            vd = lb + int(np.argmax(v[lb:p + 1]))      # fastest descent
            va = p + int(np.argmin(v[p:rb + 1]))       # fastest ascent (most negative)
            slow_d = np.nonzero(v[lb:vd + 1] < onset_frac * v[vd])[0]
            slow_a = np.nonzero(-v[va:rb + 1] < onset_frac * -v[va])[0]
            start = a0 + lb + int(slow_d.max()) if len(slow_d) else a0 + lb
            end = a0 + va + int(slow_a.min()) if len(slow_a) else a0 + rb
            bottom = a0 + p
            if t[bottom] - t[start] < min_phase_s or t[end] - t[bottom] < min_phase_s:
                continue
            if x is not None:
                seg_x = x[start:end + 1]
                travel = y[bottom] - min(y[start], y[end])
                if np.isfinite(seg_x).any() and np.nanmax(seg_x) - np.nanmin(seg_x) > max_sway_ratio * travel:
                    continue
            reps.append({"start": start, "bottom": bottom, "end": end})
    return reps


def _num(v, nd=1):
    v = float(v)
    return round(v, nd) if np.isfinite(v) else None


def _nanmin(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmin(x)) if np.isfinite(x).any() else float("nan")


def hip_rise_index(hip_y, sh_y, bottom: int, end: int, frac: float = 0.5) -> tuple[float, int]:
    """Shoulder rise / hip rise from the bottom until the hip completes ``frac`` of its rise.

    1.0 = hips and shoulders rise together (constant trunk angle);
    < 1 = hips rise first (trunk tips forward); > 1 = chest leads.
    """
    total = hip_y[bottom] - hip_y[end]
    if not np.isfinite(total) or total <= 0:
        return float("nan"), bottom
    seg = hip_y[bottom:end + 1]
    m = bottom + int(np.argmax(seg <= hip_y[bottom] - frac * total))
    hip_rise = hip_y[bottom] - hip_y[m]
    sh_rise = sh_y[bottom] - sh_y[m]
    ok = np.isfinite(sh_rise) and hip_rise > 0
    return (float(sh_rise / hip_rise) if ok else float("nan")), m


def analyze(s: Session, bar: BarTrack, calib: Calib, hip_rise_warn: float = 0.7,
            lean_warn_deg: float = 5.0, min_drop_frac: float = 0.2) -> dict:
    t = s.t
    hip, knee = s.joint("hip"), s.joint("knee")
    heel, toe = s.joint("heel"), s.joint("toe")
    L = s.leg_length()
    knee_a, hip_a = s.knee_angle(), s.hip_angle()
    facing = 1.0 if np.nanmedian(toe[:, 0] - heel[:, 0]) >= 0 else -1.0
    midfoot_x = (heel[:, 0] + toe[:, 0]) / 2

    bxy = filters.fill_gaps(bar.xy, np.isfinite(bar.xy[:, 0]))
    bxy = filters.smooth(bxy, filters.window_for(s.info.effective_fps, 0.1, 3))
    bvy = filters.derivative(bxy[:, 1], t)

    def length(px: float) -> dict:
        d = {"px": _num(px), "pct_leg": _num(100 * px / L)}
        if calib.px_per_cm:
            d["cm"] = _num(px / calib.px_per_cm)
        return d

    # The plate track is usually far more complete than the pose (the plate
    # hides the head, which the pose detector relies on), so it drives rep
    # segmentation when available; the hip is the fallback.
    use_bar = bar.source in ("plate_auto", "plate_manual", "box_manual") and bar.coverage >= 0.8
    rep_signal = bxy[:, 1] if use_bar else hip[:, 1]
    # The bar rests on the upper back, and in a side view the plate covers the
    # shoulder joint, where MediaPipe then guesses. So with a tracked bar the
    # trunk is measured hip -> bar instead of hip -> shoulder landmark.
    upper = bxy if use_bar else s.joint("shoulder")
    lean = geometry.lean_from_vertical(hip, upper)
    reps, flags = [], []
    reps_idx = segment_reps(rep_signal, t, min_drop_frac * L, x=bxy[:, 0] if use_bar else None)
    for k, r in enumerate(reps_idx, 1):
        a, p, e = r["start"], r["bottom"], r["end"]
        if use_bar:  # body "bottom" = deepest hip within the bar-defined rep, if seen
            hseg = hip[a:e + 1, 1]
            if np.isfinite(hseg).any():
                p = a + int(np.nanargmax(hseg))
        thigh = geometry.dist(hip[p], knee[p])
        ratio, m = hip_rise_index(hip[:, 1], upper[:, 1], p, e)
        below = hip[p, 1] > knee[p, 1] if np.all(np.isfinite([hip[p, 1], knee[p, 1]])) else None
        rep = {
            "rep": k,
            "t_start_s": _num(t[a], 3), "t_bottom_s": _num(t[p], 3), "t_end_s": _num(t[e], 3),
            "descent_s": _num(t[p] - t[a], 3), "ascent_s": _num(t[e] - t[p], 3),
            "min_knee_angle_deg": _num(_nanmin(knee_a[a:e + 1])),
            "hip_angle_at_bottom_deg": _num(hip_a[p]),
            "trunk_lean_at_bottom_deg": _num(lean[p]),
            "hip_below_knee": None if below is None else bool(below),
            "hip_minus_knee_height_pct_thigh": _num(100 * (hip[p, 1] - knee[p, 1]) / thigh),
            "hip_rise_index": _num(ratio, 2),
            "trunk_lean_change_early_ascent_deg": _num(lean[m] - lean[p]),
            "_idx": {"start": a, "bottom": p, "end": e, "mid_ascent": m},
        }
        cov = np.isfinite(bar.xy[a:e + 1, 0]).mean()
        if bar.source != "none" and cov >= 0.7:
            bx = bxy[a:e + 1, 0]
            pb = a + int(np.nanargmax(bxy[a:e + 1, 1]))
            rep["bar"] = {
                "horizontal_range": length(np.nanmax(bx) - np.nanmin(bx)),
                "max_drift_from_start": length(np.nanmax(np.abs(bx - bxy[a, 0]))),
                "vertical_travel": length(bxy[pb, 1] - np.nanmin(bxy[a:e + 1, 1])),
                "coverage": _num(cov, 2),
            }
            if np.isfinite(midfoot_x[pb]):
                rep["bar"]["offset_from_midfoot_at_bottom"] = length(facing * (bxy[pb, 0] - midfoot_x[pb]))
            rep["_idx"]["bar_bottom"] = pb
            if calib.px_per_cm and e > pb:
                up = -bvy[pb:e + 1] / calib.px_per_cm / 100.0  # m/s, positive = upward
                rep["bar"]["mean_concentric_velocity_mps"] = _num(
                    (bxy[pb, 1] - bxy[e, 1]) / calib.px_per_cm / 100.0 / (t[e] - t[pb]), 2)
                rep["bar"]["peak_concentric_velocity_mps"] = _num(np.nanmax(up), 2)
        lc = rep["trunk_lean_change_early_ascent_deg"]
        if rep["hip_rise_index"] is not None and lc is not None and ratio < hip_rise_warn \
                and lc > lean_warn_deg:
            flags.append(f"第 {k} 次：起身前半程{'杠铃' if use_bar else '肩部'}上升只有髋部的 {ratio:.0%}，躯干前倾增加 "
                         f"{lc:.0f}°——可能存在“先抬臀”。")
        if rep["hip_below_knee"] is False:
            flags.append(f"第 {k} 次：最低点髋关节点仍高于膝关节点（近似判断为未过平行）。")
        reps.append(rep)

    if not reps:
        flags.append("没有识别到完整的深蹲动作（髋部下降幅度不足腿长的 20%）。")
    summary = {"reps": len(reps), "rep_signal": "bar" if use_bar else "hip",
               "trunk_upper_point": "bar" if use_bar else "shoulder",
               "bar_source": bar.source,
               "bar_tracking_coverage": round(bar.coverage, 2),
               "calibration": calib.source,
               "px_per_cm": round(calib.px_per_cm, 3) if calib.px_per_cm else None}
    knees = [r["min_knee_angle_deg"] for r in reps if r["min_knee_angle_deg"] is not None]
    if knees:
        summary["deepest_knee_angle_deg"] = min(knees)
    if s.frac_valid("ankle") < 0.8:
        caveats_extra = [f"近侧脚踝只有 {s.frac_valid('ankle'):.0%} 的帧在画面内且可见：膝角、"
                         "“杠铃相对脚掌中点”在脚不可见时无法计算（显示为空）。拍摄时请让脚完整入镜。"]
    else:
        caveats_extra = []
    caveats = [
        "“过平行”以 MediaPipe 髋关节中心点与膝关节点的高度比较近似，不等同于力量举裁判标准（髋折痕低于膝盖上沿）。",
        ("躯干角与“上升比”用 髋→杠铃片中心 连线计算（杠铃片挡住了肩关节点）。" if use_bar else
         "躯干角与“上升比”用 髋→肩 关节点连线计算。") +
        f"“先抬臀”提示的阈值（上升比 < {hip_rise_warn}、前倾增加 > {lean_warn_deg}°）是本工具的启发式设定，"
        "不是经过验证的标准，可用参数调整。",
        "杠铃轨迹只追踪画面近侧的杠铃片中心；远端被身体遮挡，单机位无法测两端。",
    ]
    caveats += caveats_extra
    if bar.source == "wrist_proxy":
        caveats.append("未找到杠铃片，轨迹改用近侧手腕代替：深蹲时手腕与杠铃并不重合，轨迹形状仅供参考。")
    if calib.source == "plate":
        caveats.append("厘米/速度换算假设杠铃片直径为你输入的值。举重比赛用杠铃片常见规格为 450 mm"
                       "（该数值来自厂商页面转述，未查到 IWF 规则原文），健身房铁片尺寸差异大，请以实物测量为准。")
    elif calib.source == "none":
        caveats.append("未标定比例尺：距离以像素和“腿长百分比”表示，速度不输出。可用 --plate-cm 或 --calib 标定。")
    return {"summary": summary, "reps": reps, "flags": flags, "caveats": caveats,
            "_bar_xy": bxy}


def _v(v, unit="", fmt="{}"):
    return "—" if v is None else fmt.format(v) + unit


def sections(res: dict) -> list[dict]:
    out = []
    for r in res["reps"]:
        depth = {True: "过平行（近似）", False: "未过平行（近似）", None: "无法判断（膝不可见）"}[r["hip_below_knee"]]
        lines = [
            f"- 时间 {r['t_start_s']}–{r['t_end_s']} s；下蹲 {r['descent_s']} s，起身 {r['ascent_s']} s",
            f"- 最小膝关节夹角 {_v(r['min_knee_angle_deg'], '°')}；最低点髋角 {_v(r['hip_angle_at_bottom_deg'], '°')}，"
            f"躯干前倾 {_v(r['trunk_lean_at_bottom_deg'], '°')}",
            f"- 最低点髋-膝高度差 {_v(r['hip_minus_knee_height_pct_thigh'], '%')} 大腿长（正值=髋低于膝）→ {depth}",
            f"- 起身前半程 上身/髋 上升比 {_v(r['hip_rise_index'])}（1=同步；<1=髋先起），"
            f"躯干前倾变化 {_v(r['trunk_lean_change_early_ascent_deg'], '°', '{:+}')}",
        ]
        if "bar" in r:
            b = r["bar"]

            def L(d):
                if d is None:
                    return "—"
                return f"{d['cm']} cm" if "cm" in d else f"{d['px']} px（{d['pct_leg']}% 腿长）"
            lines += [f"- 杠铃水平摆动范围 {L(b['horizontal_range'])}，偏离起点最大 {L(b['max_drift_from_start'])}，"
                      f"垂直行程 {L(b['vertical_travel'])}",
                      f"- 最低点杠铃相对脚掌中点 {L(b.get('offset_from_midfoot_at_bottom'))}（正值=偏向脚尖）"]
            if "mean_concentric_velocity_mps" in b:
                lines.append(f"- 向心平均速度 {_v(b['mean_concentric_velocity_mps'], ' m/s')}，"
                             f"峰值 {_v(b['peak_concentric_velocity_mps'], ' m/s')}")
        out.append({"title": f"第 {r['rep']} 次", "lines": lines})
    return out


def make_drawer(s: Session, bar: BarTrack, res: dict, hip_rise_warn: float = 0.7):
    near = set(SIDES[s.side].values())
    reps = res["reps"]
    bxy = res["_bar_xy"]
    have_bar = bar.source != "none"
    knee_a, hip_a = s.knee_angle(), s.hip_angle()
    rep_of = np.full(s.info.n_frames, -1)
    for k, r in enumerate(reps):
        rep_of[r["_idx"]["start"]:] = k  # a rep's path stays on screen until the next starts

    def rep_path(k: int, upto: int):
        ix = reps[k]["_idx"]
        a, e = ix["start"], min(ix["end"], upto)
        pb = ix.get("bar_bottom", ix["bottom"])
        return [(bxy[a:min(pb, e) + 1], R.CYAN), (bxy[pb:e + 1], R.MAGENTA)] if e > pb \
            else [(bxy[a:e + 1], R.CYAN)]

    def draw(frame: np.ndarray, i: int) -> None:
        k = rep_of[i]
        if s.raw.detected[i]:
            R.draw_skeleton(frame, s.xy[i], s.vis[i], near)
        if have_bar:
            for j in range(max(0, k)):
                ix = reps[j]["_idx"]
                R.glow_polylines(frame, [(bxy[ix["start"]:ix["end"] + 1], R.GREY)], thickness=1.5)
            if k >= 0:
                ix = reps[k]["_idx"]
                ys = bxy[ix["start"]:ix["end"] + 1, 1]
                if np.isfinite(bxy[ix["start"], 0]) and np.isfinite(ys).any():
                    m = 40 * R.ui_scale(frame)
                    R.dashed_vline(frame, bxy[ix["start"], 0], np.nanmin(ys) - m, np.nanmax(ys) + m)
                R.glow_polylines(frame, rep_path(k, i))
            if np.all(np.isfinite(bar.xy[i])):
                c = tuple(int(v) for v in bar.xy[i])
                rad = int(bar.radius_px) if bar.radius_px else int(8 * R.ui_scale(frame))
                cv2.circle(frame, c, rad, R.CYAN, max(1, int(2 * R.ui_scale(frame))), cv2.LINE_AA)
        lines = [(f"knee {knee_a[i]:.0f} deg" if np.isfinite(knee_a[i]) else "knee --", R.YELLOW)]
        if k >= 0:
            lines.insert(0, (f"REP {k + 1}/{len(reps)}", R.WHITE))
        R.hud(frame, lines, title=f"SQUAT | bar: {bar.source}")

    freezes = []
    for r in reps:
        ix = r["_idx"]
        p = ix["bottom"]

        def bottom_note(fr, r=r, p=p):
            if np.isfinite(knee_a[p]):
                R.angle_arc(fr, s.joint("hip")[p], s.joint("knee")[p], s.joint("ankle")[p], R.YELLOW,
                            f"{knee_a[p]:.0f}")
            if np.isfinite(hip_a[p]):
                R.angle_arc(fr, s.joint("shoulder")[p], s.joint("hip")[p], s.joint("knee")[p], R.ORANGE,
                            f"{hip_a[p]:.0f}")
            below = r["hip_below_knee"]
            txt = {True: "below parallel", False: "ABOVE parallel", None: "depth ?"}[below]
            R.banner(fr, f"REP {r['rep']} BOTTOM  {txt}", {True: R.LIME, False: R.RED, None: R.GREY}[below])

        def end_note(fr, r=r):
            lines = [(f"down {r['descent_s']:.2f}s  up {r['ascent_s']:.2f}s", R.WHITE)]
            if r["min_knee_angle_deg"] is not None:
                lines.insert(0, (f"min knee {r['min_knee_angle_deg']:.0f} deg", R.YELLOW))
            if r["hip_rise_index"] is not None:
                lines.append((f"upper/hip rise {r['hip_rise_index']:.2f}",
                              R.RED if r["hip_rise_index"] < hip_rise_warn else R.LIME))
            if "bar" in r and r["bar"]["horizontal_range"]["px"] is not None:
                b = r["bar"]["horizontal_range"]
                lines.append((f"bar sway {b['cm']:.1f} cm" if b.get("cm") is not None
                              else f"bar sway {b['pct_leg']:.0f}% leg", R.CYAN))
                v = r["bar"].get("mean_concentric_velocity_mps")
                if v is not None:
                    lines.append((f"mean v {v:.2f} m/s", R.CYAN))
            R.hud(fr, lines, corner="bl", title=f"REP {r['rep']} DONE")
        freezes.append(Freeze(p, 0.7, bottom_note))
        freezes.append(Freeze(ix["end"], 1.2, end_note))
    return draw, freezes
