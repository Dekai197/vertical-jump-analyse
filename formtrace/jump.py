"""Approach-jump analysis: ground contacts, plant (brake), take-off, flight.

Ground contact from a fixed side-view camera is inferred per foot:
a foot is on the ground when its lowest point (heel or toe) is (a) nearly
stationary and (b) not clearly airborne, i.e. less than 20% of leg length
above the local ground level. "Local ground" is the lowest image position the
foot reaches within +/-0.5 s; it is only a coarse airborne test because the
floor line shifts in the image when the athlete moves toward/away from the
camera, so stationarity does the fine work. The jump's flight is found first (both feet far above local ground),
then take-off / landing / plant are traced outward from it.

Every time is quantised by the frame interval; the report states +/- 1 frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import filters
from . import render as R
from .landmarks import SIDES
from .pipeline import Freeze, Session

G = 9.81  # m/s^2
# Hip-to-ankle (thigh + shank) ~= 0.491 x body height: Drillis & Contini (1966) proportions as
# reproduced in Winter's textbook (Fig. 4.1). UNVERIFIED: the value is from memory; the original
# table could not be checked when this was written. Only used for the optional --height-cm scale.
LEG_FRACTION_OF_HEIGHT = 0.491


def _rolling_max(x: np.ndarray, half: int) -> np.ndarray:
    pad = np.concatenate([np.full(half, np.nan), x, np.full(half, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(win, axis=1)


def foot_contacts(s: Session, side: str, L: float, near_frac: float = 0.2,
                  speed_frac: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame contact mask for one foot, plus its bottom point and height above ground."""
    j = SIDES[side]
    heel, toe = s.xy[:, j["heel"]], s.xy[:, j["toe"]]
    lower_is_toe = toe[:, 1] >= heel[:, 1]
    bottom = np.where(lower_is_toe[:, None], toe, heel)
    bottom = np.where(np.isfinite(bottom), bottom, np.where(np.isfinite(heel), heel, toe))
    half = max(2, int(round(0.5 * s.info.effective_fps)))
    ground = _rolling_max(bottom[:, 1], half)
    height = ground - bottom[:, 1]                      # px above local ground (>= 0)
    # Speed of whichever point is lower: during roll-off the toe is stationary.
    v_heel = np.linalg.norm(filters.derivative(heel, s.t), axis=1)
    v_toe = np.linalg.norm(filters.derivative(toe, s.t), axis=1)
    speed = np.where(lower_is_toe, v_toe, v_heel)
    near = (height < near_frac * L) & np.isfinite(height)
    contact = near & (speed < speed_frac * L) & np.isfinite(speed)
    max_gap = max(1, int(round(0.1 * s.info.effective_fps)))
    return _clean_mask(contact, near, max_gap), bottom, height


def _clean_mask(m: np.ndarray, near: np.ndarray, max_gap: int) -> np.ndarray:
    """Bridge short breaks in a contact while the foot stays near the ground
    (a one-frame landmark jump on a planted foot), then drop 1-frame islands."""
    m = m.copy()
    runs = filters._runs(m)
    for (a1, b1), (a2, b2) in zip(runs, runs[1:]):
        if a2 - b1 <= max_gap and near[b1:a2].all():
            m[b1:a2] = True
    for i in range(1, len(m) - 1):
        if m[i] and not m[i - 1] and not m[i + 1]:
            m[i] = False
    return m


def _interval_ending_at(mask: np.ndarray, end: int, slack: int = 2) -> tuple[int, int] | None:
    """The contact interval that ends within ``slack`` frames before/at ``end``."""
    for a, b in reversed(filters._runs(mask)):
        if a <= end and b - 1 >= end - slack:
            return int(a), int(min(b - 1, end))
    return None


def analyze(s: Session, px_per_m: float | None = None, scale_source: str = "none",
            flight_frac: float = 0.2) -> dict:
    t = s.t
    T = len(t)
    fps = s.info.effective_fps
    dt = 1.0 / fps
    L = s.leg_length()
    cl, bl, hl = foot_contacts(s, "left", L)
    cr, br, hr = foot_contacts(s, "right", L)
    any_contact = cl | cr

    flags, caveats = [], []
    res: dict = {"summary": {}, "events": {}, "flags": flags, "caveats": caveats}

    # 1) flight core: both feet well above their local ground
    airborne = (hl > flight_frac * L) & (hr > flight_frac * L)
    runs = [(a, b) for a, b in filters._runs(airborne) if b - a >= 2]
    if not runs:
        flags.append("没有找到腾空阶段（双脚同时离地高度 > 腿长的 20%）。请确认视频包含完整起跳与落地。")
        return res
    fa, fb = max(runs, key=lambda r: r[1] - r[0])

    # 2) take-off = last contact frame before flight; landing = first contact after it
    before = np.nonzero(any_contact[:fa])[0]
    after = np.nonzero(any_contact[fb:])[0]
    if not len(before):
        flags.append("腾空前没有检测到触地，无法确定离地时刻。")
        return res
    i_off = int(before[-1])
    i_land = int(fb + after[0]) if len(after) else None

    # 3) plant: each foot's contact interval that ends at take-off
    ivs = {}
    for name, m in (("left", cl), ("right", cr)):
        iv = _interval_ending_at(m, i_off)
        if iv:
            ivs[name] = iv
    if not ivs:
        flags.append("找不到起跳前的着地阶段。")
        return res
    # Final step = the foot that touched down last; the other foot's touchdown
    # (or its previous contact) is the penultimate step.
    final_foot = max(ivs, key=lambda k: ivs[k][0])
    i_td = ivs[final_foot][0]
    other = "left" if final_foot == "right" else "right"
    i_pen = None
    if other in ivs and ivs[other][0] < i_td:
        i_pen = ivs[other][0]
    else:
        m = cl if other == "left" else cr
        prev = [a for a, b in filters._runs(m) if a < i_td]
        i_pen = int(prev[-1]) if prev else None

    # 4) kinematics. Horizontal velocity of the hip midpoint (COM proxy),
    # averaged over a short window around each event to tame landmark noise.
    hip_mid = (s.xy[:, SIDES["left"]["hip"]] + s.xy[:, SIDES["right"]["hip"]]) / 2
    hip_mid = np.where(np.isfinite(hip_mid), hip_mid, s.joint("hip"))
    direction = 1.0 if np.nanmean(np.diff(hip_mid[: i_off + 1, 0])) >= 0 else -1.0
    vx = direction * filters.derivative(hip_mid[:, 0], t)
    w = max(1, int(round(0.05 * fps)))

    def vel_window(a, b):
        a, b = max(0, a), min(T - 1, b)
        seg = vx[a:b + 1]
        if b < a or not np.isfinite(seg).any():
            return None
        d = {"px_s": round(float(np.nanmean(seg)), 1)}
        if px_per_m:
            d["m_s"] = round(float(np.nanmean(seg) / px_per_m), 2)
        return d

    first_td = i_pen if i_pen is not None else i_td
    v_app = vel_window(first_td - int(round(0.3 * fps)), first_td - 1)
    v_td = vel_window(i_td - w, i_td + w)
    v_off = vel_window(i_off - w, i_off + w)

    def pct_loss(a, b):
        if a is None or b is None or a["px_s"] <= 0:
            return None
        return round(100.0 * (a["px_s"] - b["px_s"]) / a["px_s"], 1)

    # Lowest hip point between first plant touchdown and take-off: the end of
    # braking (eccentric) and start of the push (concentric). This is the
    # freeze frame; knee angles are reported per leg in contact.
    seg_y = hip_mid[first_td:i_off + 1, 1]
    i_low = first_td + int(np.nanargmax(seg_y)) if np.isfinite(seg_y).any() else i_td
    near_side = s.side
    knees = {sd: s.knee_angle_of(sd) for sd in ("left", "right")}
    contact = {"left": cl, "right": cr}

    def knee_info(i):
        out = {}
        for sd in ("left", "right"):
            a = knees[sd][i]
            out[sd] = {"deg": _r(a), "near_camera": sd == near_side, "in_contact": bool(contact[sd][i])}
        return out

    min_flex = {}
    for sd in ("left", "right"):
        m = contact[sd][first_td:i_off + 1] & np.isfinite(knees[sd][first_td:i_off + 1])
        if m.any():
            k = first_td + int(np.nanargmin(np.where(m, knees[sd][first_td:i_off + 1], np.nan)))
            min_flex[sd] = {"deg": _r(knees[sd][k]), "t_s": round(float(t[k]), 3),
                            "near_camera": sd == near_side}

    final_contact_s = (t[i_off] - t[i_td]) + dt   # inclusive frame count
    plant_contact_s = (t[i_off] - t[first_td]) + dt
    ev = res["events"]
    ev.update({
        "penultimate_touchdown": None if i_pen is None else
        {"frame": i_pen, "t_s": round(float(t[i_pen]), 3), "foot": other},
        "final_touchdown": {"frame": i_td, "t_s": round(float(t[i_td]), 3), "foot": final_foot},
        "lowest_hip": {"frame": i_low, "t_s": round(float(t[i_low]), 3)},
        "takeoff": {"frame": i_off, "t_s": round(float(t[i_off]), 3)},
        "landing": None if i_land is None else {"frame": i_land, "t_s": round(float(t[i_land]), 3)},
    })
    summ = res["summary"]
    summ.update({
        "final_step_contact_s": round(float(final_contact_s), 3),
        "plant_contact_s": round(float(plant_contact_s), 3),
        "time_resolution_s": round(dt, 3),
        "knee_at_final_touchdown": knee_info(i_td),
        "knee_at_lowest_hip": knee_info(i_low),
        "knee_at_takeoff": knee_info(i_off),
        "min_knee_in_contact": min_flex,
        "approach_velocity": v_app,
        "velocity_at_final_touchdown": v_td,
        "velocity_at_takeoff": v_off,
        "velocity_change_penultimate_pct": pct_loss(v_app, v_td),
        "velocity_change_plant_pct": pct_loss(v_td, v_off),
        "scale": scale_source,
    })
    if i_land is not None and i_off + 1 < T:
        # take-off / landing instants lie between the last-contact and first-air frames
        t_up = (t[i_off] + t[i_off + 1]) / 2
        t_down = (t[i_land - 1] + t[i_land]) / 2
        flight = float(t_down - t_up)
        summ["flight_time_s"] = round(flight, 3)
        summ["jump_height_from_flight_cm"] = round(100 * G * flight ** 2 / 8, 1)
        lo, hi = max(flight - dt, 0), flight + dt
        summ["jump_height_range_cm"] = [round(100 * G * lo ** 2 / 8, 1), round(100 * G * hi ** 2 / 8, 1)]
    res["_series"] = {"vx": vx, "hip_mid": hip_mid, "contact_l": cl, "contact_r": cr}
    res["_idx"] = {"pen": i_pen, "first": first_td, "td": i_td, "low": i_low, "off": i_off, "land": i_land}

    caveats += [
        f"所有事件时刻都量化到帧：{fps:g} fps 下每个时刻 ±{1000 * dt:.0f} ms，"
        f"最后一步触地时间 {final_contact_s:.3f} s 的误差约 ±{1000 * dt:.0f} ms。",
        "触地判定基于“脚的最低点贴近局部地面且几乎不动”，要求手机固定不动；手持跟拍会失效。"
        "拖地滑步的脚要等停稳才算着地，着地时刻可能偏晚。",
        "水平速度用双髋中点近似身体重心、在事件前后约 ±0.05 s 内取平均；运动员朝向/远离镜头移动时，"
        "近大远小会让像素速度失真，速度变化百分比也受影响。",
        "远离镜头那条腿的关节点常被身体遮挡，其膝角可信度较低（报告中已标注近侧/远侧）。",
        "腾空高度用 h = g·t²/8 由腾空时间推算，假设起跳与落地时身体姿态相同；落地时屈膝更早会让结果偏大。",
    ]
    if scale_source == "height":
        caveats.append("米制速度按“髋→踝长度 ≈ 身高 × 0.491”换算：该比例取自 Drillis & Contini 人体比例（经 Winter 教材转引）"
                       "的常用近似值，【未经核实原表】；且个体差异会带来额外误差（未量化）。")
    if fps < 100:
        caveats.append("触地时间这类 0.1–0.5 s 的事件建议用 120/240 fps 慢动作拍摄。")
    for key, what in (("velocity_change_penultimate_pct", "助跑 → 最后一步着地"),
                      ("velocity_change_plant_pct", "最后一步着地 → 离地")):
        v = summ[key]
        if v is None:
            continue
        flags.append(f"{what}：水平速度{'下降' if v > 0 else '增加'} {abs(v)}%。")
    return res


def _r(v, nd=1):
    v = float(v)
    return round(v, nd) if np.isfinite(v) else None


def _side_cn(sd):
    return "左" if sd == "left" else "右"


def _knees(d):
    parts = []
    for sd in ("left", "right"):
        k = d[sd]
        if k["deg"] is None:
            continue
        parts.append(f"{_side_cn(sd)}膝 {k['deg']}°{'（近侧）' if k['near_camera'] else '（远侧，可信度低）'}"
                     f"{'' if k['in_contact'] else '·未触地'}")
    return "，".join(parts) or "—"


def sections(res: dict) -> list[dict]:
    if not res["events"]:
        return []
    s, e = res["summary"], res["events"]

    def v(d):
        if d is None:
            return "—"
        return f"{d['m_s']} m/s" if "m_s" in d else f"{d['px_s']} px/s"

    def pct(x):
        return "—" if x is None else (f"下降 {x}%" if x > 0 else f"增加 {abs(x)}%")
    pen = e["penultimate_touchdown"]
    lines = [
        f"- 倒数第二步（{_side_cn(pen['foot']) + '脚' if pen else '—'}）着地 {pen['t_s'] if pen else '—'} s → "
        f"最后一步（{_side_cn(e['final_touchdown']['foot'])}脚）着地 {e['final_touchdown']['t_s']} s → "
        f"髋部最低点 {e['lowest_hip']['t_s']} s → 离地 {e['takeoff']['t_s']} s → "
        f"落地 {e['landing']['t_s'] if e['landing'] else '—'} s",
        f"- 触地时间：最后一步 {s['final_step_contact_s']} s；整个双脚制动-起跳阶段 {s['plant_contact_s']} s"
        f"（各 ±{s['time_resolution_s']} s）",
        f"- 最后一步着地时膝角：{_knees(s['knee_at_final_touchdown'])}",
        f"- 髋部最低点（制动结束）膝角：{_knees(s['knee_at_lowest_hip'])}",
        f"- 离地时膝角：{_knees(s['knee_at_takeoff'])}（180°=伸直）",
    ]
    mf = s["min_knee_in_contact"]
    if mf:
        lines.append("- 触地期间最大屈膝：" + "，".join(
            f"{_side_cn(sd)}膝 {d['deg']}° @ {d['t_s']} s{'（近侧）' if d['near_camera'] else '（远侧）'}"
            for sd, d in mf.items()))
    lines += [
        f"- 水平速度：助跑（倒数第二步着地前 0.3 s 平均）{v(s['approach_velocity'])} → 最后一步着地 "
        f"{v(s['velocity_at_final_touchdown'])} → 离地 {v(s['velocity_at_takeoff'])}",
        f"- 速度变化：倒数第二步 {pct(s['velocity_change_penultimate_pct'])}；"
        f"最后一步到离地 {pct(s['velocity_change_plant_pct'])}",
    ]
    if "flight_time_s" in s:
        lo, hi = s["jump_height_range_cm"]
        lines.append(f"- 腾空时间 {s['flight_time_s']} s → 推算重心腾起高度 {s['jump_height_from_flight_cm']} cm"
                     f"（±1 帧对应 {lo}–{hi} cm）")
    return [{"title": "起跳事件与指标", "lines": lines}]


def make_drawer(s: Session, res: dict):
    near = set(SIDES[s.side].values())
    if not res["events"]:
        def draw_plain(frame, i):
            if s.raw.detected[i]:
                R.draw_skeleton(frame, s.xy[i], s.vis[i], near)
        return draw_plain, []
    ix, ser, summ = res["_idx"], res["_series"], res["summary"]
    hip_mid, cl, cr = ser["hip_mid"], ser["contact_l"], ser["contact_r"]
    knee_a = s.knee_angle()
    toe_l, toe_r = s.joint_of("left", "toe"), s.joint_of("right", "toe")
    t = s.t

    def phase(i):
        if i < ix["first"]:
            return "APPROACH", R.WHITE
        if i <= ix["off"]:
            return "PLANT", R.ORANGE
        if ix["land"] is None or i < ix["land"]:
            return "FLIGHT", R.CYAN
        return "LANDING", R.WHITE

    def draw(frame, i):
        # COM-proxy trail: approach cyan, plant orange, flight magenta
        segs = [(hip_mid[: min(i, ix["first"]) + 1], R.CYAN)]
        if i >= ix["first"]:
            segs.append((hip_mid[ix["first"]: min(i, ix["off"]) + 1], R.ORANGE))
        if i > ix["off"]:
            segs.append((hip_mid[ix["off"]: i + 1], R.MAGENTA))
        R.glow_polylines(frame, segs)
        if s.raw.detected[i]:
            R.draw_skeleton(frame, s.xy[i], s.vis[i], near)
        for m, p in ((cl, toe_l), (cr, toe_r)):  # ground-contact markers
            if m[i] and np.all(np.isfinite(p[i])):
                c = tuple(int(v) for v in p[i])
                cv2.circle(frame, c, int(10 * R.ui_scale(frame)), R.LIME, -1, cv2.LINE_AA)
        name, col = phase(i)
        lines = [(name, col), (f"t {t[i]:.2f}s", R.WHITE),
                 (f"knee {knee_a[i]:.0f} deg" if np.isfinite(knee_a[i]) else "knee --", R.YELLOW)]
        if ix["first"] <= i <= ix["off"]:
            lines.append((f"plant {t[i] - t[ix['first']] + 1 / s.info.effective_fps:.2f}s", R.ORANGE))
        R.hud(frame, lines, title="APPROACH JUMP")

    def knee_arcs(fr, i, color):
        for sd in ("left", "right"):
            a = s.knee_angle_of(sd)[i]
            if not np.isfinite(a) or (sd != s.side and not np.isfinite(s.xy[i, SIDES[sd]["ankle"], 0])):
                continue
            j = SIDES[sd]
            R.angle_arc(fr, s.xy[i, j["hip"]], s.xy[i, j["knee"]], s.xy[i, j["ankle"]],
                        color if sd == s.side else R.GREY, f"{a:.0f}")

    res_ms = 1000 / s.info.effective_fps

    def near_knee(d):
        k = d[s.side]["deg"]
        return "--" if k is None else f"{k:.0f}"

    def brake_note(fr):
        knee_arcs(fr, ix["low"], R.YELLOW)
        R.banner(fr, "BRAKE END  lowest hip", R.ORANGE)
        lines = [(f"near knee {near_knee(summ['knee_at_lowest_hip'])} deg", R.YELLOW),
                 (f"last-step contact {summ['final_step_contact_s']:.2f}s +-{res_ms:.0f}ms", R.ORANGE),
                 (f"plant contact {summ['plant_contact_s']:.2f}s", R.ORANGE)]
        vp = summ["velocity_change_penultimate_pct"]
        if vp is not None:
            lines.append((f"v penultimate {'-' if vp > 0 else '+'}{abs(vp):.0f}%", R.CYAN))
        R.hud(fr, lines, corner="bl")

    def td_note(fr):
        knee_arcs(fr, ix["td"], R.ORANGE)
        R.banner(fr, "FINAL STEP TOUCHDOWN", R.ORANGE)

    def off_note(fr):
        knee_arcs(fr, ix["off"], R.CYAN)
        R.banner(fr, "TAKE-OFF", R.CYAN)
        lines = [(f"near knee {near_knee(summ['knee_at_takeoff'])} deg", R.YELLOW)]
        vp = summ["velocity_change_plant_pct"]
        if vp is not None:
            lines.append((f"v plant {'-' if vp > 0 else '+'}{abs(vp):.0f}%", R.CYAN))
        if "jump_height_from_flight_cm" in summ:
            lines.append((f"flight {summ['flight_time_s']:.2f}s ~{summ['jump_height_from_flight_cm']:.0f}cm",
                          R.MAGENTA))
        R.hud(fr, lines, corner="bl")

    freezes = [Freeze(ix["td"], 0.8, td_note), Freeze(ix["low"], 1.6, brake_note),
               Freeze(ix["off"], 1.2, off_note)]
    return draw, freezes


def scale_from_height(s: Session, height_cm: float | None) -> float | None:
    """px per metre from body height via the hip-to-ankle segment proportion."""
    if not height_cm:
        return None
    return s.leg_length() / (LEG_FRACTION_OF_HEIGHT * height_cm / 100.0)
