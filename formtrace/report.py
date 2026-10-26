"""Human-readable (Chinese) Markdown report built from the JSON result."""

from __future__ import annotations


def _fmt(v, unit=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:g}{unit}"
    return f"{v}{unit}"


def quality_notes(meta: dict) -> list[str]:
    """Warnings about input quality that directly limit the accuracy of the numbers."""
    notes = []
    fps = meta["effective_fps"]
    notes.append(f"时间分辨率：{fps:g} fps → 任何时间点的误差约 ±{1000 / fps:.0f} ms（±1 帧）。")
    if fps < 59:
        notes.append("帧率低于 60 fps：触地时间、制动瞬间这类短事件误差较大，建议用手机慢动作（120/240 fps）拍摄。")
    if meta["timestamps_source"] != "pts":
        notes.append("视频没有可用的逐帧时间戳，按标称帧率估算时间；可变帧率视频会因此产生时间误差。")
    if meta["max_frame_gap_s"] > 1.6 / fps:
        notes.append(f"检测到丢帧（最大帧间隔 {meta['max_frame_gap_s'] * 1000:.0f} ms），已按真实时间戳计算。")
    if meta["pose_detection_rate"] < 0.9:
        notes.append(f"只有 {meta['pose_detection_rate'] * 100:.0f}% 的帧检测到人体，缺失帧由插值补齐，结果可信度下降。")
    notes.append("所有角度均为 2D 投影角：只有手机正对身体侧面、与运动平面垂直时才接近真实关节角。")
    if meta.get("leg_length_source") not in (None, "measured"):
        notes.append("脚始终不在画面内：“腿长”按 1.70 × 躯干长度估算（人体比例近似值，未经核实），"
                     "以“% 腿长”表示的距离只能作相对比较。")
    notes.append("本工具尚未与测力台、三维动捕等金标准做过对比验证，数值适合看趋势和自我对比，不宜当作精确测量。")
    return notes


def markdown(res: dict) -> str:
    m = res["meta"]
    lines = [f"# formtrace 分析报告（v{res['version']}，模式：{res['mode']}）", ""]
    lines += [f"- 视频：`{m['video']}`  {m['resolution'][0]}×{m['resolution'][1]}，"
              f"{m['frames']} 帧，{m['duration_s']} s，有效帧率 {m['effective_fps']} fps"
              f"（标称 {m['nominal_fps']}）",
              f"- 面向镜头的一侧：{'左' if m['camera_side'] == 'left' else '右'}侧（单侧指标均取此侧）",
              f"- 处理耗时：姿态提取 {m['timings'].get('extract_s', '—')} s，"
              f"渲染 {m['timings'].get('render_s', '—')} s", ""]
    lines += ["## 结果", ""]
    for k, v in res["summary"].items():
        if isinstance(v, (list, dict)):
            continue
        lines.append(f"- {LABELS.get(k, k)}：{_fmt(v)}")
    for section in res.get("sections", []):
        lines += ["", f"### {section['title']}", ""] + section["lines"]
    if res.get("flags"):
        lines += ["", "## 提示（启发式判断，非诊断）", ""] + [f"- {f}" for f in res["flags"]]
    lines += ["", "## 数据可信度", ""] + [f"- {n}" for n in quality_notes(m) + res.get("caveats", [])]
    if res.get("files"):
        lines += ["", "## 输出文件", ""] + [f"- {k}: `{v}`" for k, v in res["files"].items()]
    return "\n".join(lines) + "\n"


LABELS = {
    "min_knee_angle_deg": "最小膝关节夹角（180°=伸直）",
    "min_knee_angle_t_s": "出现时间 (s)",
    "min_hip_angle_deg": "最小髋关节夹角",
    # squat
    "reps": "识别到的次数",
    "rep_signal": "分次依据（bar=杠铃轨迹，hip=髋部轨迹）",
    "trunk_upper_point": "躯干上端参考点",
    "bar_source": "杠铃轨迹来源",
    "bar_tracking_coverage": "杠铃追踪覆盖率",
    "calibration": "比例尺标定方式",
    "px_per_cm": "每厘米像素数",
    "deepest_knee_angle_deg": "全组最小膝关节夹角（°）",
    # jump
    "final_step_contact_s": "最后一步触地时间 (s)",
    "plant_contact_s": "双脚制动-起跳阶段时长 (s)",
    "time_resolution_s": "时间分辨率 (s/帧)",
    "velocity_change_penultimate_pct": "倒数第二步水平速度变化（正=下降）%",
    "velocity_change_plant_pct": "最后一步→离地水平速度变化（正=下降）%",
    "scale": "米制换算依据",
    "flight_time_s": "腾空时间 (s)",
    "jump_height_from_flight_cm": "腾空时间推算的重心腾起高度 (cm)",
}
