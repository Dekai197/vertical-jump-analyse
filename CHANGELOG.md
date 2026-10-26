# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循语义化版本。

## [0.1.0] - 2026-10-26
### 新增
- 命令行 `formtrace analyze VIDEO`：一次解码完成 MediaPipe Pose Landmarker（Tasks API，VIDEO 模式）姿态提取。
- 逐帧真实时间戳：优先使用视频 PTS，避免手机可变帧率视频按标称帧率计时造成的误差
  （测试样片标称 26.43 fps，实际 30 fps 匀速，误差约 13%）。
- 缺帧线性插值 + Savitzky–Golay 平滑（纯 numpy 实现，无 SciPy 依赖）。
- 自动判断面向镜头的身体一侧，单侧指标只用近侧关节。
- 叠加视频：骨架、髋/踝霓虹轨迹（最近 1.5 s）、实时膝角/髋角 HUD；有 ffmpeg 时自动转 H.264。
- 输出 `report.json`、中文 `report.md`（含数据可信度说明），可选逐帧 `series.csv`。
- 首次运行自动下载官方 pose_landmarker 模型到 `~/.cache/formtrace`。
