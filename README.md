# formtrace

把手机拍的训练视频变成看得见的数据：骨架、关节角、霓虹轨迹线。

> 当前版本：v0.1.0（通用骨架 + 关节角）。深蹲杠铃轨迹、助跑起跳制动分析在后续版本加入，见 [CHANGELOG](CHANGELOG.md)。

## 安装

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
# Linux 服务器/容器还需要 EGL：sudo apt-get install libegl1 libgles2
# 推荐安装 ffmpeg，输出的视频才能在浏览器/手机里直接播放（H.264）
```

## 使用

```bash
formtrace analyze my_lift.mov            # 输出到 ./out/my_lift/
formtrace analyze my_lift.mov --csv      # 额外导出逐帧数据
formtrace analyze my_lift.mov --model heavy   # 更准但更慢
```

输出：`overlay.mp4`（叠加视频）、`report.md`（中文报告）、`report.json`、`series.csv`。

## 拍摄要求（直接决定数据准不准）

1. **正侧面、手机固定不动**（靠在水壶/背包上即可），镜头与运动平面垂直。斜拍时 2D 角度会失真。
2. 全身入镜，含脚。
3. 起跳/触地类分析尽量用 **慢动作 120/240 fps**；30 fps 时任何时间点都有 ±33 ms 误差。

## 已知限制

- 所有角度都是画面上的 2D 投影角，不是三维真实关节角。
- MediaPipe 只识别人体，**不识别杠铃**（杠铃追踪见后续版本）。
- 处理速度取决于 CPU：在 4 核云主机上，`full` 模型约 80 ms/帧（3 秒 30 fps 视频姿态提取约 7 s）。

## License

MIT
