# formtrace

把手机拍的训练视频变成看得见的数据：骨架、关节角、霓虹轨迹线。

> 当前版本：v0.3.0（通用骨架 + 深蹲/杠铃轨迹 + 助跑起跳）。版本记录见 [CHANGELOG](CHANGELOG.md)。

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

### 深蹲 + 杠铃轨迹

```bash
formtrace analyze squat.mov --mode squat                    # 自动找杠铃片
formtrace analyze squat.mov --mode squat --plate-cm 45      # 按杠铃片直径换算成厘米、m/s
formtrace frame squat.mov --at 0 -o grid.jpg                # 自动找错时：看网格读坐标
formtrace analyze squat.mov --mode squat --bar 900,1100     # 手动指定第 0 帧杠铃片中心
```

报告内容：次数、每次下蹲/起身用时、最小膝角、近似是否过平行、起身前半程“上身/髋上升比”
（1 = 同步，明显 < 1 提示可能先抬臀）、杠铃水平摆动、最低点杠铃相对脚掌中点的前后偏移、（标定后）向心速度。

### 助跑起跳

```bash
formtrace analyze jump.mov --mode jump                   # 像素速度 + 百分比
formtrace analyze jump.mov --mode jump --height-cm 180   # 近似换算成 m/s
```

视频会在 **最后一步着地、制动结束（髋部最低点）、离地** 三处定格，标出膝角、触地时间、水平速度变化；
报告给出倒数第二步/最后一步的着地时刻、触地时间、助跑→着地→离地的水平速度及其变化、腾空时间与推算高度。

## 拍摄要求（直接决定数据准不准）

1. **正侧面、手机固定不动**（靠在水壶/背包上即可），镜头与运动平面垂直。斜拍时 2D 角度会失真。
2. 全身入镜，含脚。
3. 起跳/触地类分析尽量用 **慢动作 120/240 fps**；30 fps 时任何时间点都有 ±33 ms 误差。

## 已知限制

- 所有角度都是画面上的 2D 投影角，不是三维真实关节角。
- MediaPipe 只识别人体，**不识别杠铃**：杠铃片靠圆检测 + 视觉跟踪，只能追踪画面近侧那一端，远端被身体挡住。
- 侧面拍背蹲时杠铃片会挡住头，而 MediaPipe 的人体检测依赖人脸，姿态可能间歇丢失；此时分次改用杠铃轨迹，身体指标会出现空值。
- “过平行”“先抬臀”都是近似/启发式判断，阈值未经实验验证，不能替代教练或裁判。
- 尚未与测力台、三维动捕等金标准做过对比验证。
- 处理速度取决于 CPU：在 4 核云主机上，`full` 模型约 80 ms/帧（3 秒 30 fps 视频姿态提取约 7 s）。

## License

MIT
