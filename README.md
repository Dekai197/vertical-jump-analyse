# formtrace

把手机拍的训练视频变成看得见的数据：骨架、关节角、霓虹轨迹线。

> 当前版本：v0.4.0（通用骨架 + 深蹲/杠铃轨迹 + 助跑起跳 + 本地网页界面）。版本记录见 [CHANGELOG](CHANGELOG.md)。

## 安装

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
# Linux 服务器/容器还需要 EGL：sudo apt-get install libegl1 libgles2
# 推荐安装 ffmpeg，输出的视频才能在浏览器/手机里直接播放（H.264）
```

## 使用

### 网页界面（推荐）

```bash
formtrace serve          # 自动打开 http://127.0.0.1:8765/ ，把视频拖进去即可
```

### 命令行

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
- 手机离人太近或机位太低时透视变形明显：样片中杠铃垂直行程算出约 155%“腿长”，明显偏大，这类比例值只适合同机位前后对比。

## 引用与核实状态

| 内容 | 用在哪里 | 状态 |
|---|---|---|
| MediaPipe 人体检测器以 BlazeFace 人脸检测为“人体检测的代理”（[官方文档](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md)） | 解释杠铃片挡住头部时姿态丢失；`num_poses=1` 的选择 | 已核实 |
| 腾空高度 h = g·t²/8 | 起跳 | 由匀加速运动公式推导（上升时间 = t/2），假设起跳/落地姿态相同 |
| 髋→踝 ≈ 0.491 × 身高、肩→髋 ≈ 0.288 × 身高（Drillis & Contini，经 Winter 教材转引） | `--height-cm` 换算；脚不入镜时的腿长估算 | **未核实**（原表无法访问） |
| 举重杠铃片直径 450 mm | `--plate-cm` 的说明文字 | **未核实**（只有厂商页面转述） |
| 跟踪置信度 0.5→0.3 使样片主体检出率 47%→88%；`num_poses=2` 使检出降到约 30% | 默认参数 | 本项目在一段样片上实测，未在更多视频上验证 |
| “先抬臀”阈值（上升比 < 0.7 且前倾增加 > 5°）、触地速度阈值等 | 提示文字 | 本工具的启发式设定，**无文献依据** |
- **不是“一秒出结果”**：处理速度取决于 CPU。在 4 核云主机（无 GPU）上实测 `full` 模型约 50–80 ms/帧：
  3 秒 30 fps 起跳视频总计约 15–30 s；22 秒 60 fps 深蹲视频姿态+杠铃约 70–100 s，渲染约 90 s。
  `--model lite` 更快，改参数重跑会命中缓存（< 1 s）。

## License

MIT
