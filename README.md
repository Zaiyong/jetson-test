# YOLO11 人体检测速度对比（Laptop / Jetson / Gaming PC）

用 USB 摄像头循环「抓帧 → 检测 → 抓帧 …」，记录每一帧各步骤耗时，比较三台设备在各自最优后端下的性能。

| 设备 | 后端 | 精度 | device |
|---|---|---|---|
| `laptop`（Lenovo L14，CPU） | OpenVINO | FP32 | `cpu` |
| `jetson`（Orin Nano 8GB） | TensorRT engine | FP16 | `0` |
| `gamingpc`（RTX 3060 Ti） | TensorRT engine | FP16 | `0` |

默认模型：`yolo11x`，输入 640，只检测 person（COCO class 0）；默认摄像头分辨率 1280×720@30。三台设备统一用默认值对比。

## 文件

| 文件 | 作用 |
|---|---|
| `benchmark.py` | 下载模型 → 导出（OpenVINO / TensorRT）→ 打开摄像头 → 循环检测 → 记录每帧耗时 |
| `analyze.py` | 读取 `results/` 下所有记录，输出汇总表和图表到 `report/` |
| `pyproject.toml` | uv 依赖管理（`laptop` / `gamingpc` 两个 extra） |

## 1. 安装 uv

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Linux / Jetson
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. 环境配置

### Laptop（Windows，CPU + OpenVINO）

```powershell
uv sync --extra laptop
```

- 插电源，Windows 电源模式选「最佳性能」，关闭其他占 CPU 的程序。

### Gaming PC（Windows/Linux x86_64，3060 Ti + TensorRT）

```powershell
uv sync --extra gamingpc
uv run python -c "import torch, tensorrt; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), tensorrt.__version__)"
```

- 需要支持 CUDA 12.6 的 NVIDIA 驱动。
- 第一次导出 TensorRT engine 需要几分钟，之后使用缓存。

### Jetson Orin Nano（JetPack 6，TensorRT）

Jetson 不使用 `uv sync`：TensorRT 必须用 JetPack 自带版本，torch 必须用 NVIDIA 为 Jetson 编译的 wheel。

```bash
# 性能模式（用 sudo nvpmodel -q 查看当前模式，选最大功耗 / MAXN）
sudo nvpmodel -m 0
sudo jetson_clocks

# 虚拟环境：继承系统 site-packages 以使用 JetPack 的 tensorrt
uv venv .venv-jetson --python /usr/bin/python3 --system-site-packages
source .venv-jetson/bin/activate

uv pip install ultralytics "opencv-python<5" pandas matplotlib onnx onnxslim
uv pip uninstall torch torchvision
# 按 JetPack 版本安装对应的 torch / torchvision wheel，见：
# https://docs.ultralytics.com/guides/nvidia-jetson/
uv pip install <torch wheel> <torchvision wheel>

# 检查
python -c "import torch, tensorrt; print(torch.cuda.is_available(), tensorrt.__version__)"
```

- 推荐无桌面（headless）运行，并用 `jtop` 观察温度和功耗。

## 3. 运行测试

Laptop / Gaming PC 用 `uv run benchmark.py ...`；Jetson 先 `source .venv-jetson/bin/activate`，再用 `python benchmark.py ...`。

### 0. 找到 USB 摄像头编号 + 设置分辨率

测试用两个同型号 USB 摄像头（IMX335，最高 2592×1944@30fps，MJPG），不用笔记本自带摄像头。编号顺序不固定（和插入顺序、USB 口有关），先探测：

```powershell
# 按 --width/--height/--cam-fps 打开每个编号，打印摄像头实际给出的分辨率
uv run benchmark.py --list-cameras
```

```bash
python benchmark.py --list-cameras
v4l2-ctl --list-devices          # Linux：每个 USB 摄像头通常占 2 个节点（/dev/video0、/dev/video1），用偶数那个
```

输出类似（打不开的编号会打印 OpenCV 警告，可忽略）：

```
  index 0: 1280x720 @30fps         <- 笔记本自带，不用
  index 1: 1280x720 @30fps
  index 2: 1280x720 @30fps
```

然后用 `--cameras` 明确指定 USB 摄像头（不指定时默认 `0..n-1`，会用到自带摄像头）。两个同型号摄像头名字相同，可以拔掉其中一个再 `--list-cameras`，消失的那个编号就是它。

**Windows 摄像头后端**：默认 `msmf`。实测 IMX335 摄像头在 `dshow` 下最高 20fps，`msmf` 下 720p/1080p/2592×1944 都有 30fps。两种后端的编号顺序可能不同，`--list-cameras` 和正式测试要用同一个 `--cam-api`。

**分辨率**：用 `--width --height --cam-fps` 请求，默认 `1280x720@30`（对比时保持默认）。

- 摄像头只支持固定的几种模式，请求不支持的分辨率时驱动会选最接近的；实际值会打印出来，不一致时输出 `WARNING`，并记录在 CSV 的 `cam_width/cam_height/cam_fps` 和 meta.json 里。
- 查看摄像头支持的模式：Linux `v4l2-ctl -d /dev/video0 --list-formats-ext`；Windows `ffmpeg -list_options true -f dshow -i video="摄像头名"`。
- 模型输入始终缩放到 `--imgsz`（640），所以分辨率越高，推理时间基本不变，但抓帧/MJPG 解码/预处理（`grab_ms`、`preprocess_ms`）变慢；2592×1944 每帧是 720p 的 5.5 倍像素，CPU 弱的设备（Jetson、笔记本）受影响明显。
- 三台设备对比时用**同样的摄像头、同样的分辨率**。两个摄像头直接插在电脑的不同 USB 口上，不要共用一个 hub，否则 USB 2.0 带宽不足会打不开或掉帧。

### A. 单路测试

```powershell
uv run benchmark.py --computer laptop --n_camera 1 --cameras 1
uv run benchmark.py --computer gamingpc --n_camera 1 --cameras 0
```

```bash
python benchmark.py --computer jetson --n_camera 1 --cameras 0
```

### B. 两路并发测试

```powershell
uv run benchmark.py --computer laptop --n_camera 2 --cameras 1,2
uv run benchmark.py --computer gamingpc --n_camera 2 --cameras 0,1
```

```bash
python benchmark.py --computer jetson --n_camera 2 --cameras 0,2   # Linux 上编号见 v4l2-ctl --list-devices
```

### 其他例子

```powershell
# 换分辨率
uv run benchmark.py --computer laptop --n_camera 2 --cameras 1,2 --width 1920 --height 1080

# 换模型
uv run benchmark.py --computer laptop --n_camera 1 --model yolo11m

# INT8（需要校准数据，默认 coco8.yaml，会自动下载）
uv run benchmark.py --computer gamingpc --n_camera 1 --precision int8

# 用 PyTorch .pt 作对照
uv run benchmark.py --computer gamingpc --n_camera 1 --format pt

# 用视频文件代替摄像头（不受摄像头帧率限制）
uv run benchmark.py --computer laptop --n_camera 1 --cameras D:\videos\people.mp4

# Windows 上 msmf 打不开某个摄像头时（编号先用 --list-cameras --cam-api dshow 确认）
uv run benchmark.py --computer laptop --n_camera 1 --cameras 1 --cam-api dshow
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--computer` / `-computer` | 必填 | `laptop` / `jetson` / `gamingpc` |
| `--list-cameras` | 关 | 探测编号 0..`--max-index`（默认 10）的摄像头并打印实际分辨率，然后退出 |
| `--n_camera` / `-n_camera` | `1` | 同时运行的摄像头数量 |
| `--cameras` | `0..n-1` | 摄像头编号或视频文件，逗号分隔；笔记本上要跳过自带摄像头，如 `1,2,3` |
| `--model` | `yolo11x` | `yolo11n` / `yolo11s` / `yolo11m` / `yolo11l` / `yolo11x` |
| `--format` | 按 computer | `pt` / `openvino` / `engine` |
| `--precision` | 按 computer | `fp32` / `fp16` / `int8` |
| `--device` | 按 computer | `cpu` / `0` / `intel:gpu` … |
| `--imgsz` | `640` | 模型输入尺寸 |
| `--conf` | `0.25` | 置信度阈值 |
| `--duration` | `60` | 正式测量秒数 |
| `--warmup` | `30` | 每路预热帧数（不计入统计） |
| `--width` / `--height` / `--cam-fps` | `1280` / `720` / `30` | 请求的摄像头分辨率和帧率 |
| `--cam-api` | `auto` | Windows=`msmf`，Linux=`v4l2`；也可 `dshow` / `any` |
| `--no-mjpg` | 关 | 不请求 MJPG（多路 USB 摄像头建议保留 MJPG） |
| `--force-export` | 关 | 忽略缓存，重新导出模型 |
| `--tag` | 空 | 备注，写入 meta.json |

## 4. 输出

```
models/
  yolo11x.pt                                  # 官方权重（自动下载）
  laptop/yolo11x_640_fp32_openvino_model/     # 导出缓存，按设备分开
  gamingpc/yolo11x_640_fp16.engine
results/
  <run_id>_frames.csv                         # 每帧一行
  <run_id>_meta.json                          # 参数、模型下载/导出时间、软硬件版本、摄像头实际分辨率
```

`_frames.csv` 每帧记录：

| 列 | 含义 |
|---|---|
| `grab_ms` | `cap.read()` 抓帧（含等待摄像头新帧） |
| `preprocess_ms` | 缩放 / letterbox / 归一化 |
| `inference_ms` | 模型推理 |
| `postprocess_ms` | NMS 等后处理 |
| `predict_ms` | 整个 `model.predict()` 调用 |
| `loop_ms` | 一次完整循环（抓帧 + predict） |
| `n_person` | 检测到的人数 |

TensorRT engine 与 GPU / TensorRT 版本绑定，不要在设备之间拷贝 `models/<computer>/`，每台设备各自导出。

## 5. 分析

把三台设备的 `results/` 文件拷到同一台电脑的 `results/` 下，然后：

```powershell
uv run analyze.py                                # 所有记录
uv run analyze.py "results/jetson_*_frames.csv"  # 只分析部分
uv run analyze.py --no-plot
```

输出到 `report/`：

| 文件 | 内容 |
|---|---|
| `runs.csv` | 每次测试一行：总 FPS、每路 FPS、各步骤 mean / p50 / p95、稳定性、扩展效率 |
| `cameras.csv` | 每路摄像头一行 |
| `fps.png` | 总 FPS / 每路 FPS / 纯推理 FPS |
| `steps.png` | 各步骤平均耗时（堆叠柱状图） |
| `inference_box.png` | 推理耗时分布 |
| `fps_timeline.png` | 每秒 FPS 随时间变化 |

主要指标：

| 指标 | 含义 |
|---|---|
| `total_fps` | 所有摄像头 FPS 之和（系统吞吐） |
| `fps_per_cam` | 每路平均 FPS |
| `infer_only_fps` | `1000 / inference_ms`，不受摄像头限制的纯算力 |
| `loop_cv`、`fps_1s_std` | 抖动 / 稳定性，越小越稳定 |
| `camera_bound` | `True` 表示 FPS 被摄像头帧率卡住，此时比较 `inference_ms` |
| `scaling_eff` | `n 路总 FPS / (n × 单路 FPS)`，并发扩展效率 |

## 6. 注意事项

- 公平比较：三台设备使用相同的 `--model`、`--imgsz`、摄像头分辨率、`--duration`，对着相同场景。
- 摄像头一般最高 30 fps。处理比摄像头快时 `grab_ms` 变大（在等新帧），FPS 被卡在 30；处理比摄像头慢时 `grab_ms` 接近 0，多余的帧被驱动丢弃。
- 多路 USB 摄像头尽量接在不同 USB 控制器上，并使用 MJPG，否则带宽不足会掉帧或打不开。
- 第一次运行包括权重下载和模型导出，不计入测量；之后使用 `models/` 中的缓存。
