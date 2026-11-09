"""YOLO person-detection camera benchmark (single camera / multi-camera concurrent).

Pipeline per camera thread:  grab frame -> preprocess -> inference -> postprocess -> next frame
Every frame is logged with the time of each step, so `analyze.py` can compute averages later.

Usage examples
  uv run benchmark.py --list-cameras                                   (find USB camera indices)
  uv run benchmark.py --computer laptop   --n_camera 1 --cameras 2
  uv run benchmark.py --computer laptop   --n_camera 2 --cameras 2,3 --duration 60
  uv run benchmark.py --computer gamingpc --n_camera 3
  python benchmark.py --computer jetson   --n_camera 3 --cameras 0,2,4      (inside the Jetson venv)

Default backend per computer (override with --format / --precision / --device)
  laptop   -> OpenVINO  FP32  on CPU
  jetson   -> TensorRT  FP16  on GPU 0
  gamingpc -> TensorRT  FP16  on GPU 0

The .pt weights are downloaded once to models/, exported models are cached in
models/<computer>/ (TensorRT engines are hardware specific, never copy them between machines).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import socket
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    # Without this, MSMF needs several seconds to open each camera.
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2

PERSON_CLASS = 0  # COCO "person"

PROFILES = {
    "laptop": {"format": "openvino", "precision": "fp32", "device": "cpu"},
    "jetson": {"format": "engine", "precision": "fp16", "device": "0"},
    "gamingpc": {"format": "engine", "precision": "fp16", "device": "0"},
}
EXPORT_SUFFIX = {"openvino": "_openvino_model", "engine": ".engine"}
CAM_APIS = {
    "any": cv2.CAP_ANY,
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "v4l2": cv2.CAP_V4L2,
}
CSV_FIELDS = [
    "run_id", "computer", "n_camera", "model", "format", "precision", "imgsz", "device",
    "cam", "source", "cam_width", "cam_height", "cam_fps",
    "phase", "frame", "t_start_s", "t_end_s",
    "grab_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "predict_ms", "loop_ms",
    "n_person",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--computer", "-computer", choices=list(PROFILES))
    p.add_argument("--list-cameras", action="store_true",
                   help="probe camera indices 0..--max-index with --width/--height/--cam-fps and exit")
    p.add_argument("--max-index", type=int, default=10, help="highest index probed by --list-cameras")
    p.add_argument("--n_camera", "-n_camera", "--n-camera", type=int, default=1)
    p.add_argument("--cameras", "-cameras", default=None,
                   help="comma separated camera indices or video files, e.g. 0,1,2 (default: 0..n_camera-1)")
    p.add_argument("--model", default="yolo11x", help="yolo11n / yolo11s / yolo11m / yolo11l / yolo11x")
    p.add_argument("--format", choices=["pt", "openvino", "engine"], default=None)
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"], default=None)
    p.add_argument("--device", default=None, help="cpu, 0, intel:gpu ... (default from computer profile)")
    p.add_argument("--int8-data", default="coco8.yaml", help="calibration dataset for int8 export")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--duration", type=float, default=60.0, help="measurement seconds (after warmup)")
    p.add_argument("--warmup", type=int, default=30, help="warmup frames per camera (not measured)")
    p.add_argument("--width", type=int, default=1280, help="requested camera width")
    p.add_argument("--height", type=int, default=720, help="requested camera height")
    p.add_argument("--cam-fps", type=int, default=30, help="requested camera fps")
    p.add_argument("--cam-api", choices=["auto", *CAM_APIS], default="auto",
                   help="auto = msmf on Windows (dshow caps some UVC cams at 20fps), v4l2 on Linux")
    p.add_argument("--no-mjpg", action="store_true", help="do not request MJPG (MJPG is needed for several USB cams)")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--tag", default="", help="free text label stored with the run")
    p.add_argument("--force-export", action="store_true", help="re-export even if a cached model exists")
    args = p.parse_args()

    if args.cam_api == "auto":
        args.cam_api = "msmf" if sys.platform == "win32" else ("v4l2" if sys.platform.startswith("linux") else "any")
    if args.list_cameras:
        return args
    if not args.computer:
        p.error("--computer is required")

    prof = PROFILES[args.computer]
    args.format = args.format or prof["format"]
    args.precision = args.precision or prof["precision"]
    args.device = args.device or prof["device"]
    if args.format == "pt" and args.precision == "int8":
        p.error("int8 needs an exported format (openvino or engine)")
    if args.format == "engine" and args.device in ("cpu", "intel:cpu"):
        p.error("TensorRT engine needs a CUDA device")

    if args.cameras:
        sources = [int(s) if s.strip().isdigit() else s.strip() for s in args.cameras.split(",")]
    else:
        sources = list(range(args.n_camera))
    if len(sources) < args.n_camera:
        p.error(f"--n_camera {args.n_camera} but only {len(sources)} sources in --cameras")
    args.sources = sources[: args.n_camera]
    return args


# --------------------------------------------------------------------------- model download + convert
def prepare_model(args) -> tuple[Path, dict]:
    from ultralytics import YOLO

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    pt_path = models_dir / f"{args.model}.pt"
    info = {"pt_path": str(pt_path), "download_s": None, "export_s": None, "export_cached": False}

    if not pt_path.exists():
        t0 = time.perf_counter()
        YOLO(str(pt_path))  # Ultralytics downloads the official weights to this path
        info["download_s"] = round(time.perf_counter() - t0, 3)
        print(f"[model] downloaded {pt_path} in {info['download_s']} s")

    if args.format == "pt":
        info["model_path"] = str(pt_path)
        return pt_path, info

    target = models_dir / args.computer / f"{args.model}_{args.imgsz}_{args.precision}{EXPORT_SUFFIX[args.format]}"
    if target.exists() and not args.force_export:
        info["export_cached"] = True
        info["model_path"] = str(target)
        print(f"[model] using cached {target}")
        return target, info

    export_kw = dict(
        format=args.format,
        imgsz=args.imgsz,
        batch=1,
        dynamic=False,
        device=args.device,
        half=args.precision == "fp16",
        int8=args.precision == "int8",
    )
    if export_kw["int8"]:
        export_kw["data"] = args.int8_data
    print(f"[model] exporting {pt_path} -> {args.format} ({args.precision}, imgsz={args.imgsz}) ...")
    t0 = time.perf_counter()
    out = Path(YOLO(str(pt_path)).export(**export_kw))
    info["export_s"] = round(time.perf_counter() - t0, 3)

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    shutil.move(str(out), str(target))
    info["model_path"] = str(target)
    print(f"[model] export done in {info['export_s']} s -> {target}")
    return target, info


# --------------------------------------------------------------------------- camera
def open_camera(src, args) -> tuple[cv2.VideoCapture, dict]:
    t0 = time.perf_counter()
    is_file = isinstance(src, str)
    cap = cv2.VideoCapture(src) if is_file else cv2.VideoCapture(src, CAM_APIS[args.cam_api])
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera/source {src!r} (api={args.cam_api})")
    if not is_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.cam_fps)
        if not args.no_mjpg:  # after the size: with DSHOW a later size change resets the format to YUY2
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always process a fresh frame (honoured by V4L2/DSHOW)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"camera/source {src!r} opened but returned no frame")
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    props = {
        "source": str(src),
        "width": frame.shape[1],
        "height": frame.shape[0],
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)) if fourcc > 0 else "",
        "open_ms": round((time.perf_counter() - t0) * 1000, 1),
        "first_frame": frame,
    }
    return cap, props


def list_cameras(args):
    print(f"probing camera index 0..{args.max_index} (api={args.cam_api}, "
          f"requested {args.width}x{args.height}@{args.cam_fps}, mjpg={not args.no_mjpg})")
    for idx in range(args.max_index + 1):
        try:
            cap, props = open_camera(idx, args)
        except RuntimeError:
            continue
        cap.release()
        print(f"  index {idx}: {props['width']}x{props['height']} @{props['fps']:.0f}fps {props['fourcc']}")


# --------------------------------------------------------------------------- worker (one per camera)
class CameraWorker(threading.Thread):
    def __init__(self, cam_idx, src, model_path, args, barrier, setup_lock, clock):
        super().__init__(name=f"cam{cam_idx}", daemon=True)
        self.cam_idx, self.src, self.model_path, self.args = cam_idx, src, model_path, args
        self.barrier, self.setup_lock, self.clock = barrier, setup_lock, clock
        self.records: list[tuple] = []  # raw perf_counter based records
        self.cam_props: dict = {}
        self.load_ms = None
        self.dropped = 0
        self.error: BaseException | None = None

    def run(self):
        try:
            self._run()
        except BaseException as e:  # noqa: BLE001 - reported by main thread
            self.error = e
            self.barrier.abort()

    def _run(self):
        from ultralytics import YOLO

        a = self.args
        self.pred_kw = dict(imgsz=a.imgsz, conf=a.conf, classes=[PERSON_CLASS], device=a.device, verbose=False)
        if a.format == "pt" and a.precision == "fp16":
            self.pred_kw["half"] = True

        # Open camera + load model one thread at a time (clean timings, no driver races).
        with self.setup_lock:
            self.cap, props = open_camera(self.src, a)
            first = props.pop("first_frame")
            self.cam_props = props
            t0 = time.perf_counter()
            self.model = YOLO(str(self.model_path), task="detect")  # separate instance per thread (thread safe)
            self.model.predict(first, **self.pred_kw)  # backend init + first inference
            self.load_ms = round((time.perf_counter() - t0) * 1000, 1)
            print(f"[cam{self.cam_idx}] source={props['source']} {props['width']}x{props['height']} "
                  f"@{props['fps']:.0f}fps {props['fourcc']} open={props['open_ms']}ms load={self.load_ms}ms")
            if not isinstance(self.src, str) and (props["width"], props["height"]) != (a.width, a.height):
                print(f"[cam{self.cam_idx}] WARNING: requested {a.width}x{a.height}, "
                      f"camera delivers {props['width']}x{props['height']}")

        try:
            for i in range(a.warmup):
                self._step("warmup", i)
            self.barrier.wait(timeout=900)  # all cameras start measuring at the same moment
            t_stop = self.clock["origin"] + a.duration
            i = 0
            while time.perf_counter() < t_stop:
                self._step("measure", i)
                i += 1
        finally:
            self.cap.release()

    def _step(self, phase, idx):
        t0 = time.perf_counter()
        ok, frame = self.cap.read()  # trigger / grab the next camera frame
        t1 = time.perf_counter()
        if not ok:
            self.dropped += 1
            if isinstance(self.src, str):  # video file: rewind
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        r = self.model.predict(frame, **self.pred_kw)[0]
        t2 = time.perf_counter()
        sp = r.speed  # ms, measured by Ultralytics
        self.records.append((phase, idx, t0, t2, (t1 - t0) * 1000, sp["preprocess"], sp["inference"],
                             sp["postprocess"], (t2 - t1) * 1000, (t2 - t0) * 1000, len(r.boxes)))


# --------------------------------------------------------------------------- system info
def system_info(args) -> dict:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "cpu_count": os.cpu_count(),
    }
    try:
        from ultralytics import __version__ as ul_ver
        from ultralytics.utils.torch_utils import get_cpu_info

        info["ultralytics"] = ul_ver
        info["cpu"] = get_cpu_info()
    except Exception:  # noqa: BLE001
        info["cpu"] = platform.processor()
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    for mod in ("openvino", "tensorrt"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    if Path("/etc/nv_tegra_release").exists():
        info["nv_tegra_release"] = Path("/etc/nv_tegra_release").read_text().splitlines()[0]
    return info


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    if args.list_cameras:
        list_cameras(args)
        return
    run_id = f"{args.computer}_cam{args.n_camera}_{args.model}_{args.format}_{args.precision}_{args.imgsz}_" \
             f"{datetime.now():%Y%m%d_%H%M%S}"
    print(f"[run] {run_id}  sources={args.sources} device={args.device} duration={args.duration}s")

    model_path, model_info = prepare_model(args)

    clock = {}
    barrier = threading.Barrier(args.n_camera, action=lambda: clock.__setitem__("origin", time.perf_counter()))
    setup_lock = threading.Lock()
    workers = [CameraWorker(i, s, model_path, args, barrier, setup_lock, clock) for i, s in enumerate(args.sources)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    errors = [w for w in workers if w.error and not isinstance(w.error, threading.BrokenBarrierError)]
    if errors:
        for w in errors:
            print(f"[cam{w.cam_idx}] ERROR: {w.error!r}", file=sys.stderr)
        sys.exit(1)

    # ---- write per-frame CSV + run metadata
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{run_id}_frames.csv"
    origin = clock["origin"]
    with csv_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(CSV_FIELDS)
        for w in workers:
            cp = w.cam_props
            for (phase, idx, t0, t2, grab, pre, inf, post, pred, loop, n) in w.records:
                wr.writerow([run_id, args.computer, args.n_camera, args.model, args.format, args.precision,
                             args.imgsz, args.device, w.cam_idx, cp["source"], cp["width"], cp["height"],
                             round(cp["fps"], 2), phase, idx, round(t0 - origin, 6), round(t2 - origin, 6),
                             round(grab, 3), round(pre, 3), round(inf, 3), round(post, 3), round(pred, 3),
                             round(loop, 3), n])

    meta = {
        "run_id": run_id,
        "args": {k: v for k, v in vars(args).items() if k != "sources"} | {"sources": [str(s) for s in args.sources]},
        "model": model_info,
        "system": system_info(args),
        "cameras": [w.cam_props | {"load_ms": w.load_ms, "dropped": w.dropped} for w in workers],
    }
    (out_dir / f"{run_id}_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- quick summary (detailed analysis: analyze.py)
    print(f"\n=== {run_id} ===")
    total_fps = 0.0
    for w in workers:
        m = [r for r in w.records if r[0] == "measure"]
        if not m:
            print(f"cam{w.cam_idx}: no measured frames")
            continue
        fps = len(m) / (m[-1][3] - origin)
        total_fps += fps
        mean = lambda k: statistics.fmean(r[k] for r in m)  # noqa: E731
        print(f"cam{w.cam_idx}: {len(m):5d} frames  {fps:6.1f} FPS | grab {mean(4):6.1f}  pre {mean(5):5.1f}  "
              f"infer {mean(6):6.1f}  post {mean(7):5.1f}  loop {mean(9):6.1f} ms | dropped {w.dropped}")
    print(f"total: {total_fps:.1f} FPS over {args.n_camera} camera(s)")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
