"""Analyze benchmark.py results (results/*_frames.csv) and compare computers / camera counts.

Usage
  uv run analyze.py                               # all runs in results/
  uv run analyze.py results/laptop_*_frames.csv   # selected runs
  uv run analyze.py --no-plot

Outputs (in --out, default report/)
  runs.csv     one row per run: total FPS, FPS per camera, step times mean/p50/p95, stability, scaling efficiency
  cameras.csv  one row per camera per run
  fps.png, steps.png, inference_box.png, fps_timeline.png
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

STEPS = ["grab_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "predict_ms", "loop_ms"]
CONFIG = ["computer", "n_camera", "model", "format", "precision", "imgsz", "device"]
SAME_SETUP = ["computer", "model", "format", "precision", "imgsz"]


def load(patterns: list[str]) -> pd.DataFrame:
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit(f"no result files match {patterns}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"loaded {len(files)} run file(s), {len(df)} frame rows")
    return df[df["phase"] == "measure"].copy()


def per_camera(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["run_id", "cam"])
    cam = g.agg(source=("source", "first"), cam_fps=("cam_fps", "first"), frames=("frame", "size"),
                duration_s=("t_end_s", "max"), persons_mean=("n_person", "mean"))
    cam["fps"] = cam["frames"] / cam["duration_s"]
    for s in STEPS:
        cam[f"{s}_mean"] = g[s].mean()
    # FPS stability: frames finished in each 1 s window
    per_sec = df.assign(sec=df["t_end_s"].astype(int)).groupby(["run_id", "cam", "sec"]).size().reset_index(name="n")
    per_sec = per_sec[per_sec["sec"] < per_sec.groupby(["run_id", "cam"])["sec"].transform("max")]  # drop partial last
    cam["fps_1s_std"] = per_sec.groupby(["run_id", "cam"])["n"].std()
    return cam.reset_index()


def per_run(df: pd.DataFrame, cam: pd.DataFrame) -> pd.DataFrame:
    run = df.groupby("run_id")[CONFIG].first()
    c = cam.groupby("run_id")
    run["total_fps"] = c["fps"].sum()
    run["fps_per_cam"] = c["fps"].mean()
    run["fps_per_cam_min"] = c["fps"].min()
    run["fps_1s_std"] = c["fps_1s_std"].mean()
    run["cam_fps"] = c["cam_fps"].median()
    run["frames"] = c["frames"].sum()

    g = df.groupby("run_id")
    for s in STEPS:
        run[f"{s}_mean"] = g[s].mean()
        run[f"{s}_p50"] = g[s].median()
        run[f"{s}_p95"] = g[s].quantile(0.95)
    # model-only throughput (what the hardware could do if the camera were infinitely fast)
    run["infer_only_fps"] = 1000.0 / run["inference_ms_mean"]
    run["predict_only_fps"] = 1000.0 / run["predict_ms_mean"]
    run["loop_cv"] = g["loop_ms"].std() / g["loop_ms"].mean()  # jitter: lower = more stable
    run["camera_bound"] = run["fps_per_cam"] >= 0.9 * run["cam_fps"]

    # scaling efficiency vs. the latest 1-camera run with the same setup
    run = run.reset_index()
    eff = []
    for _, r in run.iterrows():
        base = run[(run["n_camera"] == 1) & (run[SAME_SETUP] == r[SAME_SETUP]).all(axis=1)]
        if r["n_camera"] == 1 or base.empty:
            eff.append(None)
        else:
            single = base.sort_values("run_id").iloc[-1]["total_fps"]
            eff.append(r["total_fps"] / (r["n_camera"] * single))
    run["scaling_eff"] = eff
    return run.sort_values(["computer", "n_camera", "run_id"]).reset_index(drop=True)


def label(r) -> str:
    return f"{r['computer']} {r['n_camera']}cam\n{r['format']}-{r['precision']} {r['imgsz']}"


def plot(df: pd.DataFrame, run: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [label(r) for _, r in run.iterrows()]
    if len(set(labels)) < len(labels):  # repeated runs of the same setup
        labels = [f"{lb}\n#{i}" for i, lb in enumerate(labels)]
    x = range(len(run))
    w = max(6, 1.6 * len(run))

    # FPS
    fig, ax = plt.subplots(figsize=(w, 4.5))
    ax.bar([i - 0.2 for i in x], run["total_fps"], 0.4, label="total FPS (all cameras)")
    ax.bar([i + 0.2 for i in x], run["fps_per_cam"], 0.4, label="FPS per camera")
    ax.plot(list(x), run["infer_only_fps"], "k_", markersize=25, mew=2, label="1000 / inference_ms")
    ax.set_xticks(list(x), labels, fontsize=8)
    ax.set_ylabel("FPS")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fps.png", dpi=120)

    # mean time per step (stacked)
    parts = {
        "grab": run["grab_ms_mean"],
        "preprocess": run["preprocess_ms_mean"],
        "inference": run["inference_ms_mean"],
        "postprocess": run["postprocess_ms_mean"],
        "other": (run["predict_ms_mean"] - run["preprocess_ms_mean"] - run["inference_ms_mean"]
                  - run["postprocess_ms_mean"]).clip(lower=0),
    }
    fig, ax = plt.subplots(figsize=(w, 4.5))
    bottom = pd.Series(0.0, index=run.index)
    for name, v in parts.items():
        ax.bar(list(x), v, bottom=bottom, label=name)
        bottom += v
    ax.set_xticks(list(x), labels, fontsize=8)
    ax.set_ylabel("mean time per frame [ms]")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "steps.png", dpi=120)

    # inference distribution
    fig, ax = plt.subplots(figsize=(w, 4.5))
    ax.boxplot([df.loc[df["run_id"] == rid, "inference_ms"] for rid in run["run_id"]], showfliers=False)
    ax.set_xticks([i + 1 for i in x], labels, fontsize=8)
    ax.set_ylabel("inference [ms]")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "inference_box.png", dpi=120)

    # total FPS over time
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (_, r), lb in zip(run.iterrows(), labels):
        d = df[df["run_id"] == r["run_id"]]
        s = d.groupby(d["t_end_s"].astype(int)).size()
        ax.plot(s.index[:-1], s.values[:-1], label=lb.replace("\n", " "))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("total FPS (1 s window)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fps_timeline.png", dpi=120)
    plt.close("all")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*", default=["results/*_frames.csv"])
    p.add_argument("--out", default="report")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    df = load(args.inputs)
    cam = per_camera(df)
    run = per_run(df, cam)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run.to_csv(out / "runs.csv", index=False)
    cam.to_csv(out / "cameras.csv", index=False)

    cols = ["computer", "n_camera", "format", "precision", "imgsz", "total_fps", "fps_per_cam", "fps_per_cam_min",
            "grab_ms_mean", "preprocess_ms_mean", "inference_ms_mean", "inference_ms_p95", "postprocess_ms_mean",
            "loop_ms_mean", "loop_ms_p95", "infer_only_fps", "loop_cv", "fps_1s_std", "camera_bound", "scaling_eff"]
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.2f}".format):
        print("\n=== runs ===")
        print(run[cols].to_string(index=False))
        print("\n=== total FPS: computer x n_camera (latest run) ===")
        print(run.pivot_table(index="computer", columns="n_camera", values="total_fps", aggfunc="last"))
        print("\n=== mean inference ms: computer x n_camera (latest run) ===")
        print(run.pivot_table(index="computer", columns="n_camera", values="inference_ms_mean", aggfunc="last"))
    if run["camera_bound"].any():
        print("\nNOTE: camera_bound=True -> FPS is limited by the camera frame rate, not by compute. "
              "Compare infer_only_fps / inference_ms for those runs.")

    if not args.no_plot:
        plot(df, run, out)
    print(f"\nwritten: {out / 'runs.csv'}, {out / 'cameras.csv'}" + ("" if args.no_plot else ", *.png"))


if __name__ == "__main__":
    main()
