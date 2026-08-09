"""Measures three implementations across several resolutions; writes tables and a chart.

The three variants:
    cpp              - the C++ / Qt pipeline
    python           - a direct translation (every call allocates a new NumPy array)
    python-prealloc  - the same Python code, but with reused buffers passed into
                       every OpenCV call through dst=

The third variant is the control group: how much of the gap between C++ and
Python is the language, and how much is one allocation per intermediate image
per frame?

Output files:
    bench/sweep.json  - raw measurements
    bench/sweep.md    - markdown tables
    bench/sweep.png   - chart

Usage (PowerShell):
    py bench/sweep.py
    py bench/sweep.py --repeat 5 --frames 300
    py bench/sweep.py --repeat 5 --history
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

import matplotlib

# Agg: draws to a file without opening a window, so the script also runs in CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen before this)

# Share compare.py's helpers - no reason to keep two copies of the same code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import (  # noqa: E402
    check_release_build,
    environment_with_dlls,
    images_are_identical,
    run_json_command,
)

# ==== SETTINGS ====
RESOLUTIONS = [
    (640, 480, "480p"),
    (1280, 720, "720p"),
    (1920, 1080, "1080p"),
]
DEFAULT_FRAMES = 150      # frames generated per resolution
DEFAULT_WARMUP = 10
DEFAULT_REPEAT = 3        # how often each measurement repeats; the median is reported
DEFAULT_EXE = os.path.join("build", "qt_vision_bench.exe")
SWEEP_JSON = os.path.join("bench", "sweep.json")
SWEEP_MD = os.path.join("bench", "sweep.md")
SWEEP_PNG = os.path.join("bench", "sweep.png")

SERIES = ["cpp", "python", "python_prealloc"]
SERIES_TITLES = {
    "cpp": "C++ / Qt",
    "python": "Python",
    "python_prealloc": "Python (preallocated)",
}


def generate_video(width, height, frames, path):
    """Calls make_test_video.py to produce a clip at that resolution."""
    if os.path.isfile(path):
        print(f"  {path} already exists, not regenerating")
        return True

    command = [
        sys.executable,
        os.path.join("bench", "make_test_video.py"),
        "--output", path,
        "--width", str(width),
        "--height", str(height),
        "--frames", str(frames),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        print(f"error: could not generate the video ({width}x{height})", file=sys.stderr)
        print(completed.stderr.strip(), file=sys.stderr)
        return False
    return True


def measure(command, env, repeat, label, name):
    """Runs the same measurement `repeat` times, returns medians and spread."""
    runs = []
    for attempt in range(1, repeat + 1):
        print(f"[{label}] {name} {attempt}/{repeat}")
        result = run_json_command(command, env=env)
        if result is None:
            return None
        runs.append(result)

    totals = [run["process_ms_mean"] for run in runs]
    return {
        # Median: a one-off spike from some other process on the machine ruins
        # the mean but not the median.
        "total_ms": statistics.median(totals),
        "total_min": min(totals),
        "total_max": max(totals),
        "pre_ms": statistics.median([run["preprocess_ms_mean"] for run in runs]),
        "det_ms": statistics.median([run["detect_ms_mean"] for run in runs]),
        "hist_ms": statistics.median([run.get("history_ms_mean", 0.0) for run in runs]),
        "detections": runs[0]["detections_total"],
        "runs_ms": totals,
        "meta": runs[0],
    }


def build_markdown(rows, repeat, frames, history):
    lines = [
        f"Median of {repeat} runs per cell, {frames} frames per run; the range "
        "in parentheses is min-max across runs.",
        "",
        "| Resolution | Pixels | C++ (ms) | Python (ms) | Python prealloc (ms) "
        "| C++ vs Python | C++ vs prealloc | Parity |",
        "|---|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for row in rows:
        cpp = row["cpp"]
        py = row["python"]
        pre = row["python_prealloc"]
        parity = "ok" if row["parity_ok"] else "MISMATCH"
        lines.append(
            f"| {row['label']} | {row['pixels']:,} | "
            f"{cpp['total_ms']:.3f} ({cpp['total_min']:.3f}-{cpp['total_max']:.3f}) | "
            f"{py['total_ms']:.3f} ({py['total_min']:.3f}-{py['total_max']:.3f}) | "
            f"{pre['total_ms']:.3f} ({pre['total_min']:.3f}-{pre['total_max']:.3f}) | "
            f"{py['total_ms'] / cpp['total_ms']:.2f}x | "
            f"{pre['total_ms'] / cpp['total_ms']:.2f}x | {parity} |"
        )

    lines += [
        "",
        "Parity is gated, not annotated: the harness exits non-zero if any cell "
        "disagrees.",
    ]

    if history:
        lines += [
            "",
            "| Resolution | Motion-history images (C++ vs Python) |",
            "|---|---|",
        ]
        for row in rows:
            state = "identical" if row["history_parity_ok"] else "MISMATCH"
            lines.append(f"| {row['label']} | {state} - {row['history_parity_detail']} |")

    stages = [("pre_ms", "preprocess"), ("det_ms", "detection")]
    # No point filling rows with zeros while the motion-history stage is off
    if any(row["cpp"]["hist_ms"] > 0.0 for row in rows):
        stages.append(("hist_ms", "motion history"))

    lines += [
        "",
        "Per-stage medians (ms):",
        "",
        "| Resolution | Stage | C++ | Python | Python prealloc |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        for stage_key, stage_name in stages:
            lines.append(
                f"| {row['label']} | {stage_name} | "
                f"{row['cpp'][stage_key]:.3f} | {row['python'][stage_key]:.3f} | "
                f"{row['python_prealloc'][stage_key]:.3f} |"
            )

    # The environment is part of the result: the two sides link different OpenCV
    # builds, and the detection-stage numbers cannot be read without knowing that.
    cpp_meta = rows[0]["cpp"]["meta"]
    py_meta = rows[0]["python"]["meta"]
    lines += [
        "",
        "Environment:",
        "",
        "| | C++ side | Python side |",
        "|---|---|---|",
        f"| OpenCV | {cpp_meta.get('opencv_version', '?')} | "
        f"{py_meta.get('opencv_version', '?')} |",
        f"| Toolchain | {cpp_meta.get('compiler', '?')}, "
        f"{cpp_meta.get('build_type', '?')} | Python "
        f"{py_meta.get('python_version', '?')}, NumPy "
        f"{py_meta.get('numpy_version', '?')} |",
    ]

    return "\n".join(lines)


def error_bars(rows, series):
    """Distance from the median to the slowest and fastest run."""
    lower = [row[series]["total_ms"] - row[series]["total_min"] for row in rows]
    upper = [row[series]["total_max"] - row[series]["total_ms"] for row in rows]
    return [lower, upper]


def draw_chart(rows, path):
    labels = [row["label"] for row in rows]
    figure, (ax_time, ax_ratio, ax_pre) = plt.subplots(1, 3, figsize=(15.5, 4.4))

    x = list(range(len(labels)))
    width = 0.27
    offsets = {"cpp": -width, "python": 0.0, "python_prealloc": width}

    for series in SERIES:
        ax_time.bar(
            [i + offsets[series] for i in x],
            [row[series]["total_ms"] for row in rows],
            width,
            label=SERIES_TITLES[series],
            yerr=error_bars(rows, series),
            capsize=3,
        )
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels)
    ax_time.set_ylabel("mean frame time (ms)")
    ax_time.set_title("Total processing time per frame")
    ax_time.legend(fontsize=8)
    ax_time.grid(axis="y", alpha=0.3)

    # The 1.0 line matters: a series below it is where C++ is the slower one.
    ax_ratio.plot(labels, [row["python"]["total_ms"] / row["cpp"]["total_ms"] for row in rows],
                  marker="o", label="vs Python")
    ax_ratio.plot(labels,
                  [row["python_prealloc"]["total_ms"] / row["cpp"]["total_ms"] for row in rows],
                  marker="s", label="vs Python (preallocated)")
    ax_ratio.axhline(1.0, linestyle="--", linewidth=1, alpha=0.6)
    ax_ratio.set_ylabel("C++ advantage (x)")
    ax_ratio.set_title("C++ advantage, by comparison target")
    ax_ratio.legend(fontsize=8)
    ax_ratio.grid(alpha=0.3)

    # The third panel isolates where the difference comes from: preprocessing is
    # allocation-heavy, and the gap closes once the buffers are reused.
    for series in SERIES:
        ax_pre.bar(
            [i + offsets[series] for i in x],
            [row[series]["pre_ms"] for row in rows],
            width,
            label=SERIES_TITLES[series],
        )
    ax_pre.set_xticks(x)
    ax_pre.set_xticklabels(labels)
    ax_pre.set_ylabel("preprocess time (ms)")
    ax_pre.set_title("Preprocess stage: the allocation cost")
    ax_pre.legend(fontsize=8)
    ax_pre.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Benchmark across resolutions")
    parser.add_argument("--exe", default=DEFAULT_EXE)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT,
                        help="runs per cell; the median is reported")
    parser.add_argument("--history", action="store_true",
                        help="enable the hand-written motion-history stage in all variants")
    args = parser.parse_args()

    # The two modes write to separate files so neither overwrites the other
    suffix = "_history" if args.history else ""
    out_json = SWEEP_JSON.replace(".json", f"{suffix}.json")
    out_md = SWEEP_MD.replace(".md", f"{suffix}.md")
    out_png = SWEEP_PNG.replace(".png", f"{suffix}.png")

    if not os.path.isfile(args.exe):
        print(f"error: no such binary: {args.exe}\n"
              "Build it first: ./scripts/build.sh (in an MSYS2 UCRT64 shell)",
              file=sys.stderr)
        return 1

    env = environment_with_dlls()
    baseline = os.path.join("bench", "baseline.py")
    rows = []

    for width, height, label in RESOLUTIONS:
        video = os.path.join("bench", f"test_video_{label}.mp4")
        print(f"[{label}] preparing the video...")
        if not generate_video(width, height, args.frames, video):
            return 1

        common = ["--bench", video, "--warmup", str(args.warmup)]
        if args.history:
            common = common + ["--history"]

        commands = {
            "cpp": [args.exe] + common,
            "python": [sys.executable, baseline] + common,
            "python_prealloc": [sys.executable, baseline] + common + ["--preallocate"],
        }

        # The trail is dumped from the last of the repeated runs; every run over
        # the same deterministic video produces the same image.
        history_paths = {}
        if args.history:
            for series in SERIES:
                history_paths[series] = os.path.join(
                    "bench", f"history_{series}_{label}.png")
                commands[series] = commands[series] + [
                    "--dump-history", history_paths[series]]

        row = {"label": label, "pixels": width * height}
        for series in SERIES:
            result = measure(commands[series], env, args.repeat, label,
                             SERIES_TITLES[series])
            if result is None:
                return 1
            row[series] = result

        if not check_release_build(row["cpp"]["meta"]):
            return 1

        # Equivalence: all three variants must find the same number of
        # detections. If they do not, they are not doing the same work and the
        # speed comparison means nothing.
        detection_counts = {row[series]["detections"] for series in SERIES}
        row["parity_ok"] = len(detection_counts) == 1

        # Detection counts say nothing about the hand-written stage, so its
        # output image is compared pixel by pixel as well.
        row["history_parity_ok"] = None
        row["history_parity_detail"] = ""
        if args.history:
            checks = [
                images_are_identical(history_paths["cpp"], history_paths["python"]),
                images_are_identical(history_paths["cpp"],
                                     history_paths["python_prealloc"]),
            ]
            row["history_parity_ok"] = all(ok for ok, _ in checks)
            row["history_parity_detail"] = "; ".join(detail for _, detail in checks)
            row["parity_ok"] = row["parity_ok"] and row["history_parity_ok"]

        rows.append(row)

    markdown = build_markdown(rows, args.repeat, args.frames, args.history)
    draw_chart(rows, out_png)

    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump({"repeat": args.repeat, "frames": args.frames,
                   "motion_history": args.history, "rows": rows},
                  handle, indent=2)
    with open(out_md, "w", encoding="utf-8") as handle:
        handle.write(markdown + "\n")

    print()
    print(markdown)
    print(f"\nWritten: {out_json}, {out_md}, {out_png}")

    mismatches = [row["label"] for row in rows if not row["parity_ok"]]
    if mismatches:
        print(f"error: parity check failed at: {', '.join(mismatches)}",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
