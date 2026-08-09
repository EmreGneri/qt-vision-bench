"""Runs the C++ and Python implementations on one video and tabulates the result.

Produces two files:
    bench/results.json  - raw measurements
    bench/results.md    - a markdown table ready to paste into the README

Usage (PowerShell):
    py bench/compare.py
    py bench/compare.py --video bench/test_video.mp4 --frames 500
    py bench/compare.py --history

Note: the C++ binary needs the Qt and OpenCV DLLs. This script prepends the
MSYS2 bin directory to PATH itself, so it also works from plain PowerShell. Set
the QVB_DLL_DIR environment variable if MSYS2 lives somewhere else.
"""

import argparse
import json
import os
import subprocess
import sys

# ==== SETTINGS ====
# Where the Qt/OpenCV DLLs live. Override with QVB_DLL_DIR.
DEFAULT_DLL_DIR = r"C:\msys64\ucrt64\bin"
DEFAULT_EXE = os.path.join("build", "qt_vision_bench.exe")
DEFAULT_VIDEO = os.path.join("bench", "test_video.mp4")
DEFAULT_WARMUP = 10
RESULTS_JSON = os.path.join("bench", "results.json")
RESULTS_MD = os.path.join("bench", "results.md")


def environment_with_dlls():
    """Returns an environment dict with the DLL directory ahead of PATH."""
    env = os.environ.copy()
    dll_dir = env.get("QVB_DLL_DIR", DEFAULT_DLL_DIR)
    if os.path.isdir(dll_dir):
        env["PATH"] = dll_dir + os.pathsep + env.get("PATH", "")
    return env


def run_json_command(command, env=None):
    """Runs a command and returns the JSON on its stdout as a dict."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
            check=False,
        )
    except FileNotFoundError:
        print(f"error: command not found: {command[0]}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"error: command did not finish within 600 s: {' '.join(command)}",
              file=sys.stderr)
        return None

    if completed.returncode != 0:
        print(f"error: command failed ({completed.returncode}): {' '.join(command)}",
              file=sys.stderr)
        print(completed.stderr.strip(), file=sys.stderr)
        return None

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        print(f"error: output is not JSON: {exc}", file=sys.stderr)
        print(completed.stdout[:500], file=sys.stderr)
        return None


def check_release_build(cpp_result):
    """A Debug binary would make every number here meaningless."""
    build_type = cpp_result.get("build_type", "")
    if build_type == "Release":
        return True
    print(
        f"error: the C++ binary reports build_type={build_type!r}; a timing "
        "comparison is only meaningful against an optimised build.\n"
        "Rebuild with: cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release",
        file=sys.stderr,
    )
    return False


def images_are_identical(path_a, path_b):
    """Pixel-exact comparison of two motion-history dumps.

    Returns (identical, message). Reported rather than assumed: the two sides
    link different OpenCV builds, so equality is a measurement, not a given.
    """
    import cv2  # imported here so the module stays usable without the dump
    import numpy as np

    left = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        return False, "history image missing"
    if left.shape != right.shape:
        return False, f"shape mismatch {left.shape} vs {right.shape}"

    diff = np.abs(left.astype(np.int32) - right.astype(np.int32))
    differing = int((diff > 0).sum())
    if differing == 0:
        return True, f"identical over {left.size:,} pixels"
    return False, (f"{differing:,}/{left.size:,} pixels differ, "
                   f"max |delta| = {int(diff.max())}")


def format_row(label, cpp_value, py_value, ratio_text):
    return f"| {label} | {cpp_value} | {py_value} | {ratio_text} |"


def build_markdown(cpp, py, history):
    """Turns the two results into a markdown table."""
    speedup_process = py["process_ms_mean"] / cpp["process_ms_mean"]
    speedup_endtoend = cpp["end_to_end_fps"] / py["end_to_end_fps"]

    stage_note = ("with the hand-written motion-history stage"
                  if history else "library-call pipeline only")
    lines = [
        f"{cpp['resolution']}, {cpp['frames_measured']} measured frames, "
        f"{stage_note}.",
        "",
        "| Metric | C++ / Qt | Python | C++ advantage |",
        "|---|---:|---:|---:|",
        format_row("Mean frame time (ms)",
                   f"{cpp['process_ms_mean']:.3f}",
                   f"{py['process_ms_mean']:.3f}",
                   f"{speedup_process:.2f}x"),
        format_row("Median frame time (ms)",
                   f"{cpp['process_ms_median']:.3f}",
                   f"{py['process_ms_median']:.3f}",
                   f"{py['process_ms_median'] / cpp['process_ms_median']:.2f}x"),
        format_row("p95 frame time (ms)",
                   f"{cpp['process_ms_p95']:.3f}",
                   f"{py['process_ms_p95']:.3f}",
                   f"{py['process_ms_p95'] / cpp['process_ms_p95']:.2f}x"),
        format_row("Preprocess mean (ms)",
                   f"{cpp['preprocess_ms_mean']:.3f}",
                   f"{py['preprocess_ms_mean']:.3f}",
                   f"{py['preprocess_ms_mean'] / cpp['preprocess_ms_mean']:.2f}x"),
        format_row("Detection mean (ms)",
                   f"{cpp['detect_ms_mean']:.3f}",
                   f"{py['detect_ms_mean']:.3f}",
                   f"{py['detect_ms_mean'] / cpp['detect_ms_mean']:.2f}x"),
        format_row("Processing throughput (fps)",
                   f"{cpp['processing_fps']:.1f}",
                   f"{py['processing_fps']:.1f}",
                   f"{cpp['processing_fps'] / py['processing_fps']:.2f}x"),
        format_row("End-to-end throughput (fps)",
                   f"{cpp['end_to_end_fps']:.1f}",
                   f"{py['end_to_end_fps']:.1f}",
                   f"{speedup_endtoend:.2f}x"),
        format_row("Detections found",
                   str(cpp["detections_total"]),
                   str(py["detections_total"]),
                   "parity check"),
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare C++ and Python pipelines")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--exe", default=DEFAULT_EXE)
    parser.add_argument("--frames", type=int, default=0, help="0 = whole video")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--history", action="store_true",
                        help="enable the hand-written motion-history stage")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"error: no such video: {args.video}\n"
              "Generate it first: py bench/make_test_video.py", file=sys.stderr)
        return 1

    if not os.path.isfile(args.exe):
        print(f"error: no such binary: {args.exe}\n"
              "Build it first: ./scripts/build.sh (in an MSYS2 UCRT64 shell)",
              file=sys.stderr)
        return 1

    common = ["--bench", args.video, "--warmup", str(args.warmup)]
    if args.frames > 0:
        common += ["--frames", str(args.frames)]

    cpp_history = os.path.join("bench", "history_cpp.png")
    py_history = os.path.join("bench", "history_python.png")
    if args.history:
        common += ["--history"]

    print("running the C++ measurement...")
    cpp_command = [args.exe] + common
    if args.history:
        cpp_command += ["--dump-history", cpp_history]
    cpp = run_json_command(cpp_command, env=environment_with_dlls())
    if cpp is None:
        return 1
    if not check_release_build(cpp):
        return 1

    print("running the Python measurement...")
    py_command = [sys.executable, os.path.join("bench", "baseline.py")] + common
    if args.history:
        py_command += ["--dump-history", py_history]
    py_result = run_json_command(py_command)
    if py_result is None:
        return 1

    # Equivalence check: different detection counts on the same video mean the
    # two pipelines are not doing the same work, and the speed comparison is void.
    parity_ok = cpp["detections_total"] == py_result["detections_total"]

    # The detection count says nothing about the motion-history stage, so that
    # image is compared pixel by pixel when the stage is on.
    history_parity_ok = None
    history_message = ""
    if args.history:
        history_parity_ok, history_message = images_are_identical(cpp_history, py_history)
        parity_ok = parity_ok and history_parity_ok

    markdown = build_markdown(cpp, py_result, args.history)

    payload = {
        "video": args.video,
        "frames_measured": cpp["frames_measured"],
        "parity_ok": parity_ok,
        "history_parity_ok": history_parity_ok,
        "history_parity_detail": history_message,
        "cpp": cpp,
        "python": py_result,
    }

    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(RESULTS_MD, "w", encoding="utf-8") as handle:
        handle.write(markdown + "\n")

    print()
    print(markdown)
    print()
    if cpp["detections_total"] == py_result["detections_total"]:
        print(f"Detection parity OK: both implementations found "
              f"{cpp['detections_total']} detections.")
    else:
        print("error: detection counts differ, the two pipelines are not doing the "
              f"same work. C++={cpp['detections_total']} "
              f"Python={py_result['detections_total']}", file=sys.stderr)

    if args.history:
        label = "OK" if history_parity_ok else "MISMATCH"
        stream = sys.stdout if history_parity_ok else sys.stderr
        print(f"Motion-history parity {label}: {history_message}", file=stream)

    print(f"\nWritten: {RESULTS_JSON}, {RESULTS_MD}")
    return 0 if parity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
