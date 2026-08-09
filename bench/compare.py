"""C++ ve Python uygulamalarini ayni video uzerinde kosturur, sonucu tablolar.

Iki cikti uretir:
    bench/results.json  - ham olcumler
    bench/results.md    - README'ye yapistirilabilir markdown tablo

Kullanim (PowerShell):
    py bench/compare.py
    py bench/compare.py --video bench/test_video.mp4 --frames 500

Not: C++ ikilisi Qt ve OpenCV DLL'lerine ihtiyac duyuyor. Bu betik MSYS2'nin
bin klasorunu PATH'e kendisi ekliyor, boylece duz PowerShell'den de calisiyor.
"""

import argparse
import json
import os
import subprocess
import sys

# ==== AYARLAR (buradan degistir) ====
MSYS2_BIN = r"C:\msys64\ucrt64\bin"          # Qt/OpenCV DLL'lerinin yeri
DEFAULT_EXE = os.path.join("build", "qt_vision_bench.exe")
DEFAULT_VIDEO = os.path.join("bench", "test_video.mp4")
DEFAULT_WARMUP = 10
RESULTS_JSON = os.path.join("bench", "results.json")
RESULTS_MD = os.path.join("bench", "results.md")


def environment_with_dlls():
    """MSYS2 bin klasorunu PATH'in basina ekleyen bir ortam sozlugu doner."""
    env = os.environ.copy()
    if os.path.isdir(MSYS2_BIN):
        env["PATH"] = MSYS2_BIN + os.pathsep + env.get("PATH", "")
    return env


def run_json_command(command, env=None):
    """Komutu calistirir, stdout'taki JSON'u sozluk olarak doner."""
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
        print(f"HATA: komut bulunamadi: {command[0]}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"HATA: komut 600 sn icinde bitmedi: {' '.join(command)}", file=sys.stderr)
        return None

    if completed.returncode != 0:
        print(f"HATA: komut basarisiz ({completed.returncode}): {' '.join(command)}",
              file=sys.stderr)
        print(completed.stderr.strip(), file=sys.stderr)
        return None

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        print(f"HATA: cikti JSON degil: {exc}", file=sys.stderr)
        print(completed.stdout[:500], file=sys.stderr)
        return None


def format_row(label, cpp_value, py_value, ratio_text):
    return f"| {label} | {cpp_value} | {py_value} | {ratio_text} |"


def build_markdown(cpp, py):
    """Iki sonucu markdown tabloya cevirir."""
    speedup_process = py["process_ms_mean"] / cpp["process_ms_mean"]
    speedup_endtoend = cpp["end_to_end_fps"] / py["end_to_end_fps"]

    lines = [
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
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"HATA: video yok: {args.video}\n"
              "Once uretin: py bench/make_test_video.py", file=sys.stderr)
        return 1

    if not os.path.isfile(args.exe):
        print(f"HATA: ikili yok: {args.exe}\n"
              "Once derleyin: ./scripts/build.sh (MSYS2 UCRT64 kabugunda)",
              file=sys.stderr)
        return 1

    common = ["--bench", args.video, "--warmup", str(args.warmup)]
    if args.frames > 0:
        common += ["--frames", str(args.frames)]

    print("C++ olcumu calisiyor...")
    cpp = run_json_command([args.exe] + common, env=environment_with_dlls())
    if cpp is None:
        return 1

    print("Python olcumu calisiyor...")
    py_result = run_json_command(
        [sys.executable, os.path.join("bench", "baseline.py")] + common
    )
    if py_result is None:
        return 1

    # Esdegerlik kontrolu: ayni videoda ayni sayida tespit cikmiyorsa iki hat
    # ayni isi yapmiyor demektir ve hiz karsilastirmasi anlamsizdir.
    parity_ok = cpp["detections_total"] == py_result["detections_total"]

    markdown = build_markdown(cpp, py_result)

    payload = {
        "video": args.video,
        "frames_measured": cpp["frames_measured"],
        "parity_ok": parity_ok,
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
    if parity_ok:
        print(f"Parity OK: both implementations found {cpp['detections_total']} detections.")
    else:
        print("UYARI: tespit sayilari farkli, iki hat ayni isi yapmiyor. "
              f"C++={cpp['detections_total']} Python={py_result['detections_total']}",
              file=sys.stderr)

    print(f"\nWritten: {RESULTS_JSON}, {RESULTS_MD}")
    return 0 if parity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
