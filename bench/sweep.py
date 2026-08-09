"""Uc uygulamayi coklu cozunurlukte olcer, tablo ve grafik uretir.

Olculen uc varyant:
    cpp              - C++ / Qt hatti
    python           - dogrudan cevirisi (her cagri yeni NumPy dizisi ayirir)
    python-prealloc  - ayni Python kodu, ama OpenCV cagrilarina dst= ile
                       yeniden kullanilan tampon gecilerek

Ucuncu varyant bir kontrol grubu: C++ ile Python arasindaki farkin ne kadari
dilden, ne kadari kare basina bellek tahsisinden geliyor?

Uretilen dosyalar:
    bench/sweep.json  - ham olcumler
    bench/sweep.md    - markdown tablolar
    bench/sweep.png   - grafik

Kullanim (PowerShell):
    py bench/sweep.py
    py bench/sweep.py --repeat 5 --frames 300
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

import matplotlib

# Agg: pencere acmadan dosyaya cizer. Betik CI'da da kosabilsin diye.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend secimi importtan once olmali)

# compare.py ile ayni yardimcilari kullan - iki yerde ayni kodu tutmanin anlami yok
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import environment_with_dlls, run_json_command  # noqa: E402

# ==== AYARLAR (buradan degistir) ====
RESOLUTIONS = [
    (640, 480, "480p"),
    (1280, 720, "720p"),
    (1920, 1080, "1080p"),
]
DEFAULT_FRAMES = 150      # her cozunurlukte uretilen kare sayisi
DEFAULT_WARMUP = 10
DEFAULT_REPEAT = 3        # ayni olcum kac kez tekrarlanir; medyan raporlanir
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
    """make_test_video.py'yi cagirarak o cozunurlukte video uretir."""
    if os.path.isfile(path):
        print(f"  {path} zaten var, yeniden uretilmiyor")
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
        print(f"HATA: video uretilemedi ({width}x{height})", file=sys.stderr)
        print(completed.stderr.strip(), file=sys.stderr)
        return False
    return True


def measure(command, env, repeat, label, name):
    """Ayni olcumu repeat kez kosar, medyanlari ve yayilimi doner."""
    runs = []
    for attempt in range(1, repeat + 1):
        print(f"[{label}] {name} {attempt}/{repeat}")
        result = run_json_command(command, env=env)
        if result is None:
            return None
        runs.append(result)

    totals = [run["process_ms_mean"] for run in runs]
    return {
        # Medyan: arka planda calisan baska bir surecin tek seferlik etkisi
        # ortalamayi bozar, medyani bozmaz.
        "total_ms": statistics.median(totals),
        "total_min": min(totals),
        "total_max": max(totals),
        "pre_ms": statistics.median([run["preprocess_ms_mean"] for run in runs]),
        "det_ms": statistics.median([run["detect_ms_mean"] for run in runs]),
        "detections": runs[0]["detections_total"],
        "runs_ms": totals,
    }


def build_markdown(rows, repeat, frames):
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
        "Per-stage medians (ms):",
        "",
        "| Resolution | Stage | C++ | Python | Python prealloc |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        for stage_key, stage_name in (("pre_ms", "preprocess"), ("det_ms", "detection")):
            lines.append(
                f"| {row['label']} | {stage_name} | "
                f"{row['cpp'][stage_key]:.3f} | {row['python'][stage_key]:.3f} | "
                f"{row['python_prealloc'][stage_key]:.3f} |"
            )

    return "\n".join(lines)


def error_bars(rows, series):
    """Medyanin altindaki ve ustundeki en uc kosuya olan mesafe."""
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

    # 1.0 cizgisi kritik: altina dusen seri, C++'in geride kaldigi noktadir.
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

    # Ucuncu panel farkin kaynagini izole eder: on isleme adimi tahsis
    # agirlikli, tampon yeniden kullanimi devreye girince fark kapaniyor.
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
    args = parser.parse_args()

    if not os.path.isfile(args.exe):
        print(f"HATA: ikili yok: {args.exe}\n"
              "Once derleyin: ./scripts/build.sh (MSYS2 UCRT64 kabugunda)",
              file=sys.stderr)
        return 1

    env = environment_with_dlls()
    baseline = os.path.join("bench", "baseline.py")
    rows = []

    for width, height, label in RESOLUTIONS:
        video = os.path.join("bench", f"test_video_{label}.mp4")
        print(f"[{label}] video hazirlaniyor...")
        if not generate_video(width, height, args.frames, video):
            return 1

        common = ["--bench", video, "--warmup", str(args.warmup)]

        commands = {
            "cpp": [args.exe] + common,
            "python": [sys.executable, baseline] + common,
            "python_prealloc": [sys.executable, baseline] + common + ["--preallocate"],
        }

        row = {"label": label, "pixels": width * height}
        for series in SERIES:
            result = measure(commands[series], env, args.repeat, label,
                             SERIES_TITLES[series])
            if result is None:
                return 1
            row[series] = result

        # Esdegerlik: uc varyant da ayni sayida tespit bulmali. Bulmuyorsa
        # ayni isi yapmiyorlar ve hiz karsilastirmasi anlamsiz.
        detection_counts = {row[series]["detections"] for series in SERIES}
        row["parity_ok"] = len(detection_counts) == 1
        rows.append(row)

    markdown = build_markdown(rows, args.repeat, args.frames)
    draw_chart(rows, SWEEP_PNG)

    with open(SWEEP_JSON, "w", encoding="utf-8") as handle:
        json.dump({"repeat": args.repeat, "frames": args.frames, "rows": rows},
                  handle, indent=2)
    with open(SWEEP_MD, "w", encoding="utf-8") as handle:
        handle.write(markdown + "\n")

    print()
    print(markdown)
    print(f"\nWritten: {SWEEP_JSON}, {SWEEP_MD}, {SWEEP_PNG}")

    mismatches = [row["label"] for row in rows if not row["parity_ok"]]
    if mismatches:
        print(f"UYARI: tespit sayilari uyusmuyor: {', '.join(mismatches)}",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
