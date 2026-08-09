# Qt Vision Bench

A real-time motion detection tool written in C++/Qt6/OpenCV, together with a
line-by-line Python reference implementation and a benchmark harness that
measures both on identical input.

![Application window](docs/screenshot.png)

## Why this exists

During my internship at Maptech I built a real-time object detection tool in
Python. It worked, but I never knew what the language was costing me. This
project answers that question properly: the same pipeline, implemented twice,
measured on the same frames, with a correctness check proving both versions do
the same work.

The answer was not the one I expected, and the investigation is the point of
the repository.

## Results

Intel Core i7-13620H, Windows 11, Release build (`-O3`), 150-frame synthetic
videos, 10 warm-up frames, median of 5 runs per cell.

| Resolution | C++ (ms) | Python (ms) | Python prealloc (ms) | C++ vs Python | C++ vs prealloc |
|---|---:|---:|---:|---:|---:|
| 480p | 1.324 | 1.561 | 1.484 | 1.18x | 1.12x |
| 720p | 3.788 | 4.526 | 3.953 | 1.19x | 1.04x |
| 1080p | 10.162 | 10.915 | 9.605 | 1.07x | **0.95x** |

![Benchmark across resolutions](bench/sweep.png)

"Python prealloc" is the same Python file with one change: every OpenCV call
receives a reused output buffer through `dst=`, mimicking what the C++ version
does with member `cv::Mat` objects. All three variants find the identical
number of detections at every resolution, and the harness refuses to report
results if they ever disagree.

The headline: **against a straightforward Python translation, C++ wins by
1.07x-1.19x. Against Python that manages its buffers the way the C++ code does,
the advantage nearly vanishes — and at 1080p it reverses.**

### Where the time actually goes

| Resolution | Stage | C++ | Python | Python prealloc |
|---|---|---:|---:|---:|
| 480p | preprocess | 0.144 | 0.194 | 0.161 |
| 480p | detection | 1.119 | 1.300 | 1.268 |
| 720p | preprocess | 0.260 | 0.371 | 0.270 |
| 720p | detection | 3.420 | 3.753 | 3.571 |
| 1080p | preprocess | **0.517** | **1.308** | **0.503** |
| 1080p | detection | 9.472 | 9.160 | 8.853 |

Two separate effects were hiding inside the single "C++ is faster" number.

**Effect 1 — allocation, not language.** At 1080p the preprocessing stage takes
1.308 ms in plain Python and 0.517 ms in C++, a 2.5x gap that grows with
resolution. Preallocating the output buffers drops Python to 0.503 ms, matching
C++ to within measurement noise. The gap was never interpreter speed: it was one
fresh NumPy array per intermediate image per frame, a cost that scales with pixel
count. The same fix in a Python codebase costs three lines and no rewrite.

**Effect 2 — the library build, not the language either.** In the detection
stage, which is dominated by a single MOG2 call, Python is *faster* at 1080p
(9.160 ms vs 9.472 ms). That call is native code on both sides — but not the
same native code:

| | C++ side | Python side |
|---|---|---|
| OpenCV | 5.0.0 (MSYS2 package) | 4.13.0 (`opencv-python` wheel) |
| Intel IPP | **not compiled in** | **2022.2.0** |
| Parallel framework | TBB 2023.1 | Concurrency (ConcRT) |
| Baseline arch flags | `-march=nocona` (2004-era x86-64) | vendor wheel defaults |
| Compiler | gcc 16.1.0 | MSVC (wheel) |

The MSYS2 package targets a conservative CPU baseline and ships without Intel's
Performance Primitives; the official `opencv-python` wheel bundles IPP 2022.2.0.
On small frames the per-call Python overhead hides this. On 1080p frames the
IPP-accelerated kernel wins outright, in a stage where the programming language
should be irrelevant.

### What I take from this

> For a pipeline that is mostly orchestration of OpenCV kernels, the language
> is close to irrelevant. What mattered was memory management (fixable in
> Python) and library build configuration (not a language property at all). The
> honest speedup from the C++ port is ~1.1x against careful Python, not the
> order of magnitude the framing usually implies.

That does not make the C++ version pointless — it has no interpreter runtime to
deploy, no GIL, and predictable latency — but those are the arguments that
survive measurement, and the throughput argument is not.

It also means the original Python internship tool was leaving roughly 15% on
the table for want of three lines, which is a considerably more useful finding
than "rewrite it in C++".

Reproduce with `py bench/sweep.py --repeat 5`; raw measurements land in
[`bench/sweep.json`](bench/sweep.json), tables in
[`bench/sweep.md`](bench/sweep.md).

## What the pipeline does

```
frame -> grayscale -> Gaussian blur -> MOG2 background subtraction
      -> morphological opening -> contours -> area filter -> bounding boxes
```

Canny edge detection is available as an alternative output view. Every stage
can be toggled and tuned live from the UI, which is the point of having a UI at
all: parameters like the minimum contour area are impossible to pick without
watching them work.

## Architecture

| File | Responsibility |
|---|---|
| `src/pipeline.h/.cpp` | The image processing pipeline. **Contains no Qt code.** |
| `src/worker.h/.cpp` | Frame capture and processing on a dedicated thread. |
| `src/mainwindow.h/.cpp` | UI, controls, live statistics. |
| `src/main.cpp` | Entry point, argument parsing, headless benchmark mode. |
| `tests/test_pipeline.cpp` | Unit tests, no framework dependency. |
| `bench/baseline.py` | The same pipeline in Python, with and without buffer reuse. |
| `bench/compare.py` | Quick single-resolution run with a parity check. |
| `bench/sweep.py` | All three variants across resolutions; tables and chart. |

Three decisions worth explaining:

**The pipeline is Qt-free.** That is what lets the same code run under the GUI,
under the headless benchmark, and under unit tests without a display.

**Capture runs on a worker thread, driven by a timer rather than a loop.** A
`while(true)` loop inside a slot would block that thread's event loop, so
`stop()` and parameter changes would never arrive. With a zero-interval
`QTimer`, the event loop breathes between frames — which also means the config
can be updated through queued signals with no mutex anywhere.

**Detections are suppressed during warm-up.** A freshly reset MOG2 model reports
the entire first frame as foreground. Without suppression, switching video
sources paints a bounding box around the whole screen. This was caught by a unit
test, not by watching the app.

## Building

Requires MSYS2 with the UCRT64 toolchain. From an MSYS2 UCRT64 shell:

```bash
./scripts/install_deps.sh   # first time only
./scripts/build.sh          # configure, build, run tests
```

Or manually:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

The build is warning-clean with `-Wall -Wextra -Wpedantic`.

## Running

```powershell
.\scripts\run.ps1
```

From an MSYS2 UCRT64 shell the executable also runs directly:

```bash
./build/qt_vision_bench.exe                              # GUI
./build/qt_vision_bench.exe --video bench/test_video.mp4 # GUI, start on a file
./build/qt_vision_bench.exe --bench bench/test_video.mp4 # headless, prints JSON
```

## Reproducing the benchmark

```powershell
py bench/make_test_video.py     # 480p sample for the single-resolution run
py bench/compare.py             # one resolution, quick
py bench/sweep.py --repeat 5    # all variants and resolutions, tables and chart
```

The test videos are synthetic and generated from a fixed seed, so anyone cloning
this repository measures the same frames, the same motion, and the same noise. A
recorded clip would have made the numbers machine-dependent for no benefit.

Both harness scripts exit non-zero if the implementations disagree on the number
of detections. A speed comparison between pipelines that are not doing the same
work is worthless, so the parity check gates the result rather than sitting in a
footnote.

## Limitations

- **The two sides link different OpenCV builds**, as analysed above. Until they
  are built from source with matching flags, the detection-stage numbers
  describe packaging, not language.
- Single machine, single scene, three resolutions.
- Timing covers the processing pipeline only; video decode is excluded from
  `process_ms` and included in `end_to_end_fps`.
- The GUI has no recording or export function; it is an inspection tool.

## Next steps

- Build both sides against one OpenCV compiled from source with identical flags,
  then re-run the sweep. This is the only way to isolate the language.
- Add a stage that is not a thin OpenCV wrapper — custom per-pixel logic, where
  the language difference should be large — and measure whether it is.
- Add an ONNX detector via `cv::dnn`. Based on the findings here it would likely
  show no language advantage at all, which is itself worth publishing.
