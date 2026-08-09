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

The interesting part is not "C++ is faster." It is *how much*, *where*, and
*why* — and being able to defend the measurement.

## Results

Measured on an Intel Core i7-13620H, Windows 11, Release build (`-O3`),
640x480 synthetic video, 290 measured frames after 10 warm-up frames:

| Metric | C++ / Qt | Python | C++ advantage |
|---|---:|---:|---:|
| Mean frame time (ms) | 1.378 | 1.634 | 1.19x |
| Median frame time (ms) | 1.350 | 1.640 | 1.21x |
| p95 frame time (ms) | 1.715 | 1.822 | 1.06x |
| Preprocess mean (ms) | 0.128 | 0.214 | 1.67x |
| Detection mean (ms) | 1.190 | 1.278 | 1.07x |
| Processing throughput (fps) | 725.9 | 612.0 | 1.19x |
| End-to-end throughput (fps) | 609.7 | 409.4 | 1.49x |
| Detections found | 680 | 680 | parity check |

Reproduce with `py bench/compare.py`; the numbers above are the contents of
[`bench/results.md`](bench/results.md).

### Reading the numbers honestly

**The gap is modest, and that is the expected result.** Both implementations
call the same native OpenCV kernels. `cv2.GaussianBlur` is not "Python code" —
it is the same C++ routine reached through a binding. What C++ actually saves
is the per-call overhead around those kernels: interpreter dispatch, argument
marshalling, and NumPy array allocation for every intermediate image.

That explains the shape of the table:

- **Preprocess is 1.67x faster** — several cheap kernels back to back, so
  per-call overhead dominates. This is also where the C++ version reuses
  preallocated `cv::Mat` buffers instead of allocating a new array per frame.
- **Detection is only 1.07x faster** — one expensive MOG2 call dominates the
  step, and that cost is identical on both sides.
- **End-to-end is 1.49x faster** — the per-frame Python overhead in the capture
  loop is added on top of the processing gap.

A benchmark that showed C++ ten times faster here would mean the measurement
was rigged, not that the port was good.

### Known limitations

- **The two sides use different OpenCV major versions.** C++ links OpenCV 5.0.0
  (from MSYS2); the Python side runs `opencv-python` 4.13.0, the newest wheel
  available. Kernel-level improvements between the versions are therefore mixed
  into the comparison. Matching the versions is the first thing I would fix
  before quoting these numbers as final.
- Single machine, single resolution, single scene. The harness makes it cheap
  to widen this, but it has not been done yet.
- Timing covers the processing pipeline only; video decode is excluded from
  `process_ms` and included in `end_to_end_fps`.

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
| `bench/baseline.py` | The same pipeline in Python. |
| `bench/compare.py` | Runs both, checks parity, writes the results table. |

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
py bench/make_test_video.py
py bench/compare.py
```

The test video is synthetic and generated from a fixed seed, so anyone cloning
this repository measures the same frames, the same motion, and the same noise.
A recorded clip would have made the numbers machine-dependent.

`compare.py` fails with a non-zero exit code if the two implementations disagree
on the number of detections — a speed comparison between two pipelines that are
not doing the same work is worthless.

## Next steps

- Match OpenCV versions across both implementations.
- Sweep resolutions (480p / 720p / 1080p) and plot frame time against pixel count.
- Add an ONNX detector stage via `cv::dnn` as an optional comparison. Note that
  this would likely show *no* C++ advantage, since inference is the same native
  code on both sides — which is itself a result worth publishing.
