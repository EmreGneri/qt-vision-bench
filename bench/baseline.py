"""Python counterpart of the C++ pipeline - the reference implementation.

This file does exactly what src/pipeline.cpp does: the same stages, in the same
order, with the same defaults and the same warm-up behaviour. The language is
the only difference.

To keep the measurement fair:
  - identical order of operations and identical OpenCV calls,
  - timing covers the processing pipeline only, video decode is excluded,
  - the first frames (warm-up) are not measured,
  - the output JSON uses the same schema as the C++ side.

Usage (PowerShell):
    py bench/baseline.py --bench bench/test_video.mp4 --json bench/result_python.json
"""

import argparse
import json
import statistics
import sys
import time

import cv2
import numpy as np

# ==== SETTINGS (must match the C++ PipelineConfig defaults) ====
USE_GRAYSCALE = True
USE_BLUR = True
BLUR_KERNEL = 5
USE_CANNY = False
CANNY_LOW = 50.0
CANNY_HIGH = 150.0
USE_MOTION_DETECT = True
MIN_CONTOUR_AREA = 500
MOG2_VAR_THRESHOLD = 16.0
MOG2_HISTORY = 500
DRAW_BOXES = True
PIPELINE_WARMUP_FRAMES = 5   # the pipeline's own warm-up, same as the C++ side
HISTORY_DECAY = 240          # (value * decay) >> 8, identical to the C++ side

DEFAULT_BENCH_WARMUP = 10    # frames excluded from the measurement


class Pipeline:
    """Python equivalent of the Pipeline class in src/pipeline.h.

    With preallocate=True every OpenCV call receives the same buffer through
    `dst=`, mimicking the way the C++ side reuses its member cv::Mat objects.
    That variant exists to separate one question by measurement: is the gap the
    interpreter, or is it one fresh NumPy array per intermediate image per frame?
    """

    def __init__(self, preallocate=False, motion_history=False):
        # detectShadows=False: the C++ side disables shadow detection too
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=False,
        )
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.frames_since_reset = 0
        self.detections = []

        self.preallocate = preallocate
        # Reused buffers. They stay None while preallocate is off, so OpenCV
        # allocates a fresh array on every call.
        self._gray = None
        self._blurred = None
        self._edges = None
        self._mask = None
        self._output = None

        self.motion_history = motion_history
        self._history = None
        self._history_tmp = None

    def _dst(self, buffer):
        """Returns the buffer while preallocate is on, otherwise None."""
        return buffer if self.preallocate else None

    def motion_history_image(self):
        """The trail image (uint8), or None while the stage never ran."""
        return self._history

    def _update_motion_history(self, mask):
        """history = max((history * decay) >> 8, mask)

        The C++ side does this in a single pass with no temporary image. Reaching
        the same result in NumPy means traversing the image several times:
        multiply, shift, maximum, write back. Preallocation at least keeps the
        intermediate arrays from being reallocated, but the number of passes
        stays at four.
        """
        if self._history is None or self._history.shape != mask.shape:
            self._history = np.zeros(mask.shape, dtype=np.uint8)
            self._history_tmp = np.zeros(mask.shape, dtype=np.uint16)

        if self.preallocate:
            # dtype=np.uint16 is required: without it the multiplication can be
            # evaluated in uint8 and 255*240 overflows before it ever reaches the
            # uint16 output buffer. The rule differs between NumPy 1.x and 2.x,
            # so the accumulator type is stated explicitly.
            np.multiply(self._history, HISTORY_DECAY, out=self._history_tmp,
                        dtype=np.uint16)
            np.right_shift(self._history_tmp, 8, out=self._history_tmp)
            np.maximum(self._history_tmp, mask, out=self._history_tmp)
            np.copyto(self._history, self._history_tmp, casting="unsafe")
        else:
            decayed = (self._history.astype(np.uint16) * np.uint16(HISTORY_DECAY)) >> 8
            self._history = np.maximum(decayed, mask).astype(np.uint8)

    def process(self, frame):
        """Processes one frame. Returns (output_bgr, stats). Times in milliseconds."""
        t_start = time.perf_counter()
        self.detections = []

        work = frame

        # ---- preprocessing ----
        if USE_GRAYSCALE and len(frame.shape) == 3:
            self._gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self._dst(self._gray))
            work = self._gray

        if USE_BLUR:
            self._blurred = cv2.GaussianBlur(
                work, (BLUR_KERNEL, BLUR_KERNEL), 0, dst=self._dst(self._blurred)
            )
            work = self._blurred

        edges = None
        if USE_CANNY:
            canny_input = work
            if len(work.shape) == 3:
                canny_input = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            self._edges = cv2.Canny(
                canny_input, CANNY_LOW, CANNY_HIGH, edges=self._dst(self._edges)
            )
            edges = self._edges

        t_preprocess = time.perf_counter()

        # ---- detection ----
        if USE_MOTION_DETECT:
            self._mask = self.bg_sub.apply(work, self._dst(self._mask))
            mask = self._mask

            if self.frames_since_reset >= PIPELINE_WARMUP_FRAMES:
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN, self.morph_kernel, dst=self._dst(mask)
                )
                # Rebinding, not copying: the C++ side runs morphologyEx in place
                # and the motion-history stage below reads the morphed mask. If
                # this line were missing, the trail would be built from the raw
                # MOG2 output here and from the opened mask there - two different
                # images, silently.
                self._mask = mask

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                        continue
                    self.detections.append(cv2.boundingRect(contour))

            self.frames_since_reset += 1

        t_detect = time.perf_counter()

        # ---- motion history ----
        # Not updated before warm-up ends: same condition as the C++ side.
        history_active = (
            self.motion_history
            and USE_MOTION_DETECT
            and self.frames_since_reset > PIPELINE_WARMUP_FRAMES
            and self._mask is not None
        )
        if history_active:
            self._update_motion_history(self._mask)

        t_history = time.perf_counter()

        # ---- output image ----
        if self.motion_history and USE_MOTION_DETECT and self._history is not None:
            output = cv2.cvtColor(self._history, cv2.COLOR_GRAY2BGR,
                                  dst=self._dst(self._output))
        elif edges is not None:
            output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR, dst=self._dst(self._output))
        elif len(work.shape) == 2:
            output = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR, dst=self._dst(self._output))
        else:
            output = work.copy()
        self._output = output

        if DRAW_BOXES:
            for (x, y, w, h) in self.detections:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        t_end = time.perf_counter()

        stats = {
            "total_ms": (t_end - t_start) * 1000.0,
            "preprocess_ms": (t_preprocess - t_start) * 1000.0,
            "detect_ms": (t_detect - t_preprocess) * 1000.0,
            "history_ms": (t_history - t_detect) * 1000.0 if history_active else 0.0,
            "detections": len(self.detections),
        }
        return output, stats


def percentile(values, p):
    """Same linearly interpolated percentile as the C++ side."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (p / 100.0) * (len(ordered) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def run_bench(video_path, max_frames, warmup_frames, json_out, preallocate=False,
              motion_history=False, history_out=""):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"error: cannot open video: {video_path}", file=sys.stderr)
        return 1

    pipeline = Pipeline(preallocate=preallocate, motion_history=motion_history)
    total_ms, pre_ms, det_ms, hist_ms = [], [], [], []
    detections_total = 0
    seen = 0
    measured = 0

    wall_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        seen += 1

        _, stats = pipeline.process(frame)

        if seen <= warmup_frames:
            continue

        total_ms.append(stats["total_ms"])
        pre_ms.append(stats["preprocess_ms"])
        det_ms.append(stats["detect_ms"])
        hist_ms.append(stats["history_ms"])
        detections_total += stats["detections"]
        measured += 1

        if max_frames > 0 and measured >= max_frames:
            break

    wall_seconds = time.perf_counter() - wall_start
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if measured == 0:
        print(
            f"error: no frames measured (video too short for warmup={warmup_frames}?)",
            file=sys.stderr,
        )
        return 1

    # The trail image after the last frame, written losslessly so the harness can
    # compare it against the C++ one pixel by pixel.
    if history_out:
        if pipeline.motion_history_image() is None:
            print("error: --dump-history requires --history and a video long "
                  "enough to leave warm-up", file=sys.stderr)
            return 1
        if not cv2.imwrite(history_out, pipeline.motion_history_image()):
            print(f"error: cannot write history image to {history_out}", file=sys.stderr)
            return 1

    mean_total = statistics.fmean(total_ms)
    result = {
        "implementation": "python-prealloc" if preallocate else "python",
        "preallocate": preallocate,
        "video": video_path,
        "resolution": f"{width}x{height}",
        "frames_measured": measured,
        "warmup_frames": warmup_frames,
        "process_ms_mean": round(mean_total, 4),
        "process_ms_median": round(statistics.median(total_ms), 4),
        "process_ms_p95": round(percentile(total_ms, 95.0), 4),
        "process_ms_min": round(min(total_ms), 4),
        "process_ms_max": round(max(total_ms), 4),
        "preprocess_ms_mean": round(statistics.fmean(pre_ms), 4),
        "detect_ms_mean": round(statistics.fmean(det_ms), 4),
        "history_ms_mean": round(statistics.fmean(hist_ms), 4),
        "motion_history": motion_history,
        "processing_fps": round(1000.0 / mean_total, 2) if mean_total > 0 else 0.0,
        "end_to_end_fps": round(measured / wall_seconds, 2) if wall_seconds > 0 else 0.0,
        "wall_seconds": round(wall_seconds, 4),
        "detections_total": detections_total,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
    }

    text = json.dumps(result, indent=2)
    print(text)

    if json_out:
        try:
            with open(json_out, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError as exc:
            print(f"error: cannot write JSON to {json_out}: {exc}", file=sys.stderr)
            return 1

    return 0


def main():
    parser = argparse.ArgumentParser(description="Python reference implementation")
    parser.add_argument("--bench", required=True, help="video file to process")
    parser.add_argument("--frames", type=int, default=0, help="0 = whole video")
    parser.add_argument("--warmup", type=int, default=DEFAULT_BENCH_WARMUP)
    parser.add_argument("--json", dest="json_out", default="")
    parser.add_argument("--preallocate", action="store_true",
                        help="reuse output buffers via dst= (mimics the C++ side)")
    parser.add_argument("--history", action="store_true",
                        help="enable the hand-written motion-history stage")
    parser.add_argument("--dump-history", dest="history_out", default="",
                        help="write the final motion-history image (PNG)")
    args = parser.parse_args()

    return run_bench(args.bench, args.frames, args.warmup, args.json_out,
                     preallocate=args.preallocate, motion_history=args.history,
                     history_out=args.history_out)


if __name__ == "__main__":
    raise SystemExit(main())
