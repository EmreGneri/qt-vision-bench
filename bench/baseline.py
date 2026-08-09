"""C++ hattinin Python karsiligi - karsilastirma icin referans uygulama.

Bu dosya src/pipeline.cpp'nin birebir aynisini yapar: ayni adimlar, ayni sira,
ayni varsayilan parametreler, ayni isinma davranisi. Tek fark dil.

Olcum adil olsun diye:
  - Ayni islem sirasi ve ayni OpenCV cagrilari kullanilir.
  - Sure yalnizca isleme hattini kapsar, video cozme (decode) disaridadir.
  - Ilk kareler (isinma) olcume girmez.
  - Cikti JSON semasi C++ tarafiyla ayni.

Kullanim (PowerShell):
    py bench/baseline.py --bench bench/test_video.mp4 --json bench/result_python.json
"""

import argparse
import json
import statistics
import sys
import time

import cv2
import numpy as np

# ==== AYARLAR (C++ PipelineConfig varsayilanlariyla ayni olmali) ====
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
PIPELINE_WARMUP_FRAMES = 5   # hattin kendi isinmasi (C++ tarafiyla ayni)

DEFAULT_BENCH_WARMUP = 10    # olcume girmeyen kare sayisi


class Pipeline:
    """src/pipeline.h icindeki Pipeline sinifinin Python esdegeri.

    preallocate=True verilirse her OpenCV cagrisina `dst=` ile ayni tampon
    gecilir, yani C++ tarafinin uye cv::Mat'leri yeniden kullanmasi taklit
    edilir. Bu, "fark yorumlayicidan mi yoksa kare basina NumPy tahsisinden mi
    geliyor" sorusunu olcerek ayirmak icin var.
    """

    def __init__(self, preallocate=False):
        # detectShadows=False: C++ tarafi da golge tespitini kapatiyor
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=False,
        )
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.frames_since_reset = 0
        self.detections = []

        self.preallocate = preallocate
        # Yeniden kullanilan tamponlar. preallocate kapaliyken hep None kalir,
        # yani OpenCV her cagrida yeni dizi ayirir.
        self._gray = None
        self._blurred = None
        self._edges = None
        self._mask = None
        self._output = None

    def _dst(self, buffer):
        """preallocate acikken tamponu, kapaliyken None doner."""
        return buffer if self.preallocate else None

    def process(self, frame):
        """Tek kareyi isler. (output_bgr, stats) doner. Sureler milisaniye."""
        t_start = time.perf_counter()
        self.detections = []

        work = frame

        # ---- on isleme ----
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

        # ---- tespit ----
        if USE_MOTION_DETECT:
            self._mask = self.bg_sub.apply(work, self._dst(self._mask))
            mask = self._mask

            if self.frames_since_reset >= PIPELINE_WARMUP_FRAMES:
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN, self.morph_kernel, dst=self._dst(mask)
                )
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                        continue
                    self.detections.append(cv2.boundingRect(contour))

            self.frames_since_reset += 1

        t_detect = time.perf_counter()

        # ---- cikti goruntusu ----
        if edges is not None:
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
            "detections": len(self.detections),
        }
        return output, stats


def percentile(values, p):
    """C++ tarafindaki lineer interpolasyonlu yuzdelikle ayni hesap."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (p / 100.0) * (len(ordered) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def run_bench(video_path, max_frames, warmup_frames, json_out, preallocate=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"error: cannot open video: {video_path}", file=sys.stderr)
        return 1

    pipeline = Pipeline(preallocate=preallocate)
    total_ms, pre_ms, det_ms = [], [], []
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
    args = parser.parse_args()

    return run_bench(args.bench, args.frames, args.warmup, args.json_out,
                     preallocate=args.preallocate)


if __name__ == "__main__":
    raise SystemExit(main())
