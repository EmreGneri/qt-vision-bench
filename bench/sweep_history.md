Median of 5 runs per cell, 150 frames per run; the range in parentheses is min-max across runs.

| Resolution | Pixels | C++ (ms) | Python (ms) | Python prealloc (ms) | C++ vs Python | C++ vs prealloc | Parity |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 480p | 307,200 | 1.650 (1.587-1.713) | 2.752 (2.617-2.978) | 2.337 (2.216-3.022) | 1.67x | 1.42x | ok |
| 720p | 921,600 | 4.904 (4.871-5.064) | 8.079 (7.839-8.222) | 7.439 (7.348-7.643) | 1.65x | 1.52x | ok |
| 1080p | 2,073,600 | 12.779 (12.755-12.870) | 20.602 (17.783-20.893) | 16.559 (16.373-17.035) | 1.61x | 1.30x | ok |

Parity is gated, not annotated: the harness exits non-zero if any cell disagrees.

| Resolution | Motion-history images (C++ vs Python) |
|---|---|
| 480p | identical - identical over 307,200 pixels; identical over 307,200 pixels |
| 720p | identical - identical over 921,600 pixels; identical over 921,600 pixels |
| 1080p | identical - identical over 2,073,600 pixels; identical over 2,073,600 pixels |

Per-stage medians (ms):

| Resolution | Stage | C++ | Python | Python prealloc |
|---|---|---:|---:|---:|
| 480p | preprocess | 0.130 | 0.213 | 0.153 |
| 480p | detection | 1.326 | 1.546 | 1.322 |
| 480p | motion history | 0.122 | 0.757 | 0.779 |
| 720p | preprocess | 0.299 | 0.441 | 0.322 |
| 720p | detection | 4.122 | 4.419 | 4.330 |
| 720p | motion history | 0.355 | 2.740 | 2.666 |
| 1080p | preprocess | 0.618 | 1.531 | 0.595 |
| 1080p | detection | 11.188 | 11.455 | 10.426 |
| 1080p | motion history | 0.803 | 7.003 | 5.233 |

Environment:

| | C++ side | Python side |
|---|---|---|
| OpenCV | 5.0.0 | 4.13.0 |
| Toolchain | gcc 16.1.0, Release | Python 3.14.4, NumPy 2.4.3 |
