Median of 5 runs per cell, 150 frames per run; the range in parentheses is min-max across runs.

| Resolution | Pixels | C++ (ms) | Python (ms) | Python prealloc (ms) | C++ vs Python | C++ vs prealloc | Parity |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 480p | 307,200 | 1.470 (1.424-1.524) | 1.594 (1.586-1.736) | 1.563 (1.493-1.586) | 1.08x | 1.06x | ok |
| 720p | 921,600 | 4.534 (4.428-4.577) | 5.184 (5.136-5.352) | 4.624 (4.556-4.818) | 1.14x | 1.02x | ok |
| 1080p | 2,073,600 | 11.728 (11.604-12.429) | 12.699 (12.530-12.880) | 10.887 (10.492-11.022) | 1.08x | 0.93x | ok |

Parity is gated, not annotated: the harness exits non-zero if any cell disagrees.

Per-stage medians (ms):

| Resolution | Stage | C++ | Python | Python prealloc |
|---|---|---:|---:|---:|
| 480p | preprocess | 0.127 | 0.145 | 0.149 |
| 480p | detection | 1.286 | 1.397 | 1.351 |
| 720p | preprocess | 0.290 | 0.323 | 0.285 |
| 720p | detection | 4.127 | 4.415 | 4.207 |
| 1080p | preprocess | 0.587 | 1.505 | 0.548 |
| 1080p | detection | 10.930 | 10.681 | 10.073 |

Environment:

| | C++ side | Python side |
|---|---|---|
| OpenCV | 5.0.0 | 4.13.0 |
| Toolchain | gcc 16.1.0, Release | Python 3.14.4, NumPy 2.4.3 |
