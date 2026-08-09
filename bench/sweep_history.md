Median of 5 runs per cell, 150 frames per run; the range in parentheses is min-max across runs.

| Resolution | Pixels | C++ (ms) | Python (ms) | Python prealloc (ms) | C++ vs Python | C++ vs prealloc | Parity |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 480p | 307,200 | 1.441 (1.397-1.710) | 2.326 (2.259-2.421) | 2.189 (2.172-2.331) | 1.61x | 1.52x | ok |
| 720p | 921,600 | 4.050 (4.041-4.124) | 7.149 (7.088-7.410) | 6.660 (6.442-6.754) | 1.76x | 1.64x | ok |
| 1080p | 2,073,600 | 11.028 (10.949-11.153) | 16.134 (16.025-16.288) | 14.365 (14.327-14.445) | 1.46x | 1.30x | ok |

Per-stage medians (ms):

| Resolution | Stage | C++ | Python | Python prealloc |
|---|---|---:|---:|---:|
| 480p | preprocess | 0.124 | 0.192 | 0.147 |
| 480p | detection | 1.147 | 1.306 | 1.231 |
| 480p | motion history | 0.120 | 0.646 | 0.732 |
| 720p | preprocess | 0.259 | 0.390 | 0.300 |
| 720p | detection | 3.326 | 3.901 | 3.721 |
| 720p | motion history | 0.321 | 2.455 | 2.396 |
| 1080p | preprocess | 0.533 | 1.350 | 0.520 |
| 1080p | detection | 9.513 | 9.144 | 8.884 |
| 1080p | motion history | 0.790 | 5.114 | 4.675 |
