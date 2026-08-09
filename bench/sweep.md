Median of 5 runs per cell, 150 frames per run; the range in parentheses is min-max across runs.

| Resolution | Pixels | C++ (ms) | Python (ms) | Python prealloc (ms) | C++ vs Python | C++ vs prealloc | Parity |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 480p | 307,200 | 1.312 (1.295-1.398) | 1.511 (1.491-1.568) | 1.479 (1.438-1.501) | 1.15x | 1.13x | ok |
| 720p | 921,600 | 3.650 (3.630-3.791) | 4.464 (4.450-4.535) | 3.977 (3.902-4.041) | 1.22x | 1.09x | ok |
| 1080p | 2,073,600 | 10.121 (10.038-10.358) | 10.991 (10.755-11.072) | 9.546 (9.479-9.701) | 1.09x | 0.94x | ok |

Per-stage medians (ms):

| Resolution | Stage | C++ | Python | Python prealloc |
|---|---|---:|---:|---:|
| 480p | preprocess | 0.107 | 0.179 | 0.160 |
| 480p | detection | 1.155 | 1.268 | 1.261 |
| 720p | preprocess | 0.265 | 0.363 | 0.273 |
| 720p | detection | 3.275 | 3.736 | 3.588 |
| 1080p | preprocess | 0.509 | 1.337 | 0.522 |
| 1080p | detection | 9.432 | 9.143 | 8.779 |
