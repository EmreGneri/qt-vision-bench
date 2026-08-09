Median of 5 runs per cell, 150 frames per run; the range in parentheses is min-max across runs.

| Resolution | Pixels | C++ (ms) | Python (ms) | Python prealloc (ms) | C++ vs Python | C++ vs prealloc | Parity |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 480p | 307,200 | 1.324 (1.304-1.352) | 1.561 (1.492-1.616) | 1.484 (1.441-1.522) | 1.18x | 1.12x | ok |
| 720p | 921,600 | 3.788 (3.618-3.848) | 4.526 (4.469-4.597) | 3.953 (3.944-4.039) | 1.19x | 1.04x | ok |
| 1080p | 2,073,600 | 10.162 (10.107-10.290) | 10.915 (10.891-11.101) | 9.605 (9.548-9.636) | 1.07x | 0.95x | ok |

Per-stage medians (ms):

| Resolution | Stage | C++ | Python | Python prealloc |
|---|---|---:|---:|---:|
| 480p | preprocess | 0.144 | 0.194 | 0.161 |
| 480p | detection | 1.119 | 1.300 | 1.268 |
| 720p | preprocess | 0.260 | 0.371 | 0.270 |
| 720p | detection | 3.420 | 3.753 | 3.571 |
| 1080p | preprocess | 0.517 | 1.308 | 0.503 |
| 1080p | detection | 9.472 | 9.160 | 8.853 |
