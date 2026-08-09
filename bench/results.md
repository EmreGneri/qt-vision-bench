640x480, 290 measured frames, library-call pipeline only.

| Metric | C++ / Qt | Python | C++ advantage |
|---|---:|---:|---:|
| Mean frame time (ms) | 1.434 | 1.561 | 1.09x |
| Median frame time (ms) | 1.414 | 1.532 | 1.08x |
| p95 frame time (ms) | 1.750 | 1.929 | 1.10x |
| Preprocess mean (ms) | 0.121 | 0.140 | 1.16x |
| Detection mean (ms) | 1.251 | 1.355 | 1.08x |
| Processing throughput (fps) | 697.3 | 640.4 | 1.09x |
| End-to-end throughput (fps) | 568.6 | 473.6 | 1.20x |
| Detections found | 680 | 680 | parity check |
