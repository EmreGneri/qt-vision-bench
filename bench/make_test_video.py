"""Generates a deterministic synthetic test video.

Why synthetic: the comparison has to be reproducible. The seed is fixed so that
everyone who clones the repository gets the same frames, the same motion and the
same noise. A real camera recording would make the numbers drift from machine to
machine for no benefit.

Usage (PowerShell):
    py bench/make_test_video.py
    py bench/make_test_video.py --frames 600 --width 1280 --height 720
"""

import argparse
import os
import sys

import cv2
import numpy as np

# ==== SETTINGS ====
DEFAULT_OUTPUT = os.path.join("bench", "test_video.mp4")
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FRAMES = 300
DEFAULT_FPS = 30
RANDOM_SEED = 42          # fixed for reproducibility
NOISE_SIGMA = 4.0         # per-frame sensor noise, to give MOG2 something to do


def build_background(width, height, rng):
    """Static background: a vertical gradient plus an unchanging texture.

    A flat grey background would be too easy for MOG2; the texture brings it
    closer to the variance of a real scene.
    """
    gradient = np.linspace(40, 140, height, dtype=np.float32).reshape(height, 1)
    background = np.repeat(gradient, width, axis=1)

    # Fixed texture: identical in every frame, so it is part of the background
    texture = rng.normal(0.0, 8.0, size=(height, width)).astype(np.float32)
    background = background + texture

    background = np.clip(background, 0, 255).astype(np.uint8)
    return cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser(description="Generate a deterministic test video")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_SEED)
    background = build_background(args.width, args.height, rng)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        print(
            f"error: cannot open the video writer: {args.output}\n"
            "The mp4v codec may be missing; try an .avi path with --output.",
            file=sys.stderr,
        )
        return 1

    # Moving objects: (first frame, x0, y0, width, height, dx, dy, colour)
    objects = [
        (0, 20, 60, 90, 70, 3.0, 0.8, (255, 255, 255)),
        (0, 400, 300, 60, 60, -2.0, -1.4, (200, 200, 200)),
        (90, 300, 40, 120, 90, 1.2, 2.2, (240, 240, 240)),  # enters later
    ]

    for frame_index in range(args.frames):
        frame = background.copy()

        for start, x0, y0, w, h, dx, dy, color in objects:
            if frame_index < start:
                continue
            step = frame_index - start
            # Triangle wave so objects bounce off the edges and stay in frame
            x = int(x0 + dx * step) % (2 * max(args.width - w, 1))
            y = int(y0 + dy * step) % (2 * max(args.height - h, 1))
            if x > args.width - w:
                x = 2 * (args.width - w) - x
            if y > args.height - h:
                y = 2 * (args.height - h) - y
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, cv2.FILLED)

        # Per-frame noise, so the background model has a variance to learn
        noise = rng.normal(0.0, NOISE_SIGMA, size=frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        writer.write(frame)

        if (frame_index + 1) % 50 == 0:
            print(f"{frame_index + 1}/{args.frames} frames written")

    writer.release()

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"OK: {args.output}  ({args.frames} frames, {size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
