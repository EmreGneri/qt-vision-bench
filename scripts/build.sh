#!/usr/bin/env bash
# Configure + build + run the tests.
#
# Usage (in an MSYS2 UCRT64 shell):
#   ./scripts/build.sh              # Release
#   ./scripts/build.sh Debug        # Debug
#
# From PowerShell:
#   C:\msys64\usr\bin\bash.exe -lc "/c/Users/Emre/projects/qt-vision-bench/scripts/build.sh"

set -euo pipefail

BUILD_TYPE="${1:-Release}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

cd "$PROJECT_DIR"

echo "=== CMake configure ($BUILD_TYPE) ==="
cmake -G Ninja -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo
echo "=== Build ==="
cmake --build "$BUILD_DIR"

echo
echo "=== Tests ==="
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo
if [ "$BUILD_TYPE" != "Release" ]; then
    echo "NOTE: this is a $BUILD_TYPE build; the benchmark scripts only accept Release."
fi
echo "DONE: $BUILD_DIR/qt_vision_bench.exe"
