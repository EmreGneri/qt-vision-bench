#!/usr/bin/env bash
# Yapilandir + derle + testleri kos.
#
# Kullanim (MSYS2 UCRT64 kabugunda):
#   ./scripts/build.sh              # Release
#   ./scripts/build.sh Debug        # Debug
#
# PowerShell'den:
#   C:\msys64\usr\bin\bash.exe -lc "/c/Users/Emre/projects/qt-vision-bench/scripts/build.sh"

set -euo pipefail

BUILD_TYPE="${1:-Release}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

cd "$PROJECT_DIR"

echo "=== CMake yapilandirma ($BUILD_TYPE) ==="
cmake -G Ninja -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo
echo "=== Derleme ==="
cmake --build "$BUILD_DIR"

echo
echo "=== Testler ==="
cd "$BUILD_DIR"
ctest --output-on-failure

echo
echo "TAMAM: $BUILD_DIR/qt_vision_bench.exe"
