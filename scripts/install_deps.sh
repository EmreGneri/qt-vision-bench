#!/usr/bin/env bash
# Installs the project's dependencies into an MSYS2 UCRT64 environment.
#
# Usage (from PowerShell):
#   C:\msys64\usr\bin\bash.exe -lc "/c/Users/Emre/projects/qt-vision-bench/scripts/install_deps.sh"
#
# The mirrors reset the connection now and then; pacman caches what it already
# downloaded, so retrying picks up where it stopped.

set -u

PACKAGES=(
  mingw-w64-ucrt-x86_64-toolchain
  mingw-w64-ucrt-x86_64-cmake
  mingw-w64-ucrt-x86_64-ninja
  mingw-w64-ucrt-x86_64-qt6-base
  mingw-w64-ucrt-x86_64-qt6-tools
  mingw-w64-ucrt-x86_64-opencv
)

MAX_ATTEMPTS=10

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== ATTEMPT $attempt / $MAX_ATTEMPTS ==="

  # Clear a lock file left behind by an earlier run, but only when no pacman
  # process is actually holding it.
  if [ -f /var/lib/pacman/db.lck ] && ! pgrep -x pacman >/dev/null 2>&1; then
    echo "removing stale db.lck"
    rm -f /var/lib/pacman/db.lck
  fi

  if pacman -S --needed --noconfirm --disable-download-timeout "${PACKAGES[@]}"; then
    echo "INSTALL OK"
    exit 0
  fi

  echo "attempt failed, retrying in 10 s"
  sleep 10
done

echo "INSTALL FAILED: gave up after $MAX_ATTEMPTS attempts"
exit 1
