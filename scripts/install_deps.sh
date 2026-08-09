#!/usr/bin/env bash
# MSYS2 UCRT64 ortamina projenin bagimliliklarini kurar.
#
# Kullanim (PowerShell'den):
#   C:\msys64\usr\bin\bash.exe -lc "/c/Users/Emre/projects/qt-vision-bench/scripts/install_deps.sh"
#
# Aynalar zaman zaman baglantiyi resetliyor; pacman indirdiklerini onbellekte
# tuttugu icin tekrar denemek kaldigi yerden devam ediyor.

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
  echo "=== DENEME $attempt / $MAX_ATTEMPTS ==="

  # Onceki calismadan kalan kilit dosyasi varsa ve pacman calismiyorsa temizle
  if [ -f /var/lib/pacman/db.lck ] && ! pgrep -x pacman >/dev/null 2>&1; then
    echo "kalinti db.lck siliniyor"
    rm -f /var/lib/pacman/db.lck
  fi

  if pacman -S --needed --noconfirm --disable-download-timeout "${PACKAGES[@]}"; then
    echo "KURULUM BASARILI"
    exit 0
  fi

  echo "deneme basarisiz, 10 sn sonra tekrar"
  sleep 10
done

echo "KURULUM BASARISIZ: $MAX_ATTEMPTS deneme sonunda tamamlanamadi"
exit 1
