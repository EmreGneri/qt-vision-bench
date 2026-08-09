# Uygulamayi duz PowerShell'den baslatir.
#
# Neden bu betik gerekli: ikili, Qt6 ve OpenCV DLL'lerine dinamik bagli. Bu
# DLL'ler MSYS2'nin klasorunde duruyor ve sistem PATH'inde yok. MSYS2 kabugundan
# calistirirken PATH zaten dogru, disaridan calistirirken degil.
#
# Kullanim:
#   .\scripts\run.ps1
#   .\scripts\run.ps1 -Video bench\test_video.mp4

param(
    [string]$Video = ''
)

# ==== AYARLAR (buradan degistir) ====
$MSYS2_BIN = 'C:\msys64\ucrt64\bin'   # Qt/OpenCV DLL'lerinin bulundugu klasor

$projectRoot = Split-Path $PSScriptRoot -Parent
$exePath = Join-Path $projectRoot 'build\qt_vision_bench.exe'

if (-not (Test-Path $exePath)) {
    Write-Error "Ikili bulunamadi: $exePath`nOnce derleyin: MSYS2 UCRT64 kabugunda ./scripts/build.sh"
    exit 1
}

if (-not (Test-Path $MSYS2_BIN)) {
    Write-Error "MSYS2 klasoru yok: $MSYS2_BIN`nBetigin basindaki MSYS2_BIN degerini kendi kurulumunuza gore duzeltin."
    exit 1
}

$env:PATH = $MSYS2_BIN + ';' + $env:PATH

if ($Video -ne '') {
    & $exePath --video $Video
} else {
    & $exePath
}
