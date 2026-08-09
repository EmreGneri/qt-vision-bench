# Launches the application from plain PowerShell.
#
# Why this script is needed: the binary links Qt6 and OpenCV dynamically. Those
# DLLs live in the MSYS2 directory, which is not on the system PATH. Running from
# an MSYS2 shell the PATH is already right; running from outside it is not.
#
# Usage:
#   .\scripts\run.ps1
#   .\scripts\run.ps1 -Video bench\test_video.mp4
#
# Set QVB_DLL_DIR if MSYS2 is installed somewhere other than C:\msys64.

param(
    [string]$Video = ''
)

$dllDir = if ($env:QVB_DLL_DIR) { $env:QVB_DLL_DIR } else { 'C:\msys64\ucrt64\bin' }

$projectRoot = Split-Path $PSScriptRoot -Parent
$exePath = Join-Path $projectRoot 'build\qt_vision_bench.exe'

if (-not (Test-Path $exePath)) {
    Write-Error "Binary not found: $exePath`nBuild it first: ./scripts/build.sh in an MSYS2 UCRT64 shell"
    exit 1
}

if (-not (Test-Path $dllDir)) {
    Write-Error "DLL directory not found: $dllDir`nSet QVB_DLL_DIR to the folder holding the Qt/OpenCV DLLs."
    exit 1
}

$env:PATH = $dllDir + ';' + $env:PATH

if ($Video -ne '') {
    & $exePath --video $Video
} else {
    & $exePath
}
