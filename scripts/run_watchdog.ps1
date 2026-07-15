<#
.SYNOPSIS
  로컬 Watchdog을 실행한다. Watchdog 자체의 재기동은 Windows 작업 스케줄러가 담당한다.
#>
param(
    [string]$ProjectRoot = "",
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) {
    $ProjectRoot = Join-Path $PSScriptRoot ".."
}
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
if (-not $PythonPath) {
    $PythonPath = Join-Path $resolvedRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Watchdog Python 실행 파일을 찾을 수 없다."
}

$env:PYTHONUTF8 = "1"
Push-Location -LiteralPath $resolvedRoot
try {
    & $PythonPath -m gateway.watchdog
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
