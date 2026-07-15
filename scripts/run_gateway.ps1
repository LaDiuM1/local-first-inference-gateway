<#
.SYNOPSIS
  FastAPI 게이트웨이를 loopback에서 독립 실행하고 종료 시 backoff로 재기동한다.
#>
param(
    [int]$Port = 8000,
    [string]$ProjectRoot = "",
    [string]$PythonPath = "",
    [string[]]$GatewayCommand = @(),
    [int]$MaxRestarts = 0,
    [double]$BackoffSeconds = 2,
    [double]$MaxBackoffSeconds = 30
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "supervisor_common.ps1")

if (-not $ProjectRoot) {
    $ProjectRoot = Join-Path $PSScriptRoot ".."
}
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
if (-not $PythonPath) {
    $PythonPath = Join-Path $resolvedRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "게이트웨이 Python 실행 파일을 찾을 수 없다."
}

$command = $GatewayCommand
if ($command.Count -eq 0) {
    $command = @(
        (Resolve-Path -LiteralPath $PythonPath).Path,
        "-m",
        "uvicorn",
        "gateway.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "$Port",
        "--log-level",
        "info"
    )
}
$arguments = @()
if ($command.Count -gt 1) {
    $arguments = $command[1..($command.Count - 1)]
}

$env:PYTHONUTF8 = "1"
$supervisor = @{
    Name              = "gateway"
    Executable        = $command[0]
    ArgumentList      = $arguments
    MaxRestarts       = $MaxRestarts
    BackoffSeconds    = $BackoffSeconds
    MaxBackoffSeconds = $MaxBackoffSeconds
}
Push-Location -LiteralPath $resolvedRoot
try {
    Invoke-SupervisedProcess @supervisor
}
finally {
    Pop-Location
}
