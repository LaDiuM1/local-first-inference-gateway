# run_monitor.ps1 — 운영 모니터를 온디맨드로 실행한다.
#
# 운영 관측 로그(%ProgramData%\local-first-inference-gateway\logs)는 관리자만 읽을 수 있으므로
# 요청 지표까지 보려면 관리자 PowerShell에서 실행한다. 비관리자로 실행하면 실시간
# 프로브(구성요소 상태·GPU·작업)만 표시된다. 모니터는 127.0.0.1에만 바인딩되며
# Cloudflare 터널이나 공개 호스트에 등록하지 않는다. 종료는 Ctrl+C — 상주하지 않는다.

param(
    [int]$Port = 29100
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "가상환경 Python이 없다: $Python — 프로젝트 루트에서 uv sync를 먼저 실행한다."
}

$identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Warning "비관리자 셸 — 운영 관측 로그를 읽지 못해 요청 지표가 비어 보일 수 있다."
}

Start-Process "http://127.0.0.1:$Port/"
& $Python -m gateway.monitor --port $Port
