<#
.SYNOPSIS
  범용 추론 Ollama를 loopback에서 독립 실행하고 종료 시 backoff로 재기동한다.
#>
param(
    [int]$Port = 11434,
    [string]$OllamaPath = "",
    [string]$ModelsPath = "",
    [string[]]$ServeCommand = @(),
    [int]$MaxRestarts = 0,
    [double]$BackoffSeconds = 2,
    [double]$MaxBackoffSeconds = 30
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "supervisor_common.ps1")

function Resolve-OllamaPath {
    if ($OllamaPath) {
        if (-not (Test-Path -LiteralPath $OllamaPath -PathType Leaf)) {
            throw "지정한 ollama 실행 파일이 없다."
        }
        return (Resolve-Path -LiteralPath $OllamaPath).Path
    }
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
        "C:\Program Files\Ollama\ollama.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    throw "ollama 실행 파일을 찾을 수 없다."
}

$command = $ServeCommand
if ($command.Count -eq 0) {
    $command = @((Resolve-OllamaPath), "serve")
}
if ($ModelsPath) {
    if (-not (Test-Path -LiteralPath $ModelsPath -PathType Container)) {
        throw "지정한 Ollama 모델 저장소가 없다."
    }
    $env:OLLAMA_MODELS = (Resolve-Path -LiteralPath $ModelsPath).Path
}
$env:OLLAMA_HOST = "127.0.0.1:$Port"
$env:OLLAMA_CONTEXT_LENGTH = "8192"
$env:OLLAMA_KEEP_ALIVE = "2h"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_NO_CLOUD = "1"
Remove-Item Env:OLLAMA_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue

$arguments = @()
if ($command.Count -gt 1) {
    $arguments = $command[1..($command.Count - 1)]
}
$supervisor = @{
    Name              = "chat-ollama"
    Executable        = $command[0]
    ArgumentList      = $arguments
    MaxRestarts       = $MaxRestarts
    BackoffSeconds    = $BackoffSeconds
    MaxBackoffSeconds = $MaxBackoffSeconds
}
Invoke-SupervisedProcess @supervisor
