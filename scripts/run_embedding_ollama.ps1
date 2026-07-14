<#
.SYNOPSIS
  임베딩 전용 CPU Ollama 인스턴스를 loopback에서 실행하고, 종료되면 같은 환경·포트로 자동 재기동한다.

.DESCRIPTION
  chat·vision용 Ollama와 분리된 임베딩 전용 인스턴스를 127.0.0.1:$Port(기본 11435)에 띄운다.
  GPU를 숨겨 CPU 전용으로 두고, 단일 모델·요청에 맞춘 보수적 환경을 설정하며, cloud 기능을 끄고
  loopback에만 바인딩한다. 모델 저장소는 기존 Ollama 저장소를 그대로 쓴다.

  자식이 종료되면 제한된 backoff 뒤 재기동한다. 반복 실패 시 backoff가 상한까지 커져 tight loop를
  막고, 오래 정상 구동하다 종료한 경우에는 backoff를 초기화해 즉시 재기동한다. 감독이 어떤 이유로
  종료되든(정상·Ctrl+C·강제 종료) 자식이 고아로 남지 않도록 자식을 kill-on-close 잡 오브젝트에 넣어
  OS가 정리하게 하고, 정상 종료 경로에서는 finally에서도 살아 있는 자식을 정리한다.

  부팅 시 자동 시작 등록은 이 스크립트 범위 밖이다(6단계 전체 스택 자동 복구).

.NOTES
  -ServeCommand는 기본적으로 `ollama serve`를 자동 구성한다. 테스트는 실제 Ollama 대신 짧게
  끝나는 가짜 자식을 주입해 시작·재기동·backoff·정리 경로를 결정적으로 검증한다.
#>
param(
    [int]$Port = 11435,
    [string]$OllamaPath = "",
    [string[]]$ServeCommand = @(),
    [int]$MaxRestarts = 0,
    [double]$BackoffSeconds = 2,
    [double]$MaxBackoffSeconds = 30
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $stamp = (Get-Date).ToString("HH:mm:ss")
    Write-Host "[embed-ollama $stamp] $Message"
}

function Resolve-OllamaPath {
    if ($OllamaPath) {
        if (-not (Test-Path -LiteralPath $OllamaPath)) {
            throw "지정한 ollama 경로가 없다: $OllamaPath"
        }
        return $OllamaPath
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
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    throw "ollama 실행 파일을 찾을 수 없다. -OllamaPath로 지정하라."
}

function Get-ServeCommand {
    if ($ServeCommand.Count -gt 0) {
        return $ServeCommand
    }
    return @((Resolve-OllamaPath), "serve")
}

function New-KillOnCloseJob {
    # 감독이 어떤 이유로 종료되든(정상·Ctrl+C·강제 종료) 자식이 고아로 남지 않도록 kill-on-close 잡을
    # 만든다. 감독이 쥔 이 잡 핸들이 프로세스 종료로 닫히면 OS가 잡에 든 자식을 정리한다.
    if (-not ('Win32Job' -as [type])) {
        Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class Win32Job {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    public static extern IntPtr CreateJobObject(IntPtr a, string lpName);
    [DllImport("kernel32.dll")]
    public static extern bool SetInformationJobObject(IntPtr hJob, int infoClass, IntPtr lpInfo, uint cbInfoLength);
    [DllImport("kernel32.dll")]
    public static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);
}
'@
    }
    $job = [Win32Job]::CreateJobObject([IntPtr]::Zero, $null)
    if ($job -eq [IntPtr]::Zero) {
        throw "잡 오브젝트를 만들 수 없다."
    }
    # JOBOBJECT_EXTENDED_LIMIT_INFORMATION(x64 144바이트)의 LimitFlags(오프셋 16)에
    # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE(0x2000)만 세운다 — 다른 한도는 두지 않는다.
    $size = 144
    $info = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size)
    try {
        for ($offset = 0; $offset -lt $size; $offset += 4) {
            [System.Runtime.InteropServices.Marshal]::WriteInt32($info, $offset, 0)
        }
        [System.Runtime.InteropServices.Marshal]::WriteInt32($info, 16, 0x2000)
        $configured = [Win32Job]::SetInformationJobObject($job, 9, $info, $size)
        if (-not $configured) {
            throw "잡 오브젝트에 kill-on-close 설정을 적용하지 못했다."
        }
    }
    finally {
        [System.Runtime.InteropServices.Marshal]::FreeHGlobal($info)
    }
    return $job
}

# 임베딩 전용 인스턴스 환경 — GPU 숨김(CPU 전용), loopback 바인딩, 단일 모델·요청 보수 설정.
# cloud 기능은 Ollama가 문서화한 OLLAMA_NO_CLOUD로 끄고, loopback 바인딩과 cloud 자격 제거를 더한다.
# 모델 저장소는 기존 저장소를 그대로 쓴다.
$env:OLLAMA_HOST = "127.0.0.1:$Port"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_NO_CLOUD = "1"
Remove-Item Env:OLLAMA_API_KEY -ErrorAction SilentlyContinue

$command = Get-ServeCommand
$executable = $command[0]
$arguments = @()
if ($command.Count -gt 1) {
    $arguments = $command[1..($command.Count - 1)]
}

Write-Log "임베딩 Ollama 감독 시작 — host 127.0.0.1:$Port, 명령 '$executable'"

$job = New-KillOnCloseJob
$child = $null
$delay = $BackoffSeconds
$attempt = 0
try {
    while ($MaxRestarts -le 0 -or $attempt -lt $MaxRestarts) {
        $attempt++
        Write-Log "임베딩 Ollama 시작 (attempt $attempt)"

        $startParams = @{
            FilePath    = $executable
            NoNewWindow = $true
            PassThru    = $true
        }
        if ($arguments.Count -gt 0) {
            $startParams.ArgumentList = $arguments
        }
        $child = Start-Process @startParams
        # 종료 후에도 ExitCode를 읽을 수 있도록 Handle을 먼저 캐시한다.
        $null = $child.Handle
        if (-not [Win32Job]::AssignProcessToJobObject($job, $child.Handle)) {
            # 잡에 넣지 못하면 강제 종료 경로(finally 미실행)에서 자식이 고아로 남을 수 있어
            # 모든 종료 경로의 정리 계약을 지킬 수 없다 — 방금 띄운 자식을 끝내고 감독을 실패시킨다.
            Stop-Process -Id $child.Id -Force -ErrorAction SilentlyContinue
            $child = $null
            throw "자식을 kill-on-close 잡에 넣지 못했다 — 고아 방지를 보장할 수 없어 감독을 중단한다."
        }
        $ranFor = [System.Diagnostics.Stopwatch]::StartNew()
        $child.WaitForExit()
        $ranFor.Stop()
        $ranSeconds = [int]$ranFor.Elapsed.TotalSeconds
        Write-Log "임베딩 Ollama 종료 (attempt $attempt, exit $($child.ExitCode), ${ranSeconds}s 실행)"
        $child = $null

        if ($MaxRestarts -gt 0 -and $attempt -ge $MaxRestarts) {
            Write-Log "재기동 한도($MaxRestarts) 도달 — 감독을 종료한다."
            break
        }

        if ($ranFor.Elapsed.TotalSeconds -ge $MaxBackoffSeconds) {
            # 충분히 오래 구동하다 종료했으면 즉시 재기동하고 backoff를 초기화한다.
            $delay = $BackoffSeconds
            Write-Log "정상 구동 후 종료 — 즉시 재기동, backoff 초기화"
        }
        else {
            Write-Log "재기동 backoff ${delay}s 대기"
            Start-Sleep -Seconds $delay
            $delay = [math]::Min($delay * 2, $MaxBackoffSeconds)
        }
    }
}
finally {
    if ($child -and -not $child.HasExited) {
        Write-Log "감독 종료 — 살아 있는 자식 정리 (PID $($child.Id))"
        Stop-Process -Id $child.Id -Force -ErrorAction SilentlyContinue
    }
}
