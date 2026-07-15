<#
.SYNOPSIS
  Windows 작업이 소유하는 자식 프로세스를 backoff와 kill-on-close 잡으로 감독하는 공용 함수다.
#>

$ErrorActionPreference = "Stop"

function New-KillOnCloseJob {
    if (-not ('OpenAtWin32Job' -as [type])) {
        Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class OpenAtWin32Job {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    public static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll")]
    public static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint length);
    [DllImport("kernel32.dll")]
    public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
}
'@
    }
    $job = [OpenAtWin32Job]::CreateJobObject([IntPtr]::Zero, $null)
    if ($job -eq [IntPtr]::Zero) {
        throw "잡 오브젝트를 만들 수 없다."
    }
    $size = 144
    $info = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size)
    try {
        for ($offset = 0; $offset -lt $size; $offset += 4) {
            [System.Runtime.InteropServices.Marshal]::WriteInt32($info, $offset, 0)
        }
        [System.Runtime.InteropServices.Marshal]::WriteInt32($info, 16, 0x2000)
        if (-not [OpenAtWin32Job]::SetInformationJobObject($job, 9, $info, $size)) {
            throw "잡 오브젝트에 kill-on-close 설정을 적용하지 못했다."
        }
    }
    finally {
        [System.Runtime.InteropServices.Marshal]::FreeHGlobal($info)
    }
    return $job
}

function Invoke-SupervisedProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [string[]]$ArgumentList = @(),
        [int]$MaxRestarts = 0,
        [double]$BackoffSeconds = 2,
        [double]$MaxBackoffSeconds = 30
    )

    $job = New-KillOnCloseJob
    $child = $null
    $delay = $BackoffSeconds
    $attempt = 0
    try {
        while ($MaxRestarts -le 0 -or $attempt -lt $MaxRestarts) {
            $attempt++
            Write-Host "[$Name] 시작 (attempt $attempt)"
            $startParams = @{
                FilePath    = $Executable
                NoNewWindow = $true
                PassThru    = $true
            }
            if ($ArgumentList.Count -gt 0) {
                $startParams.ArgumentList = $ArgumentList
            }
            $child = Start-Process @startParams
            $null = $child.Handle
            if (-not [OpenAtWin32Job]::AssignProcessToJobObject($job, $child.Handle)) {
                Stop-Process -Id $child.Id -Force -ErrorAction SilentlyContinue
                $child = $null
                throw "자식을 kill-on-close 잡에 넣지 못했다."
            }
            $ranFor = [System.Diagnostics.Stopwatch]::StartNew()
            $child.WaitForExit()
            $ranFor.Stop()
            Write-Host "[$Name] 종료 (exit $($child.ExitCode), $([int]$ranFor.Elapsed.TotalSeconds)s 실행)"
            $child = $null

            if ($MaxRestarts -gt 0 -and $attempt -ge $MaxRestarts) {
                break
            }
            if ($ranFor.Elapsed.TotalSeconds -ge $MaxBackoffSeconds) {
                $delay = $BackoffSeconds
                continue
            }
            Write-Host "[$Name] backoff ${delay}s"
            Start-Sleep -Seconds $delay
            $delay = [math]::Min($delay * 2, $MaxBackoffSeconds)
        }
    }
    finally {
        if ($child -and -not $child.HasExited) {
            Stop-Process -Id $child.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
