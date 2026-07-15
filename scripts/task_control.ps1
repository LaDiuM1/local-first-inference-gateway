<#
.SYNOPSIS
  정해진 Windows 작업 또는 cloudflared 서비스의 로컬 상태를 조회하고 개별 제어한다.

.DESCRIPTION
  Watchdog은 State와 Restart를, 설치기는 배포 사본을 교체하기 전에 Stop을 쓴다. 중지 절차를 한
  곳에 두어 두 경로가 같은 방식으로 멈추고 같은 제한 시간을 기다리게 한다.
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("State", "Stop", "Restart")]
    [string]$Action,
    [Parameter(Mandatory = $true)]
    [ValidateSet("Task", "Service")]
    [string]$Kind,
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [string]$TaskPath = "\LocalFirstInferenceGateway\"
)

$ErrorActionPreference = "Stop"

if ($Kind -eq "Service") {
    if ($Action -eq "Stop") {
        # cloudflared 공식 서비스는 배포 사본을 쓰지 않는다 — 교체를 위해 중지할 대상이 아니다.
        throw "서비스 중지는 이 스크립트가 다루지 않는다."
    }
    $service = Get-Service -Name $Name -ErrorAction SilentlyContinue
    if (-not $service) {
        if ($Action -eq "State") {
            Write-Output "Missing"
            exit 0
        }
        throw "관리 대상 서비스를 찾을 수 없다."
    }
    $escapedName = $Name.Replace("'", "''")
    $serviceInfo = Get-CimInstance Win32_Service -Filter "Name='$escapedName'"
    if ($Action -eq "State") {
        if ($serviceInfo.StartMode -eq "Disabled") {
            Write-Output "Disabled"
        }
        elseif ($service.Status -eq "Running") {
            Write-Output "Running"
        }
        else {
            Write-Output "Ready"
        }
        exit 0
    }
    if ($serviceInfo.StartMode -eq "Disabled") {
        throw "관리 대상 서비스가 비활성화되어 있다."
    }
    if ($service.Status -ne "Running") {
        Start-Service -Name $Name
    }
    exit 0
}

$tasks = @(Get-ScheduledTask -TaskPath $TaskPath -TaskName $Name -ErrorAction SilentlyContinue)
$task = $tasks |
    Where-Object { $_.TaskName -eq $Name -and $_.TaskPath -eq $TaskPath } |
    Select-Object -First 1
if (-not $task) {
    if ($Action -eq "State") {
        Write-Output "Missing"
        exit 0
    }
    throw "관리 대상 작업을 찾을 수 없다."
}
if ($Action -eq "State") {
    Write-Output $task.State.ToString()
    exit 0
}
if ($Action -eq "Restart" -and $task.State -eq "Disabled") {
    throw "관리 대상 작업이 비활성화되어 있다."
}
if ($task.State -eq "Running") {
    Stop-ScheduledTask -InputObject $task
    # 중지 요청은 즉시 돌아온다 — 실제로 멈춘 것을 확인해야 배포 사본을 잡은 프로세스가 없다.
    $stopDeadline = (Get-Date).AddSeconds(15)
    do {
        Start-Sleep -Milliseconds 200
        $task = Get-ScheduledTask -TaskPath $TaskPath -TaskName $Name
    } while ($task.State -eq "Running" -and (Get-Date) -lt $stopDeadline)
    if ($task.State -eq "Running") {
        throw "관리 대상 작업이 제한 시간 안에 중지되지 않았다."
    }
}
if ($Action -eq "Stop") {
    exit 0
}
Start-ScheduledTask -TaskPath $TaskPath -TaskName $Name
