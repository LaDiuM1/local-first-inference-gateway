<#
.SYNOPSIS
  6단계 Windows 작업의 소유권 판정과 시작·재시작 계획을 정한다.

.DESCRIPTION
  Windows 상태를 읽거나 바꾸지 않는 판정만 담는다 — 조회와 실제 등록·시작은 install_stage6.ps1이
  맡는다. 덕분에 실제 작업을 만들지 않고 소유권 경계와 수명주기 결정을 그대로 확인할 수 있다.
#>

$ErrorActionPreference = "Stop"

function Test-OwnedTask {
    <#
    .SYNOPSIS
      기존 작업이 이 저장소가 등록한 작업인지 판정한다 — 이름만 같은 남의 작업은 덮어쓰지 않는다.
    #>
    param(
        [Parameter(Mandatory = $true)]$Task,
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][string]$Execute,
        [Parameter(Mandatory = $true)][string]$Arguments
    )
    if ($Task.Description -ne $Description) {
        return $false
    }
    $actions = @($Task.Actions)
    if ($actions.Count -ne 1) {
        return $false
    }
    $sameExecutable = [string]::Equals(
        $actions[0].Execute, $Execute, [StringComparison]::OrdinalIgnoreCase
    )
    if (-not $sameExecutable) {
        return $false
    }
    return $actions[0].Arguments -eq $Arguments
}

function Get-TaskStopPlan {
    <#
    .SYNOPSIS
      배포 사본을 교체하기 전에 중지할 작업을 순서대로 정한다.

    .DESCRIPTION
      네 작업 모두 배포 사본을 작업 디렉터리로 쓰므로, 실행 중인 작업이 하나라도 남으면 사본을 옮기지
      못한다. Watchdog을 가장 먼저 중지한다 — 나중에 세우면 정비 중에 멈춘 작업을 도로 복구해 교체를
      막는다. 이미 멈춰 있는 작업은 건드리지 않는다.
    #>
    param(
        [Parameter(Mandatory = $true)][string[]]$Names,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$RunningNames,
        [Parameter(Mandatory = $true)][string]$WatchdogName
    )
    $ordered = @()
    if ($RunningNames -contains $WatchdogName) {
        $ordered += $WatchdogName
    }
    foreach ($name in $Names) {
        if ($name -eq $WatchdogName) {
            continue
        }
        if ($RunningNames -contains $name) {
            $ordered += $name
        }
    }
    return , $ordered
}

function Get-TaskStartPlan {
    <#
    .SYNOPSIS
      교체를 마친 뒤 시작할 작업을 등록 순서대로 정한다.

    .DESCRIPTION
      교체 전에 이 저장소가 소유한 실행 중 작업은 모두 중지했으므로, 여기서 시작하는 작업은 새 배포
      사본과 새로 쓴 보호 키를 읽는 새 인스턴스가 된다. StartTasks면 전부 세우고, 아니면 작업 전에
      실행 중이던 것만 되돌린다 — 관리자가 일부러 멈춰 둔 작업을 임의로 시작하지 않는다.
    #>
    param(
        [Parameter(Mandatory = $true)][string[]]$Names,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$RunningBefore,
        [switch]$StartTasks
    )
    $ordered = @()
    foreach ($name in $Names) {
        if ($StartTasks -or ($RunningBefore -contains $name)) {
            $ordered += $name
        }
    }
    return , $ordered
}
