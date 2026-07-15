<#
.SYNOPSIS
  검증을 통과한 배포 사본만 SYSTEM 작업이 실행하는 자리에 놓는다.

.DESCRIPTION
  staging을 먼저 검증하고 소유 작업을 중지한 뒤 live를 교체한다. 이후 단계가 실패하면 직전 배포본과
  작업 실행 상태를 복구한다. 작업 상태 변경은 주입된 TaskControl 경계만 사용한다.
#>

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "stage6_tasks.ps1")

function Remove-DirectoryIfPresent {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Switch-Deployment {
    <# .SYNOPSIS 검증된 사본을 live로 옮기고 직전 배포본을 남긴다. #>
    param(
        [Parameter(Mandatory = $true)][string]$Staging,
        [Parameter(Mandatory = $true)][string]$Live,
        [Parameter(Mandatory = $true)][string]$Previous
    )
    $livePresent = Test-Path -LiteralPath $Live
    if ($livePresent) {
        try {
            Move-Item -LiteralPath $Live -Destination $Previous
        }
        catch {
            throw "기존 배포 사본을 비키지 못해 교체하지 않았다: $Live`n배포 사본의 파일을 잡고 있는 프로세스가 남아 있는지 확인하라."
        }
    }
    try {
        Move-Item -LiteralPath $Staging -Destination $Live
    }
    catch {
        $failure = $_
        if (-not $livePresent) {
            throw $failure
        }
        try {
            Move-Item -LiteralPath $Previous -Destination $Live
        }
        catch {
            throw "교체에 실패한 뒤 직전 배포본을 되돌리지도 못해 배포 자리가 비어 있다 - 관리자가 직접 확인하라. 직전 배포본은 $Previous에 있다.`n  원인: $($failure.Exception.Message)`n  복구 실패: $($_.Exception.Message)"
        }
        throw $failure
    }
}

function Restore-Deployment {
    <#
    .SYNOPSIS
      교체한 사본이 확인을 통과하지 못하면 직전 배포본을 되돌린다.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Live,
        [Parameter(Mandatory = $true)][string]$Previous
    )
    # 확인을 통과하지 못한 사본은 그 자리에 두지 않는다 — 그대로 두면 다음 시작에서 SYSTEM이 실행한다.
    Remove-DirectoryIfPresent $Live
    if (Test-Path -LiteralPath $Previous) {
        Move-Item -LiteralPath $Previous -Destination $Live
    }
}

function Stop-RunningTasks {
    param(
        [Parameter(Mandatory = $true)]$TaskControl,
        [Parameter(Mandatory = $true)][string[]]$Names,
        [Parameter(Mandatory = $true)][string]$WatchdogName
    )
    $runningNames = @(& $TaskControl.GetRunningNames)
    $plan = Get-TaskStopPlan `
        -Names $Names -RunningNames $runningNames -WatchdogName $WatchdogName
    foreach ($name in $plan) {
        & $TaskControl.StopTask $name
    }
}

function Start-PlannedTasks {
    param(
        [Parameter(Mandatory = $true)]$TaskControl,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$Names
    )
    foreach ($name in $Names) {
        & $TaskControl.StartTask $name
    }
}

function Start-StoppedTasks {
    # 이 갱신이 멈춘 작업만 등록 순서대로 되돌린다 — 원래 멈춰 있던 작업은 시작하지 않는다.
    param(
        [Parameter(Mandatory = $true)]$TaskControl,
        [Parameter(Mandatory = $true)][string[]]$Names,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$Stopped
    )
    $plan = Get-TaskStartPlan -Names $Names -RunningBefore $Stopped
    Start-PlannedTasks -TaskControl $TaskControl -Names $plan
}

function Update-Deployment {
    <# .SYNOPSIS 배포 교체와 작업 실행 상태를 하나의 복구 가능한 절차로 처리한다. #>
    param(
        [Parameter(Mandatory = $true)][string]$Live,
        [Parameter(Mandatory = $true)][string]$Staging,
        [Parameter(Mandatory = $true)][string]$Previous,
        [Parameter(Mandatory = $true)][scriptblock]$StageAndVerify,
        [Parameter(Mandatory = $true)][scriptblock]$AfterSwap,
        [Parameter(Mandatory = $true)]$TaskControl,
        [Parameter(Mandatory = $true)][string[]]$Names,
        [Parameter(Mandatory = $true)][string]$WatchdogName,
        [switch]$StartTasks
    )
    # 남은 사본을 이어 쓰지 않고 매번 새로 만든다 — 이어 쓰면 이름·크기·시각이 우연히 맞는 항목을
    # 복사가 건너뛰어 검증하지 않은 내용이 사본에 남을 수 있다.
    Remove-DirectoryIfPresent $Staging
    Remove-DirectoryIfPresent $Previous
    & $StageAndVerify

    $runningBefore = @(& $TaskControl.GetRunningNames)
    $stopPlan = Get-TaskStopPlan `
        -Names $Names -RunningNames $runningBefore -WatchdogName $WatchdogName
    $stopped = @()
    try {
        foreach ($name in $stopPlan) {
            & $TaskControl.StopTask $name
            $stopped += $name
        }
    }
    catch {
        # 배포 자리는 아직 그대로다 — 이미 멈춘 작업만 원래대로 되돌린다.
        Start-StoppedTasks -TaskControl $TaskControl -Names $Names -Stopped $stopped
        throw
    }

    try {
        Switch-Deployment -Staging $Staging -Live $Live -Previous $Previous
    }
    catch {
        # 배포 자리가 비어 있으면 직전 배포본을 되돌리지 못한 것이다 — 실행할 코드가 없는 자리에
        # 작업을 세우지 않고, 배포 자리가 그대로일 때만 실행 상태를 되돌린다.
        if (Test-Path -LiteralPath $Live) {
            Start-StoppedTasks -TaskControl $TaskControl -Names $Names -Stopped $stopped
        }
        throw
    }

    try {
        & $AfterSwap
        $startPlan = Get-TaskStartPlan `
            -Names $Names -RunningBefore $runningBefore -StartTasks:$StartTasks
        Start-PlannedTasks -TaskControl $TaskControl -Names $startPlan
    }
    catch {
        $failure = $_
        try {
            Restore-FailedUpdate `
                -TaskControl $TaskControl `
                -Names $Names `
                -WatchdogName $WatchdogName `
                -RunningBefore $runningBefore `
                -Live $Live `
                -Previous $Previous
        }
        catch {
            throw "교체 이후 실패를 되돌리는 중에 다시 실패했다 - 배포 자리와 작업 실행 상태를 관리자가 직접 확인하라.`n  원인: $($failure.Exception.Message)`n  복구 실패: $($_.Exception.Message)"
        }
        throw $failure
    }
    Remove-DirectoryIfPresent $Previous
}

function Restore-FailedUpdate {
    <#
    .SYNOPSIS
      교체 이후 단계가 실패하면 직전 배포본과 작업 전 실행 상태를 되돌린다.
    #>
    param(
        [Parameter(Mandatory = $true)]$TaskControl,
        [Parameter(Mandatory = $true)][string[]]$Names,
        [Parameter(Mandatory = $true)][string]$WatchdogName,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$RunningBefore,
        [Parameter(Mandatory = $true)][string]$Live,
        [Parameter(Mandatory = $true)][string]$Previous
    )
    # 새 배포본으로 이미 올라온 작업이 있으면 먼저 멈춘다 — 배포 자리를 잡고 있으면 되돌리지 못한다.
    Stop-RunningTasks -TaskControl $TaskControl -Names $Names -WatchdogName $WatchdogName
    Restore-Deployment -Live $Live -Previous $Previous
    Start-PlannedTasks -TaskControl $TaskControl -Names $RunningBefore
}
