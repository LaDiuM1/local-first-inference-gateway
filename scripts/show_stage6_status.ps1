<#
.SYNOPSIS
  비밀을 읽거나 출력하지 않고 6단계 Windows 작업과 cloudflared 서비스 상태를 보여 준다.
#>
param(
    [string]$TaskPath = "\LocalFirstInferenceGateway\"
)

$ErrorActionPreference = "Stop"
foreach ($name in @("Gateway", "Chat Ollama", "Embedding Ollama", "Watchdog")) {
    $task = Get-ScheduledTask -TaskPath $TaskPath -TaskName $name -ErrorAction SilentlyContinue
    if (-not $task) {
        [pscustomobject]@{
            Component = $name
            Kind      = "ScheduledTask"
            State     = "Missing"
            LastRun   = $null
            LastCode  = $null
        }
        continue
    }
    $info = Get-ScheduledTaskInfo -InputObject $task
    [pscustomobject]@{
        Component = $name
        Kind      = "ScheduledTask"
        State     = $task.State
        LastRun   = $info.LastRunTime
        LastCode  = $info.LastTaskResult
    }
}
$service = Get-Service -Name "cloudflared" -ErrorAction SilentlyContinue
[pscustomobject]@{
    Component = "cloudflared"
    Kind      = "WindowsService"
    State     = $(if ($service) { $service.Status } else { "Missing" })
    LastRun   = $null
    LastCode  = $null
}
