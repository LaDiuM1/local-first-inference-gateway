<#
.SYNOPSIS
  6단계 로컬 프로세스를 로그인 없이 시작·복구하는 Windows 작업으로 멱등 등록한다.

.DESCRIPTION
  관리자 전용 배포 사본과 운영 상태 경로를 검증한 뒤 게이트웨이, 두 Ollama, Watchdog을 SYSTEM 시작
  작업으로 등록한다. staging 검증과 실패 시 배포·실행 상태 복구를 거치며, 비소유 작업과 cloudflared
  서비스는 변경하지 않는다. Tunnel 토큰은 받거나 저장하지 않는다.
#>
param(
    [string]$ProjectRoot = "",
    [string]$OllamaPath = "",
    [string]$OllamaModelsPath = "",
    [switch]$ReissueWatchdogKey,
    [switch]$StartTasks,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "execution_security.ps1")
. (Join-Path $PSScriptRoot "stage6_tasks.ps1")
. (Join-Path $PSScriptRoot "deployment_swap.ps1")

$TaskPath = "\LocalFirstInferenceGateway\"
$ManagedPrefix = "Managed by local-first-inference-gateway Stage 6:"
$PowerShellPath = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
# 배포에 쓰는 Windows 기본 도구 — PATH에는 사용자가 쓸 수 있는 디렉터리가 앞설 수 있으므로 이름이
# 아니라 경로로 실행한다.
$RobocopyPath = "C:\Windows\System32\robocopy.exe"
$TaskControlScript = Join-Path $PSScriptRoot "task_control.ps1"
$TaskNames = @("Gateway", "Chat Ollama", "Embedding Ollama", "Watchdog")
$WatchdogTaskName = "Watchdog"
# SYSTEM이 실행하거나 import하는 것만 배포한다 — 테스트·문서·git 이력은 배포 사본에 두지 않는다.
$PayloadDirectories = @("src", "scripts", ".venv")
# routing.yaml과 docs\API.md는 아래에서 존재를 먼저 확인한다. 게이트웨이 설정 .env는 있을 때만 옮긴다.
$PayloadFiles = @("routing.yaml", "docs\API.md", ".env")

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Resolve-OllamaExecutable {
    param([string]$RequestedPath)
    if ($RequestedPath) {
        if (-not (Test-Path -LiteralPath $RequestedPath -PathType Leaf)) {
            throw "지정한 Ollama 실행 파일이 없다."
        }
        return (Resolve-Path -LiteralPath $RequestedPath).Path
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
    throw "Ollama 실행 파일을 찾을 수 없다."
}

function Resolve-ModelsDirectory {
    param([string]$RequestedPath)
    $candidate = $RequestedPath
    if (-not $candidate -and $env:OLLAMA_MODELS) {
        $candidate = $env:OLLAMA_MODELS
    }
    if (-not $candidate) {
        $candidate = Join-Path $env:USERPROFILE ".ollama\models"
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
        throw "Ollama 모델 저장소를 찾을 수 없다. -OllamaModelsPath로 지정하라."
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Get-PythonBasePrefix {
    param([string]$PythonPath)
    # venv의 python은 stdlib을 만든 배포본에서 읽는다. SYSTEM이 import하는 코드이므로 함께 검증한다.
    $basePrefix = & $PythonPath -c "import sys; print(sys.base_prefix)"
    if ($LASTEXITCODE -ne 0 -or -not $basePrefix) {
        throw "Python 배포본 위치를 확인할 수 없다: $PythonPath"
    }
    return $basePrefix.Trim()
}

function Resolve-CloudflaredExecutable {
    param([string]$CommandLine)
    $executable = Get-ServiceExecutablePath $CommandLine
    if ([System.IO.Path]::GetFileName($executable) -ne "cloudflared.exe") {
        throw "cloudflared 이름의 기존 서비스가 공식 실행 파일을 실행하지 않는다: $executable"
    }
    return $executable
}

function Get-SystemExecutedPaths {
    param(
        [string]$PythonBasePrefix,
        [string]$OllamaExecutable,
        [string]$CloudflaredExecutable
    )
    # SYSTEM 작업이 실행하거나 import하는 경로 — 하나라도 사용자가 바꿀 수 있으면 등록하지 않는다.
    $targets = @(
        [pscustomobject]@{
            Path = $PowerShellPath; Purpose = "작업 action 실행 파일"; Recurse = $false
        },
        [pscustomobject]@{
            Path = $OllamaExecutable; Purpose = "Ollama 실행 파일"; Recurse = $false
        },
        [pscustomobject]@{
            Path = $PythonBasePrefix
            Purpose = "Python 배포본과 표준 라이브러리"
            Recurse = $true
        }
    )
    # cloudflared도 SYSTEM 서비스로 공개 트래픽을 게이트웨이에 넣는다 — 관리자가 직접 설치하는
    # 서비스라 권한을 고치지는 않지만, 실행 파일 자리는 같은 기준으로 확인한다.
    if ($CloudflaredExecutable) {
        $targets += [pscustomobject]@{
            Path = $CloudflaredExecutable
            Purpose = "cloudflared 서비스 실행 파일"
            Recurse = $false
        }
    }
    return $targets
}

function New-HiddenPowerShellAction {
    param(
        [string]$ScriptPath,
        [string]$WorkingDirectory,
        [string[]]$ScriptArguments = @()
    )
    $argumentText = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' +
        $ScriptPath + '"'
    foreach ($argument in $ScriptArguments) {
        $escaped = $argument.Replace('"', '\"')
        $argumentText += ' "' + $escaped + '"'
    }
    return New-ScheduledTaskAction `
        -Execute $PowerShellPath `
        -Argument $argumentText `
        -WorkingDirectory $WorkingDirectory
}

function Set-PrivateDirectory {
    param([string]$Path)
    # 상속을 끊고 SYSTEM과 관리자 그룹만 남긴다. 소유자는 DACL과 무관하게 권한을 다시 쓸 수 있으므로,
    # 설치를 실행한 개별 관리자 계정이 아니라 권한 상승 없이는 쓸 수 없는 Administrators로 둔다.
    $null = New-Item -ItemType Directory -Path $Path -Force
    $acl = New-Object Security.AccessControl.DirectorySecurity
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetOwner(
        (New-Object Security.Principal.SecurityIdentifier($script:AdministratorsSid))
    )
    $inheritance = [Security.AccessControl.InheritanceFlags]"ContainerInherit, ObjectInherit"
    $propagation = [Security.AccessControl.PropagationFlags]::None
    $allow = [Security.AccessControl.AccessControlType]::Allow
    foreach ($sidValue in @("S-1-5-18", $script:AdministratorsSid)) {
        $sid = New-Object Security.Principal.SecurityIdentifier($sidValue)
        $account = $sid.Translate([Security.Principal.NTAccount])
        $rule = New-Object Security.AccessControl.FileSystemAccessRule(
            $account,
            [Security.AccessControl.FileSystemRights]::FullControl,
            $inheritance,
            $propagation,
            $allow
        )
        $acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $Path -AclObject $acl
}

function Copy-PayloadDirectory {
    param([string]$Source, [string]$Destination)
    # 사본은 늘 빈 자리에 새로 만들므로 /MIR는 원본 트리를 그대로 옮겨 온다 — 낡은 모듈이 남아
    # import되는 일이 없다. 보안 설명자는 복사하지 않아 대상 항목이 사본 디렉터리의 제한된 ACL을
    # 그대로 상속한다.
    $null = & $RobocopyPath $Source $Destination /MIR /NJH /NJS /NP /NDL /NFL /NC /NS /R:1 /W:1
    $copyExitCode = $LASTEXITCODE
    $global:LASTEXITCODE = 0
    if ($copyExitCode -ge 8) {
        throw "배포 사본을 만들지 못했다(robocopy $copyExitCode): $Source`n원본 트리의 파일을 다른 프로세스가 잠그고 있는지 확인하라."
    }
}

function Publish-Deployment {
    param([string]$Source, [string]$Destination)
    Set-PrivateDirectory $Destination
    foreach ($relativePath in $PayloadDirectories) {
        Copy-PayloadDirectory (Join-Path $Source $relativePath) (Join-Path $Destination $relativePath)
    }
    # 복사한 venv는 아직 원본 트리의 소스를 sys.path에 얹는다 — 사본 안을 가리키게 옮긴다.
    Set-DeployedEditablePointers $Source $Destination
    foreach ($relativePath in $PayloadFiles) {
        $sourceFile = Join-Path $Source $relativePath
        if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) {
            continue
        }
        $targetFile = Join-Path $Destination $relativePath
        $targetDirectory = [System.IO.Path]::GetDirectoryName($targetFile)
        $null = New-Item -ItemType Directory -Path $targetDirectory -Force
        Copy-Item -LiteralPath $sourceFile -Destination $targetFile -Force
    }
    # 복사한 항목의 소유자는 설치를 실행한 관리자 계정이 된다 — 배포 사본 안에서만 소유자를 옮긴다.
    Set-ApprovedOwner -Path $Destination -Recurse
}

function Protect-StateFile {
    param([object[]]$Files)
    foreach ($file in $Files) {
        if (-not (Test-Path -LiteralPath $file.Path)) {
            continue
        }
        # 소유자를 먼저 옮겨야 DACL이 막고 있어도 그 다음 초기화가 소유자 권한으로 통과한다. 명시적
        # ACE를 지우면 상태 디렉터리의 제한만 상속한다.
        Set-ApprovedOwner -Path $file.Path
        Reset-InheritedAccess -Path $file.Path
        Assert-AdminOnlyStatePath -Path $file.Path -Purpose $file.Purpose
    }
}

function Assert-DeploymentIsContained {
    param([string]$Root, [string]$PythonBasePrefix)
    Assert-AdminOnlyExecutionPath -Path $Root -Purpose "게이트웨이·Watchdog 배포 사본" -Recurse
    # 사본의 ACL만으로는 부족하다 — 사본의 Python이 실제로 어디서 코드를 읽는지 물어 확인한다.
    Assert-ContainedImportPaths `
        -PythonPath (Join-Path $Root ".venv\Scripts\python.exe") `
        -WorkingDirectory $Root `
        -ApprovedRoots @($Root, $PythonBasePrefix)
}

function Ensure-TaskFolder {
    $scheduler = New-Object -ComObject "Schedule.Service"
    $scheduler.Connect()
    $root = $scheduler.GetFolder("\")
    try {
        $null = $scheduler.GetFolder($TaskPath)
    }
    catch {
        $null = $root.CreateFolder($TaskPath.Trim("\"), $null)
    }
}

function Get-ExistingTask {
    param([string]$Name)
    return Get-ScheduledTask -TaskPath $TaskPath -TaskName $Name -ErrorAction SilentlyContinue
}

function Assert-TaskIsOwnedOrAbsent {
    param(
        [string]$Name,
        [Microsoft.Management.Infrastructure.CimInstance]$Action
    )
    $existing = Get-ExistingTask $Name
    if (-not $existing) {
        return
    }
    $owned = Test-OwnedTask `
        -Task $existing `
        -Description "$ManagedPrefix $Name" `
        -Execute $Action.Execute `
        -Arguments $Action.Arguments
    if (-not $owned) {
        throw "같은 이름의 기존 작업 '$Name'이 있지만 이 저장소가 소유한 작업으로 확인되지 않는다."
    }
}

function Register-OwnedTask {
    param(
        [string]$Name,
        [Microsoft.Management.Infrastructure.CimInstance]$Action
    )
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $principal = New-ScheduledTaskPrincipal `
        -UserId "SYSTEM" `
        -LogonType ServiceAccount `
        -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet `
        -RestartCount 999 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -StartWhenAvailable `
        -MultipleInstances IgnoreNew `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries
    $definition = New-ScheduledTask `
        -Action $Action `
        -Trigger $trigger `
        -Principal $principal `
        -Settings $settings `
        -Description "$ManagedPrefix $Name"
    $null = Register-ScheduledTask `
        -TaskName $Name `
        -TaskPath $TaskPath `
        -InputObject $definition `
        -Force
}

function Get-RunningTaskNames {
    # 이 저장소가 소유한 작업만 본다 — 소유 여부는 아무것도 바꾸기 전에 이미 확인했다.
    # 이름을 하나씩 내보낸다 — 부르는 쪽이 @()로 받아 없음·하나·여럿을 같게 다룬다.
    foreach ($name in $TaskNames) {
        $task = Get-ExistingTask $name
        if ($task -and $task.State -eq "Running") {
            $name
        }
    }
}

function Publish-WatchdogKey {
    <#
    .SYNOPSIS
      Watchdog 운영 키가 없거나 재발급을 요청받았으면 새로 발급해 보호 파일을 교체한다.

    .DESCRIPTION
      배포 사본을 교체한 뒤에 발급한다 — 새 키를 쓰는 것은 새 배포본의 키 CLI이고, Watchdog은 교체
      뒤 다시 세우는 인스턴스에서 이 파일을 한 번 읽는다.
    #>
    if (-not $ReissueWatchdogKey -and (Test-Path -LiteralPath $watchdogKeyPath)) {
        return
    }
    Push-Location -LiteralPath $deploymentRoot
    try {
        & $deployedPython -m gateway.key_cli `
            --store $keyStorePath `
            issue `
            --client "stage6-watchdog" `
            --protected-output $watchdogKeyPath
        if ($LASTEXITCODE -ne 0) {
            $global:LASTEXITCODE = 0
            throw "Watchdog 운영 키를 안전하게 발급하지 못했다."
        }
    }
    finally {
        Pop-Location
    }
    # 저장소와 보호 키는 원자적 교체가 Administrators 소유로 남기지만 잠금 파일은 그 경로를 거치지
    # 않는다. 소유자를 되돌리고 세 파일이 모두 관리 주체 것인지 확인한다.
    Protect-StateFile $protectedStateFiles
}

function New-TaskControl {
    # Windows 작업 상태를 읽고 바꾸는 유일한 경계 — 수명주기 판단은 순수 계획 함수가 한다.
    # 중지 절차는 Watchdog이 쓰는 것과 같은 스크립트에 둔다.
    return [pscustomobject]@{
        GetRunningNames = { Get-RunningTaskNames }
        StopTask        = {
            param([string]$Name)
            & $TaskControlScript -Action Stop -Kind Task -Name $Name -TaskPath $TaskPath
        }
        StartTask       = {
            param([string]$Name)
            Start-ScheduledTask -TaskPath $TaskPath -TaskName $Name
        }
    }
}

if (-not $ProjectRoot) {
    $ProjectRoot = Join-Path $PSScriptRoot ".."
}
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$sourcePython = Join-Path $resolvedRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $sourcePython -PathType Leaf)) {
    throw "프로젝트 가상환경 Python이 없다. 먼저 uv sync를 실행하라."
}
foreach ($relativePath in @(
    "routing.yaml",
    "docs\API.md",
    "scripts\run_gateway.ps1",
    "scripts\run_chat_ollama.ps1",
    "scripts\run_embedding_ollama.ps1",
    "scripts\run_watchdog.ps1",
    "scripts\task_control.ps1"
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $resolvedRoot $relativePath))) {
        throw "필수 파일이 없다: $relativePath"
    }
}
$resolvedOllama = Resolve-OllamaExecutable $OllamaPath
$resolvedModels = Resolve-ModelsDirectory $OllamaModelsPath
$cloudflaredService = Get-Service -Name "cloudflared" -ErrorAction SilentlyContinue
$cloudflaredExecutable = ""
if ($cloudflaredService) {
    $serviceInfo = Get-CimInstance Win32_Service -Filter "Name='cloudflared'"
    $cloudflaredExecutable = Resolve-CloudflaredExecutable $serviceInfo.PathName
}
$pythonBasePrefix = Get-PythonBasePrefix $sourcePython
$systemExecutedPaths = Get-SystemExecutedPaths `
    $pythonBasePrefix $resolvedOllama $cloudflaredExecutable

if ($ValidateOnly) {
    Write-Output "VALID project"
    Write-Output "VALID Python environment"
    Write-Output "VALID Ollama executable and model store"
    foreach ($target in $systemExecutedPaths) {
        $risks = @(Get-ExecutionPathRisks -Path $target.Path -Recurse:$target.Recurse)
        if ($risks.Count -eq 0) {
            Write-Output "SAFE SYSTEM 실행 경로: $($target.Purpose) - $($target.Path)"
        }
        else {
            Write-Output "$script:UnsafeExecutionPathMessage ($($target.Purpose)): $($risks[0])"
        }
    }
    if ($cloudflaredService) {
        Write-Output "VALID cloudflared Windows service"
    }
    else {
        Write-Output "PENDING cloudflared Windows service"
    }
    foreach ($name in $TaskNames) {
        if (Get-ExistingTask $name) {
            Write-Output "PRESENT scheduled task: $name"
        }
        else {
            Write-Output "PENDING scheduled task: $name"
        }
    }
    exit 0
}

# SYSTEM 실행 경로 검증이 먼저다 — 불변식이 깨져 있으면 권한 확인이나 등록을 시도하지 않는다.
foreach ($target in $systemExecutedPaths) {
    Assert-AdminOnlyExecutionPath `
        -Path $target.Path -Purpose $target.Purpose -Recurse:$target.Recurse
}

if (-not (Test-Administrator)) {
    throw "작업 등록과 보호 저장소 ACL 설정에는 관리자 PowerShell이 필요하다."
}

# 게이트웨이와 Watchdog이 인증 근거로 읽는 자리를 그 코드에게 직접 묻는다 — 잠그는 자리와 읽는
# 자리가 어긋나면 잠금이 아무것도 막지 못한다.
$statePaths = Get-ProtectedStatePaths $sourcePython
$stateDirectory = $statePaths.Directory
$deploymentRoot = Join-Path $stateDirectory "app"
$stagingRoot = Join-Path $stateDirectory "app.staging"
$previousRoot = Join-Path $stateDirectory "app.previous"
$keyStorePath = $statePaths.KeyStore
$watchdogKeyPath = $statePaths.WatchdogKey
$deployedPython = Join-Path $deploymentRoot ".venv\Scripts\python.exe"
# 이 스크립트의 키 CLI가 상태 디렉터리에 만드는 파일 — 게이트웨이 인증이 이 내용을 근거로 판정한다.
$protectedStateFiles = @(
    [pscustomobject]@{ Path = $keyStorePath; Purpose = "API 키 저장소" },
    [pscustomobject]@{ Path = "$keyStorePath.lock"; Purpose = "API 키 저장소 잠금 파일" },
    [pscustomobject]@{ Path = $watchdogKeyPath; Purpose = "Watchdog 보호 키" }
)

$gatewayAction = New-HiddenPowerShellAction `
    (Join-Path $deploymentRoot "scripts\run_gateway.ps1") $deploymentRoot `
    @("-ProjectRoot", $deploymentRoot, "-PythonPath", $deployedPython)
$chatAction = New-HiddenPowerShellAction `
    (Join-Path $deploymentRoot "scripts\run_chat_ollama.ps1") $deploymentRoot `
    @("-OllamaPath", $resolvedOllama, "-ModelsPath", $resolvedModels)
$embeddingAction = New-HiddenPowerShellAction `
    (Join-Path $deploymentRoot "scripts\run_embedding_ollama.ps1") $deploymentRoot `
    @("-OllamaPath", $resolvedOllama, "-ModelsPath", $resolvedModels)
$watchdogAction = New-HiddenPowerShellAction `
    (Join-Path $deploymentRoot "scripts\run_watchdog.ps1") $deploymentRoot `
    @("-ProjectRoot", $deploymentRoot, "-PythonPath", $deployedPython)
$actionsByName = [ordered]@{
    "Gateway"          = $gatewayAction
    "Chat Ollama"      = $chatAction
    "Embedding Ollama" = $embeddingAction
    "Watchdog"         = $watchdogAction
}

# 남의 작업이 하나라도 섞여 있으면 키 발급이나 배포를 시작하기 전에 멈춘다.
foreach ($name in $TaskNames) {
    Assert-TaskIsOwnedOrAbsent $name $actionsByName[$name]
}

# 채택 여부는 잠그기 전 권한으로 정하고, 남은 파일은 잠근 뒤에 확인한다 — 잠그기 전에 확인하면
# 확인과 잠금 사이에 비관리자가 심은 키 저장소를 그대로 인증 근거로 채택한다.
$stateDirectoryWasAdminOnly = Test-StateDirectoryWasAdminOnly $stateDirectory
Set-PrivateDirectory $stateDirectory
if (-not $stateDirectoryWasAdminOnly) {
    Assert-NoPlantedStateFiles $stateDirectory
}
Protect-StateFile $protectedStateFiles
if ((Test-Path -LiteralPath $watchdogKeyPath) -and -not (Test-Path -LiteralPath $keyStorePath)) {
    throw "Watchdog 보호 키는 있지만 API 키 저장소가 없다. 자동으로 덮어쓰지 않는다."
}
Update-Deployment `
    -Live $deploymentRoot `
    -Staging $stagingRoot `
    -Previous $previousRoot `
    -TaskControl (New-TaskControl) `
    -Names $TaskNames `
    -WatchdogName $WatchdogTaskName `
    -StartTasks:$StartTasks `
    -StageAndVerify {
        # 기존 작업이 실행 중인 채로 끝난다 — 여기서 실패하면 배포 자리도 실행 상태도 그대로다.
        Publish-Deployment $resolvedRoot $stagingRoot
        Assert-DeploymentIsContained -Root $stagingRoot -PythonBasePrefix $pythonBasePrefix
    } `
    -AfterSwap {
        # 옮긴 사본이 새 자리에서 자기 코드를 읽도록 포인터를 맞추고, 실제 실행 결과로 다시 확인한다.
        Set-DeployedEditablePointers $stagingRoot $deploymentRoot
        Assert-DeploymentIsContained -Root $deploymentRoot -PythonBasePrefix $pythonBasePrefix
        Publish-WatchdogKey
        Ensure-TaskFolder
        foreach ($name in $TaskNames) {
            Register-OwnedTask $name $actionsByName[$name]
        }
    }

if ($cloudflaredService) {
    Write-Output "cloudflared 공식 Windows 서비스 확인 완료: $cloudflaredExecutable"
}
else {
    Write-Warning "cloudflared 서비스가 없다. Cloudflare 대시보드의 원격 관리형 Tunnel 설치 명령을 관리자 CMD에서 직접 실행하라: cloudflared.exe service install <TUNNEL_TOKEN>"
}
Write-Output "Stage 6 Windows 작업 등록 완료 - SYSTEM 실행 경로: $deploymentRoot"
