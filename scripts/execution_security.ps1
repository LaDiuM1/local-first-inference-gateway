<#
.SYNOPSIS
  SYSTEM이 실행하는 코드와 SYSTEM이 신뢰하는 운영 상태가 승인된 관리 주체 것인지 판정한다.

.DESCRIPTION
  실행 경로와 운영 상태의 소유자·DACL, Python import 경로가 승인된 관리 경계 안에 있는지 확인한다.
  상태 경로는 실행 코드에서 직접 조회하고, 배포 사본의 editable 포인터는 사본 내부로 옮긴다.
#>

$ErrorActionPreference = "Stop"

# 소유자 설정에 쓰는 Windows 기본 도구 — PATH에는 사용자가 쓸 수 있는 디렉터리가 앞설 수 있으므로
# 이름이 아니라 경로로 실행한다.
$script:IcaclsPath = "C:\Windows\System32\icacls.exe"

# 설치 중단과 -ValidateOnly 보고가 같은 문구를 쓴다.
$script:UnsafeExecutionPathMessage = "UNSAFE SYSTEM 실행 경로 - 승인되지 않은 주체가 수정할 수 있다"
$script:ExternalImportPathMessage = "UNSAFE SYSTEM import 경로 - 배포 사본 밖의 코드를 import한다"
$script:UnsafeStatePathMessage = "UNSAFE 운영 상태 파일 - 승인되지 않은 주체가 수정할 수 있다"
$script:UnadoptableStateDirectoryMessage = "UNSAFE 운영 상태 디렉터리 - 비관리자도 파일을 만들 수 있던 자리다"

$script:AdministratorsSid = "S-1-5-32-544"
# 권한 상승 없이는 쓸 수 없는 주체 — SYSTEM, Administrators, TrustedInstaller.
$script:ApprovedAdministrativeSids = @(
    "S-1-5-18",
    $script:AdministratorsSid,
    "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"
)

# 대상 자체의 내용이나 권한을 바꿀 수 있는 권한.
$script:TargetModifyRights = [System.Security.AccessControl.FileSystemRights]"WriteData, AppendData, Delete, DeleteSubdirectoriesAndFiles, ChangePermissions, TakeOwnership"
# 상위 디렉터리에서 이미 있는 자식을 갈아치울 수 있는 권한.
$script:ContainerReplaceRights = [System.Security.AccessControl.FileSystemRights]"Delete, DeleteSubdirectoriesAndFiles, ChangePermissions, TakeOwnership"

function Test-ApprovedAdministrativeSid {
    param([System.Security.Principal.SecurityIdentifier]$Sid)
    return $script:ApprovedAdministrativeSids -contains $Sid.Value
}

function Get-PrincipalName {
    param([System.Security.Principal.SecurityIdentifier]$Sid)
    try {
        return $Sid.Translate([System.Security.Principal.NTAccount]).Value
    }
    catch {
        return $Sid.Value
    }
}

function Get-PathSecurity {
    param([string]$Path)
    $sections = [System.Security.AccessControl.AccessControlSections]"Owner, Access"
    if (Test-Path -LiteralPath $Path -PathType Container) {
        return New-Object System.Security.AccessControl.DirectorySecurity($Path, $sections)
    }
    return New-Object System.Security.AccessControl.FileSecurity($Path, $sections)
}

function Get-UnapprovedWriters {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][System.Security.AccessControl.FileSystemRights]$Rights
    )
    $risks = @()
    $security = Get-PathSecurity $Path
    $owner = $security.GetOwner([System.Security.Principal.SecurityIdentifier])
    if ($owner -and -not (Test-ApprovedAdministrativeSid $owner)) {
        $risks += "$Path - 소유자 $(Get-PrincipalName $owner)이(가) 권한을 다시 쓸 수 있다"
    }
    $rules = $security.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier])
    foreach ($rule in $rules) {
        if ($rule.AccessControlType -ne [System.Security.AccessControl.AccessControlType]::Allow) {
            continue
        }
        # 상속 전용 ACE는 이 항목이 아니라 앞으로 만들 자식에만 적용된다.
        if ($rule.PropagationFlags -band [System.Security.AccessControl.PropagationFlags]::InheritOnly) {
            continue
        }
        if (-not ($rule.FileSystemRights -band $Rights)) {
            continue
        }
        if (Test-ApprovedAdministrativeSid $rule.IdentityReference) {
            continue
        }
        $risks += "$Path - $(Get-PrincipalName $rule.IdentityReference)에게 수정 권한이 있다"
    }
    return $risks
}

function Get-AncestorDirectories {
    # 경로 문자열만 다룬다 — Split-Path는 와일드카드로 해석하거나(-Path) -Parent를 받지 못한다(-LiteralPath).
    param([string]$Path)
    $ancestors = @()
    $current = [System.IO.Path]::GetDirectoryName($Path)
    while ($current) {
        $ancestors += $current
        $current = [System.IO.Path]::GetDirectoryName($current)
    }
    return $ancestors
}

function Get-ExecutionPathRisks {
    <#
    .SYNOPSIS
      경로를 승인되지 않은 주체가 수정할 수 있는 이유를 모두 돌려준다 — 비어 있으면 안전하다.

    .DESCRIPTION
      대상 자체와 상위 디렉터리 전체를 본다. -Recurse를 주면 디렉터리 아래의 모든 항목까지 확인해
      상속을 끊고 따로 권한을 연 파일이 없는지 검사한다.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$Recurse
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return @("$Path - 경로가 없다")
    }
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $risks = @()
    $risks += Get-UnapprovedWriters -Path $resolved -Rights $script:TargetModifyRights
    foreach ($ancestor in Get-AncestorDirectories $resolved) {
        $risks += Get-UnapprovedWriters -Path $ancestor -Rights $script:ContainerReplaceRights
    }
    if ($Recurse -and (Test-Path -LiteralPath $resolved -PathType Container)) {
        $entries = [System.IO.Directory]::EnumerateFileSystemEntries(
            $resolved, "*", [System.IO.SearchOption]::AllDirectories
        )
        foreach ($entry in $entries) {
            $risks += Get-UnapprovedWriters -Path $entry -Rights $script:TargetModifyRights
        }
    }
    return $risks
}

function Test-AdminOnlyExecutionPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$Recurse
    )
    return @(Get-ExecutionPathRisks -Path $Path -Recurse:$Recurse).Count -eq 0
}

function Assert-NoModifyRisk {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Purpose,
        [Parameter(Mandatory = $true)][string]$Headline,
        [switch]$Recurse
    )
    $risks = @(Get-ExecutionPathRisks -Path $Path -Recurse:$Recurse)
    if ($risks.Count -eq 0) {
        return
    }
    $shown = @($risks | Select-Object -First 5)
    $message = "$Headline ($Purpose):`n  " + ($shown -join "`n  ")
    if ($risks.Count -gt $shown.Count) {
        $message += "`n  ... 외 $($risks.Count - $shown.Count)건"
    }
    throw $message
}

function Assert-AdminOnlyExecutionPath {
    <#
    .SYNOPSIS
      SYSTEM이 실행할 경로가 안전하지 않으면 아무것도 바꾸기 전에 멈춘다.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Purpose,
        [switch]$Recurse
    )
    Assert-NoModifyRisk `
        -Path $Path `
        -Purpose $Purpose `
        -Headline $script:UnsafeExecutionPathMessage `
        -Recurse:$Recurse
}

function Get-ServiceExecutablePath {
    <#
    .SYNOPSIS
      서비스 등록 정보의 실행 명령줄에서 Windows가 실제로 실행하는 실행 파일 경로만 뽑는다.

    .DESCRIPTION
      SYSTEM으로 실행되는 서비스도 SYSTEM 실행 경로다. 등록 정보는 인자까지 붙은 명령줄이므로 이름이
      들어 있는지 보는 것만으로는 무엇을 실행하는지 알 수 없다 — 사용자가 쓸 수 있는 자리의 비슷한
      이름도 그대로 통과한다. 경로를 정확히 뽑아 실행 경로 검사에 그대로 걸 수 있게 한다.
    #>
    param([Parameter(Mandatory = $true)][string]$CommandLine)
    $text = $CommandLine.Trim()
    if ($text.StartsWith('"')) {
        $end = $text.IndexOf('"', 1)
        if ($end -lt 0) {
            throw "서비스 실행 명령줄의 따옴표가 닫히지 않았다: $CommandLine"
        }
        return $text.Substring(1, $end - 1)
    }
    # 따옴표가 없으면 Windows는 첫 공백 앞까지를 실행 파일로 먼저 해석한다 — 공백이 든 경로를 따옴표
    # 없이 등록한 서비스가 실제로 실행하는 자리도 그 자리이므로 그대로 검사 대상이 된다.
    $space = $text.IndexOf(' ')
    if ($space -lt 0) {
        return $text
    }
    return $text.Substring(0, $space)
}

function Assert-AdminOnlyStatePath {
    <#
    .SYNOPSIS
      SYSTEM이 인증에 쓰는 운영 상태 파일이 안전하지 않으면 멈춘다.

    .DESCRIPTION
      API 키 저장소와 Watchdog 보호 키는 게이트웨이가 호출 주체를 판정하는 근거다. 승인되지 않은 주체가
      고칠 수 있으면 키를 심어 인증을 통과하거나 보호 키를 읽을 수 있으므로 소유자와 권한을 함께 본다.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Purpose
    )
    Assert-NoModifyRisk -Path $Path -Purpose $Purpose -Headline $script:UnsafeStatePathMessage
}

function Invoke-Icacls {
    # icacls는 /C를 주면 개별 항목이 실패해도 종료 코드를 0으로 돌려주고, /T는 대상이 아예 없어도
    # "0개 처리"를 성공으로 돌려준다. 실패를 놓치지 않도록 /C를 쓰지 않고 대상 존재도 먼저 확인한다.
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$FailureMessage
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$FailureMessage`n  대상 경로가 없다: $Path"
    }
    $null = & $script:IcaclsPath $Path @Arguments "/Q"
    if ($LASTEXITCODE -ne 0) {
        $exitCode = $LASTEXITCODE
        $global:LASTEXITCODE = 0
        throw "$FailureMessage (icacls $exitCode): $Path"
    }
}

function Set-ApprovedOwner {
    <#
    .SYNOPSIS
      경로의 소유자를 Administrators 그룹으로 옮긴다.

    .DESCRIPTION
      파일을 만든 관리자 계정이 소유자로 남으면, 소유자는 DACL과 무관하게 권한을 다시 쓸 수 있으므로
      권한 상승 없이 실행되는 그 계정의 프로세스가 운영 키나 배포 사본을 고칠 수 있다. 소유자를 권한
      상승 없이는 쓸 수 없는 그룹으로 옮겨 그 경로를 막는다.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$Recurse
    )
    $arguments = @("/setowner", "*$script:AdministratorsSid")
    if ($Recurse) {
        $arguments += "/T"
    }
    Invoke-Icacls `
        -Path $Path `
        -Arguments $arguments `
        -FailureMessage "소유자를 Administrators로 설정하지 못했다"
}

function Reset-InheritedAccess {
    <#
    .SYNOPSIS
      경로의 명시적 ACE를 지워 상위 디렉터리의 제한만 상속하게 한다.
    #>
    param([Parameter(Mandatory = $true)][string]$Path)
    Invoke-Icacls `
        -Path $Path `
        -Arguments @("/reset") `
        -FailureMessage "권한을 상위 디렉터리 상속으로 되돌리지 못했다"
}

function Get-ProtectedStatePaths {
    <#
    .SYNOPSIS
      SYSTEM 작업이 인증 근거로 읽는 상태 파일의 자리를 그 코드에게 직접 묻는다.

    .DESCRIPTION
      설치기가 잠그는 자리와 게이트웨이·Watchdog이 읽는 자리가 어긋나면 잠금은 아무것도 막지 못한다.
      경로를 여기에 다시 적지 않고 배포할 코드가 정한 값을 그대로 쓴다.
    #>
    param([Parameter(Mandatory = $true)][string]$PythonPath)
    # 인용부호 없는 한 줄로 묻는다 — 네이티브 명령에 넘기는 인자의 따옴표는 그대로 전달되지 않는다.
    $probe = 'from gateway import paths; print(paths.STATE_DIRECTORY); print(paths.API_KEY_STORE_PATH); print(paths.WATCHDOG_KEY_PATH)'
    $reported = @(& $PythonPath -c $probe)
    if ($LASTEXITCODE -ne 0 -or $reported.Count -ne 3) {
        throw "운영 상태 파일의 자리를 확인할 수 없다: $PythonPath"
    }
    return [pscustomobject]@{
        Directory   = $reported[0]
        KeyStore    = $reported[1]
        WatchdogKey = $reported[2]
    }
}

function Test-StateDirectoryWasAdminOnly {
    <#
    .SYNOPSIS
      상태 디렉터리에 이미 있는 파일을 채택해도 되는지, 잠그기 전 권한으로 판정한다.

    .DESCRIPTION
      비관리자도 파일을 만들 수 있던 자리에서 발견한 파일은 누가 썼는지 확인할 수 없다. 디렉터리를
      잠가도 그 파일의 소유자와 명시적 ACE는 남으므로, 그런 자리의 파일은 채택하지 않는다. 처음
      만드는 자리도 만드는 순간에는 상위 디렉터리의 권한을 상속하므로 채택 가능한 자리로 보지 않는다.
    #>
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    return @(Get-ExecutionPathRisks -Path $Path).Count -eq 0
}

function Assert-NoPlantedStateFiles {
    <#
    .SYNOPSIS
      채택할 수 없는 자리에 파일이 남아 있으면 멈춘다 — 디렉터리를 잠근 뒤에 부른다.

    .DESCRIPTION
      잠그기 전에 세면 세는 시점과 잠그는 시점 사이에 파일을 심을 수 있다. 그 사이에 놓인 API 키
      저장소는 소유권과 권한을 정리하는 순간 그대로 인증 근거가 된다. 잠근 뒤에는 승인된 관리 주체만
      파일을 만들 수 있으므로, 이 시점에 남아 있는 파일이 곧 잠그기 전에 놓인 파일이다.
    #>
    param([Parameter(Mandatory = $true)][string]$Path)
    $entries = @(Get-ChildItem -LiteralPath $Path -Force)
    if ($entries.Count -eq 0) {
        return
    }
    $names = @($entries | Select-Object -First 5 | ForEach-Object { $_.Name })
    $message = "$script:UnadoptableStateDirectoryMessage ($Path):" +
        "`n  이미 있는 항목: " + ($names -join ", ") +
        "`n  관리자가 직접 확인해 옮기거나 지운 뒤 다시 실행하라."
    throw $message
}

# --- import 경로 격리: SYSTEM의 Python이 배포 사본 안의 코드만 읽는지 확인한다 ---

function Test-PathInsideRoot {
    <#
    .SYNOPSIS
      경로가 루트 자신이거나 그 아래인지 판정한다 — 디렉터리 경계에서만 일치로 본다.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root
    )
    $normalizedPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    $normalizedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
    $sameEntry = [string]::Equals(
        $normalizedPath, $normalizedRoot, [StringComparison]::OrdinalIgnoreCase
    )
    if ($sameEntry) {
        return $true
    }
    # 이름이 겹치기만 하는 이웃(C:\app-copy)이 C:\app 안으로 보이지 않도록 구분자까지 함께 본다.
    return $normalizedPath.StartsWith(
        $normalizedRoot + '\', [StringComparison]::OrdinalIgnoreCase
    )
}

function Get-ExternalImportPaths {
    <#
    .SYNOPSIS
      승인된 루트 밖에 있는 import 경로를 모두 돌려준다 — 비어 있으면 격리가 성립한다.
    #>
    param(
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$Paths,
        [Parameter(Mandatory = $true)][string[]]$ApprovedRoots
    )
    $external = @()
    foreach ($path in $Paths) {
        $contained = $false
        foreach ($root in $ApprovedRoots) {
            if (Test-PathInsideRoot -Path $path -Root $root) {
                $contained = $true
                break
            }
        }
        if (-not $contained) {
            $external += $path
        }
    }
    return $external
}

function Get-PythonImportPaths {
    <#
    .SYNOPSIS
      Python이 실제로 import에 쓰는 경로를 돌려준다 — sys.path 전체와 gateway 패키지 위치.

    .DESCRIPTION
      설정이 아니라 실행 결과를 묻는다. 작업이 실행될 때와 같은 작업 디렉터리에서 물어야 sys.path의
      현재 디렉터리 항목도 실제와 같아진다.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )
    $probe = 'import gateway, json, os, sys; print(json.dumps([os.path.abspath(entry or os.getcwd()) for entry in sys.path] + [os.path.abspath(gateway.__file__)]))'
    Push-Location -LiteralPath $WorkingDirectory
    try {
        $reported = & $PythonPath -c $probe
        $probeExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($probeExitCode -ne 0 -or -not $reported) {
        throw "Python이 실제로 import하는 경로를 확인할 수 없다: $PythonPath"
    }
    return @($reported | ConvertFrom-Json)
}

function Assert-ContainedImportPaths {
    <#
    .SYNOPSIS
      SYSTEM이 실행할 Python이 승인된 루트 밖의 코드를 import하면 아무것도 바꾸기 전에 멈춘다.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string[]]$ApprovedRoots
    )
    $imported = Get-PythonImportPaths -PythonPath $PythonPath -WorkingDirectory $WorkingDirectory
    $external = @(Get-ExternalImportPaths -Paths $imported -ApprovedRoots $ApprovedRoots)
    if ($external.Count -eq 0) {
        return
    }
    $message = $script:ExternalImportPathMessage + ":`n  " + ($external -join "`n  ")
    throw $message
}

function Get-RetargetedPathEntry {
    <#
    .SYNOPSIS
      .pth 한 줄이 원본 트리를 가리키면 배포 사본의 같은 상대 경로로 바꿔 돌려준다.
    #>
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Entry,
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    # 절대 경로 줄만 옮긴다 — site가 실행하는 `import ...` 줄과 site-packages 기준 상대 경로는
    # 원본 트리를 가리키지 않으므로 그대로 둔다.
    if ($Entry.IndexOfAny([System.IO.Path]::GetInvalidPathChars()) -ge 0) {
        return $Entry
    }
    if (-not [System.IO.Path]::IsPathRooted($Entry)) {
        return $Entry
    }
    if (-not (Test-PathInsideRoot -Path $Entry -Root $Source)) {
        return $Entry
    }
    $sourceRoot = [System.IO.Path]::GetFullPath($Source).TrimEnd('\')
    $relative = [System.IO.Path]::GetFullPath($Entry).Substring($sourceRoot.Length).TrimStart('\')
    if (-not $relative) {
        return $Destination
    }
    return (Join-Path $Destination $relative)
}

function Set-DeployedEditablePointers {
    <#
    .SYNOPSIS
      배포 사본의 venv가 원본 작업 트리를 다시 import하지 않도록 .pth 경로를 사본 안으로 옮긴다.

    .DESCRIPTION
      원본을 가리키는 경로 줄만 사본의 같은 상대 경로로 바꾼다. 사본 안을 가리키는 줄은 다시 쓰지
      않으므로 갱신을 반복해도 결과가 같다. 여기서 처리하지 못한 경로는 Assert-ContainedImportPaths가
      등록 전에 잡는다.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    $sitePackages = Join-Path $Destination ".venv\Lib\site-packages"
    if (-not (Test-Path -LiteralPath $sitePackages -PathType Container)) {
        return
    }
    foreach ($file in Get-ChildItem -LiteralPath $sitePackages -Filter "*.pth" -File) {
        $entries = [System.IO.File]::ReadAllLines($file.FullName)
        $retargeted = @()
        foreach ($entry in $entries) {
            $retargeted += Get-RetargetedPathEntry `
                -Entry $entry -Source $Source -Destination $Destination
        }
        if (($retargeted -join "`n") -ne ($entries -join "`n")) {
            [System.IO.File]::WriteAllLines($file.FullName, [string[]]$retargeted)
        }
    }
}
