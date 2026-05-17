# Quick-launch local single-user tldw_server without Docker or Make.

[CmdletBinding()]
param(
    [string]$HostAddress,
    [int]$Port,
    [switch]$SkipInstall,
    [switch]$ForceInstall
)

$ErrorActionPreference = "Stop"

function Invoke-CheckedNative {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "[quick-launch] Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

function Resolve-QuickLaunchPort {
    param(
        [bool]$HasPortParameter,
        [int]$PortParameter
    )

    if ($HasPortParameter) {
        return $PortParameter
    }

    if ($env:TLDW_PORT) {
        if ($env:TLDW_PORT -match '^\d+$') {
            return [int]$env:TLDW_PORT
        }

        Write-Warning "[quick-launch] Ignoring invalid TLDW_PORT='$($env:TLDW_PORT)'; using 8000."
    }

    return 8000
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not $HostAddress) {
    $HostAddress = if ($env:TLDW_HOST) { $env:TLDW_HOST } else { "127.0.0.1" }
}
$HasPortParameter = $PSBoundParameters.ContainsKey("Port")
$Port = Resolve-QuickLaunchPort -HasPortParameter $HasPortParameter -PortParameter $Port

$VenvDir = if ($env:TLDW_VENV_DIR) { $env:TLDW_VENV_DIR } else { ".venv" }
$VenvPython = Join-Path $VenvDir "Scripts/python.exe"
$InstallMarker = Join-Path $VenvDir ".initialized"
$EnvFile = if ($env:TLDW_ENV_FILE) { $env:TLDW_ENV_FILE } else { "tldw_Server_API/Config_Files/.env" }

if ($env:TLDW_PYTHON) {
    $PythonExe = $env:TLDW_PYTHON
    $PythonArgs = @()
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonExe = "py"
    $PythonArgs = @("-3")
} else {
    $PythonExe = "python"
    $PythonArgs = @()
}

Write-Host "=== tldw_server quick launch ==="
Write-Host ""

Invoke-CheckedNative -FilePath $PythonExe -Arguments ($PythonArgs + @("-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"))

$VenvCreated = $false
if (-not (Test-Path $VenvPython)) {
    Write-Host "[quick-launch] Creating virtualenv at $VenvDir"
    Invoke-CheckedNative -FilePath $PythonExe -Arguments ($PythonArgs + @("-m", "venv", $VenvDir))
    $VenvCreated = $true
}

if (
    -not $SkipInstall `
    -and $env:TLDW_SKIP_INSTALL -ne "1" `
    -and ($ForceInstall -or $env:TLDW_FORCE_INSTALL -eq "1" -or $VenvCreated -or -not (Test-Path $InstallMarker))
) {
    Write-Host "[quick-launch] Installing/updating local Python dependencies..."
    Invoke-CheckedNative -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    Invoke-CheckedNative -FilePath $VenvPython -Arguments @("-m", "pip", "install", "-e", ".")
    New-Item -Path $InstallMarker -ItemType File -Force | Out-Null
} elseif ($SkipInstall -or $env:TLDW_SKIP_INSTALL -eq "1") {
    Write-Host "[quick-launch] Skipping dependency install"
} else {
    Write-Host "[quick-launch] Dependency setup already completed; set TLDW_FORCE_INSTALL=1 or pass -ForceInstall to reinstall/update."
}

Write-Host "[quick-launch] Configuring local single-user profile..."
Invoke-CheckedNative -FilePath $VenvPython -Arguments @(
    "-m",
    "tldw_Server_API.cli.wizard.cli",
    "init",
    "--profile",
    "local-single",
    "--env-file",
    $EnvFile,
    "--default",
    "--yes"
)

Write-Host ""
Write-Host "[quick-launch] Starting API at http://$HostAddress`:$Port"
Write-Host "[quick-launch] Docs:   http://$HostAddress`:$Port/docs"
Write-Host "[quick-launch] Health: http://$HostAddress`:$Port/health"
Write-Host ""

$env:TLDW_ENV_FILE = $EnvFile
Invoke-CheckedNative -FilePath $VenvPython -Arguments @(
    "-m",
    "uvicorn",
    "tldw_Server_API.app.main:app",
    "--host",
    $HostAddress,
    "--port",
    "$Port"
)
