# Quick-launch local single-user tldw_server without Docker or Make.

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("api", "webui", "all", "help")]
    [string]$Mode = "all",
    [string]$HostAddress,
    [int]$Port,
    [int]$WebUIPort,
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

    if ($env:TLDW_API_PORT) {
        if ($env:TLDW_API_PORT -match '^\d+$') {
            return [int]$env:TLDW_API_PORT
        }

        Write-Warning "[quick-launch] Ignoring invalid TLDW_API_PORT='$($env:TLDW_API_PORT)'; checking TLDW_PORT."
    }

    if ($env:TLDW_PORT) {
        if ($env:TLDW_PORT -match '^\d+$') {
            return [int]$env:TLDW_PORT
        }

        Write-Warning "[quick-launch] Ignoring invalid TLDW_PORT='$($env:TLDW_PORT)'; using 8000."
    }

    return 8000
}

function Resolve-QuickLaunchWebUIPort {
    param(
        [bool]$HasPortParameter,
        [int]$PortParameter
    )

    if ($HasPortParameter) {
        return $PortParameter
    }

    if ($env:TLDW_WEBUI_PORT) {
        if ($env:TLDW_WEBUI_PORT -match '^\d+$') {
            return [int]$env:TLDW_WEBUI_PORT
        }

        Write-Warning "[quick-launch] Ignoring invalid TLDW_WEBUI_PORT='$($env:TLDW_WEBUI_PORT)'; using 8080."
    }

    return 8080
}

function Resolve-QuickLaunchApiUrl {
    if ($env:NEXT_PUBLIC_API_URL) {
        return $env:NEXT_PUBLIC_API_URL
    }

    $ApiUrlHost = $HostAddress
    if ($ApiUrlHost -eq "0.0.0.0") {
        $ApiUrlHost = "127.0.0.1"
        Write-Host "[quick-launch] API is bound to 0.0.0.0; using 127.0.0.1 for local browser requests."
        Write-Host "[quick-launch] Set NEXT_PUBLIC_API_URL to your LAN URL for non-local browser clients."
    }

    return "http://$ApiUrlHost`:$Port"
}

function Resolve-QuickLaunchApiStartDelay {
    if ($env:TLDW_API_START_DELAY) {
        if ($env:TLDW_API_START_DELAY -match '^\d+$') {
            return [int]$env:TLDW_API_START_DELAY
        }

        Write-Warning "[quick-launch] Ignoring invalid TLDW_API_START_DELAY='$($env:TLDW_API_START_DELAY)'; using 2."
    }

    return 2
}

function Show-Usage {
    Write-Host "Usage: .\quick-launch.ps1 [api|webui|all] [-HostAddress 127.0.0.1] [-Port 8000] [-WebUIPort 8080]"
    Write-Host ""
    Write-Host "Modes:"
    Write-Host "  api     Start the FastAPI backend only on http://$HostAddress`:$Port"
    Write-Host "  webui   Start the Next.js WebUI only on http://127.0.0.1:$WebUIPort"
    Write-Host "  all     Start the backend and WebUI (default)"
    Write-Host ""
    Write-Host "Environment:"
    Write-Host "  TLDW_PYTHON          Python executable for venv creation"
    Write-Host "  TLDW_VENV_DIR        Virtualenv directory (default: .venv)"
    Write-Host "  TLDW_ENV_FILE        Env file path"
    Write-Host "  TLDW_HOST            Backend host (default: 127.0.0.1)"
    Write-Host "  TLDW_API_PORT        Backend port (default: TLDW_PORT or 8000)"
    Write-Host "  TLDW_PORT            Legacy backend port override"
    Write-Host "  TLDW_WEBUI_PORT      WebUI port (default: 8080)"
    Write-Host "  NEXT_PUBLIC_API_URL  Override WebUI API URL (browser calls this URL directly)"
    Write-Host "  TLDW_WEBUI_DEV       Set to 1 to run the hot-reloading Next.js dev server instead of"
    Write-Host "                       the production build (for WebUI development; uses several GB of RAM)"
    Write-Host ""
    Write-Host "The WebUI is built once and rebuilt only when the git revision or API address"
    Write-Host "changes. Delete apps/tldw-frontend/.next to force a rebuild."
}

function Test-Bun {
    if (-not (Get-Command bun -ErrorAction SilentlyContinue)) {
        throw "[quick-launch] Bun is required to launch the WebUI but was not found in PATH. Install Bun from https://bun.sh/docs/installation, then rerun this launcher."
    }
}

function Test-WebUIDirectory {
    if (-not (Test-Path $WebUIDir)) {
        throw "[quick-launch] WebUI directory not found: $WebUIDir. Update your checkout before launching the WebUI."
    }
}

function Initialize-ApiEnvironment {
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
}

function run_api {
    Write-Host "=== tldw_server quick launch: API ==="
    Write-Host ""
    Initialize-ApiEnvironment

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
}

# Print one value from the env file the way the backend reads it (python-dotenv).
function Get-QuickLaunchEnvFileValue {
    param([string]$Key)

    if (-not (Test-Path $VenvPython)) {
        return ""
    }
    $Value = & $VenvPython -c "import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get(sys.argv[2]) or '')" $EnvFile $Key
    if ($LASTEXITCODE -ne 0) {
        return ""
    }
    return "$Value".Trim()
}

function Get-QuickLaunchGitRevision {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        return "no-git"
    }
    # Windows PowerShell turns redirected native stderr into errors; do not stop on them here.
    $ErrorActionPreference = "Continue"
    $Revision = & git rev-parse --verify --quiet HEAD 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $Revision) {
        return "no-git"
    }
    return "$Revision".Trim()
}

# `next build` takes minutes and several GB of RAM on a laptop, so build only when needed.
# The key names what a build bakes in (networking profile, API address, source revision).
# It is written into .next after a successful build; `next build` empties .next first,
# so a failed build leaves no key and the next launch builds again.
# ponytail: without git (zip download) the revision is "no-git"; delete .next after updating.
function Invoke-QuickLaunchWebUIBuild {
    param(
        [string]$BuildScript,
        [string]$ApiUrl
    )

    $KeyFile = Join-Path $WebUIDir ".next/quick-launch-build-key"
    $BuildKey = "$BuildScript $ApiUrl $(Get-QuickLaunchGitRevision)"

    if ((Test-Path (Join-Path $WebUIDir ".next/BUILD_ID")) -and (Test-Path $KeyFile) -and ("$(Get-Content -Raw $KeyFile)".Trim() -eq $BuildKey)) {
        Write-Host "[quick-launch] Reusing the existing WebUI build."
        return
    }

    Write-Host "[quick-launch] Building the WebUI; the first launch takes a few minutes..."
    Push-Location $WebUIDir
    try {
        Invoke-CheckedNative -FilePath "bun" -Arguments @("run", $BuildScript)
    } finally {
        Pop-Location
    }
    Set-Content -Path $KeyFile -Value $BuildKey
}

function run_webui {
    Test-Bun
    Test-WebUIDirectory

    $HasApiUrlOverride = [bool]$env:NEXT_PUBLIC_API_URL
    $ApiUrl = Resolve-QuickLaunchApiUrl

    if ($env:TLDW_WEBUI_DEV -eq "1") {
        # Developer opt-in: hot reload, but `next dev` holds several GB of RAM.
        $env:NEXT_PUBLIC_API_URL = $ApiUrl
        Write-Host ""
        Write-Host "[quick-launch] Starting WebUI dev server (TLDW_WEBUI_DEV=1) at http://127.0.0.1:$WebUIPort"
        Write-Host "[quick-launch] Using API URL: $($env:NEXT_PUBLIC_API_URL)"
        Write-Host ""
        Set-Location $WebUIDir
        Invoke-CheckedNative -FilePath "bun" -Arguments @("run", "dev", "--", "-p", "$WebUIPort")
        return
    }

    if ($HasApiUrlOverride) {
        # Advanced networking: the browser calls NEXT_PUBLIC_API_URL directly. build:dev is
        # still an optimized `next build`; only its networking profile differs from build:prod.
        $env:NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "advanced"
        Invoke-QuickLaunchWebUIBuild -BuildScript "build:dev" -ApiUrl $ApiUrl
        $WebUIHost = "0.0.0.0"
        Write-Host "[quick-launch] Using API URL: $ApiUrl"
    } else {
        # Quickstart networking, as in the Docker WebUI: the browser talks only to the WebUI,
        # which proxies /api and /health to the backend origin baked in at build time.
        $env:NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
        $env:TLDW_INTERNAL_API_ORIGIN = $ApiUrl
        Invoke-QuickLaunchWebUIBuild -BuildScript "build:prod" -ApiUrl $ApiUrl
        # The WebUI server (never the browser bundle) turns SINGLE_USER_API_KEY into a cookie
        # session for loopback browsers, so nobody pastes a key. Loopback-only for that reason.
        if (-not $env:AUTH_MODE) { $env:AUTH_MODE = Get-QuickLaunchEnvFileValue -Key "AUTH_MODE" }
        if (-not $env:SINGLE_USER_API_KEY) { $env:SINGLE_USER_API_KEY = Get-QuickLaunchEnvFileValue -Key "SINGLE_USER_API_KEY" }
        if (-not $env:TLDW_WEBUI_EXPOSE_RUNTIME_AUTH) { $env:TLDW_WEBUI_EXPOSE_RUNTIME_AUTH = "1" }
        $WebUIHost = "127.0.0.1"
        Write-Host "[quick-launch] WebUI proxies API requests to: $ApiUrl"
    }

    Write-Host ""
    Write-Host "[quick-launch] Starting WebUI at http://127.0.0.1:$WebUIPort"
    Write-Host ""
    Set-Location $WebUIDir
    Invoke-CheckedNative -FilePath "bun" -Arguments @("run", "start", "--", "-p", "$WebUIPort", "-H", $WebUIHost)
}

function run_all {
    Initialize-ApiEnvironment

    Write-Host "=== tldw_server quick launch: API + WebUI ==="
    Write-Host "[quick-launch] Starting API in a new PowerShell window at http://$HostAddress`:$Port"
    Write-Host "[quick-launch] Starting WebUI in this window at http://127.0.0.1:$WebUIPort"

    $apiArgs = @(
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $PSCommandPath,
        "api",
        "-HostAddress",
        $HostAddress,
        "-Port",
        "$Port",
        "-WebUIPort",
        "$WebUIPort",
        "-SkipInstall"
    )
    $PsExe = if ($PSVersionTable.PSEdition -eq "Core") { "pwsh" } else { "powershell" }
    Start-Process -FilePath $PsExe -ArgumentList $apiArgs | Out-Null
    $ApiStartDelay = Resolve-QuickLaunchApiStartDelay
    Start-Sleep -Seconds $ApiStartDelay
    run_webui
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not $HostAddress) {
    $HostAddress = if ($env:TLDW_HOST) { $env:TLDW_HOST } else { "127.0.0.1" }
}

$HasPortParameter = $PSBoundParameters.ContainsKey("Port")
$Port = Resolve-QuickLaunchPort -HasPortParameter $HasPortParameter -PortParameter $Port
$HasWebUIPortParameter = $PSBoundParameters.ContainsKey("WebUIPort")
$WebUIPort = Resolve-QuickLaunchWebUIPort -HasPortParameter $HasWebUIPortParameter -PortParameter $WebUIPort

$VenvDir = if ($env:TLDW_VENV_DIR) { $env:TLDW_VENV_DIR } else { ".venv" }
$VenvPython = Join-Path $VenvDir "Scripts/python.exe"
$InstallMarker = Join-Path $VenvDir ".initialized"
$EnvFile = if ($env:TLDW_ENV_FILE) { $env:TLDW_ENV_FILE } else { "tldw_Server_API/Config_Files/.env" }
$WebUIDir = Join-Path $ScriptDir "apps/tldw-frontend"

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

switch ($Mode) {
    "api" {
        run_api
    }
    "webui" {
        Write-Host "=== tldw_server quick launch: WebUI ==="
        run_webui
    }
    "all" {
        run_all
    }
    "help" {
        Show-Usage
    }
}
