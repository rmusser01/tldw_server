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

    return "http://$ApiUrlHost`:$Port"
}

function Resolve-QuickLaunchApiStartDelay {
    if ($env:TLDW_API_START_DELAY) {
        if ($env:TLDW_API_START_DELAY -match '^\d+$') {
            return [int]$env:TLDW_API_START_DELAY
        }

        Write-Warning "[quick-launch] Ignoring invalid TLDW_API_START_DELAY='$($env:TLDW_API_START_DELAY)'; using 120."
    }

    return 120
}

# Exact loopback names only (no case folding, no other 127.x addresses).
function Test-QuickLaunchLoopbackHost {
    param([string]$HostName)

    return @("127.0.0.1", "localhost", "::1") -ccontains $HostName
}

# The host of an http(s) URL, without IPv6 brackets.
function Get-QuickLaunchUrlHost {
    param([string]$Url)

    if ($Url -cmatch '^[^:/]+://(?:[^/@]*@)?(?:\[([^\]]*)\]|([^:/]*))') {
        if ($Matches[1]) {
            return $Matches[1]
        }
        return $Matches[2]
    }
    return ""
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
    Write-Host "  TLDW_API_START_DELAY Maximum seconds 'all' waits for the API's /health before"
    Write-Host "                       starting the WebUI (default: 120, minimum: 5)"
    Write-Host "  NEXT_PUBLIC_API_URL  Browser calls this API URL directly; no automatic sign-in. The"
    Write-Host "                       production build also reads it from apps/tldw-frontend/.env*"
    Write-Host "                       (TLDW_WEBUI_DEV=1 ignores those files and uses the API above)"
    Write-Host "  TLDW_WEBUI_DEV       Set to 1 for the hot-reloading Next.js dev server instead of the"
    Write-Host "                       production build (WebUI development; several GB of RAM; no"
    Write-Host "                       automatic sign-in)"
    Write-Host "  TLDW_WEBUI_EXPOSE_RUNTIME_AUTH"
    Write-Host "                       Default 1: the WebUI signs browsers on this computer in with"
    Write-Host "                       SINGLE_USER_API_KEY, so programs on this computer can sign in"
    Write-Host "                       while it runs. Set 0 (shell, env file, or apps/tldw-frontend/.env*)"
    Write-Host "                       on a shared computer and paste the API key instead."
    Write-Host ""
    Write-Host "In 'all' mode on a loopback API host the API's cookies drop the Secure flag so sign-in"
    Write-Host "works over plain http; SESSION_COOKIE_SECURE=1 (shell or env file) keeps it, e.g. behind"
    Write-Host "a TLS proxy. These settings are read from the shell and $EnvFile only."
    Write-Host ""
    Write-Host "The WebUI is built once and rebuilt when apps/ changes in git or the API address"
    Write-Host "changes. To force a rebuild, for example after changing other NEXT_PUBLIC_* values or"
    Write-Host "updating a zip download (which has no git revision),"
    Write-Host "delete apps/tldw-frontend/.next and launch again."
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

    $SavedEnv = Save-QuickLaunchEnv -Names @("TLDW_ENV_FILE")
    try {
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
    } finally {
        Restore-QuickLaunchEnv -Saved $SavedEnv
    }
}

# Print KEY from dotenv FILE the way the backend reads it (python-dotenv); empty if unreadable.
function Get-QuickLaunchEnvValue {
    param(
        [string]$Path,
        [string]$Key
    )

    if (-not (Test-Path $Path) -or -not (Test-Path $VenvPython)) {
        return ""
    }
    $Value = & $VenvPython -c "import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get(sys.argv[2]) or '')" $Path $Key
    if ($LASTEXITCODE -ne 0) {
        return ""
    }
    return "$Value".Trim()
}

# Set $env:<Name> to the first non-empty value of: the shell, the env file, -Default.
function Set-QuickLaunchSetting {
    param(
        [string]$Name,
        [string]$Default = ""
    )

    $Value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if (-not $Value) {
        $Value = Get-QuickLaunchEnvValue -Path $EnvFile -Key $Name
    }
    if (-not $Value) {
        $Value = $Default
    }
    if ($Value) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

function Save-QuickLaunchEnv {
    param([string[]]$Names)

    $Saved = @{}
    foreach ($Name in $Names) {
        $Saved[$Name] = [Environment]::GetEnvironmentVariable($Name, "Process")
    }
    return $Saved
}

# "Unset" is restored as unset. Passing $null to SetEnvironmentVariable is not enough: PowerShell
# hands a .NET string argument "" instead, and .NET 9 (PowerShell 7.5+) keeps "" as an empty
# variable rather than deleting it.
function Restore-QuickLaunchEnv {
    param([hashtable]$Saved)

    foreach ($Name in $Saved.Keys) {
        if ($null -eq $Saved[$Name]) {
            Remove-Item -Path "Env:$Name" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($Name, $Saved[$Name], "Process")
        }
    }
}

function Get-QuickLaunchFrontendTree {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        return "no-git"
    }
    # Windows PowerShell turns redirected native stderr into errors; do not stop on them here.
    $ErrorActionPreference = "Continue"
    $Tree = & git rev-parse --verify --quiet HEAD:apps 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $Tree) {
        return "no-git"
    }
    return "$Tree".Trim()
}

# Loopback /health probe that never uses a proxy: Invoke-WebRequest follows the system proxy
# and only PowerShell 6+ has -NoProxy, so use WebRequest with no proxy (works on 5.1 and 7).
function Test-QuickLaunchApiHealth {
    param([string]$Url)

    try {
        $Request = [System.Net.WebRequest]::Create($Url)
        $Request.Proxy = $null
        $Request.Timeout = 2000
        $Request.GetResponse().Close()
        return $true
    } catch {
        return $false
    }
}

# Wait for /health rather than a fixed sleep: the production WebUI starts within seconds.
# Called just before the WebUI server starts, so a first-launch build overlaps API boot.
function Wait-QuickLaunchApi {
    $Url = "http://$ApiUrlHost`:$Port/health"
    $MaxWait = [Math]::Max(5, (Resolve-QuickLaunchApiStartDelay))
    Write-Host "[quick-launch] Waiting for the API at $Url (up to ${MaxWait}s) ..."
    $Clock = [System.Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-QuickLaunchApiHealth -Url $Url)) {
        if ($Clock.Elapsed.TotalSeconds -ge $MaxWait) {
            throw "[quick-launch] The API did not answer at $Url within ${MaxWait}s; see the API window, or raise TLDW_API_START_DELAY."
        }
        Start-Sleep -Seconds 1
    }
}

# `next build` takes minutes and several GB of RAM on a laptop, so build only when needed.
# The key names what a build bakes in: networking profile, API address, and the git tree of
# apps/ (so backend-only updates reuse the build). It is written into .next only after a
# successful build, and removed before building, so a failed build always rebuilds next time.
# ponytail: without git (zip download) the tree is "no-git"; delete .next after updating.
function Invoke-QuickLaunchWebUIBuild {
    param(
        [string]$BuildScript,
        [string]$ApiUrl
    )

    $KeyFile = Join-Path $WebUIDir ".next/quick-launch-build-key"
    $BuildKey = "$BuildScript $ApiUrl $(Get-QuickLaunchFrontendTree)"

    if ((Test-Path (Join-Path $WebUIDir ".next/BUILD_ID")) -and (Test-Path $KeyFile) -and ("$(Get-Content -Raw $KeyFile)".Trim() -eq $BuildKey)) {
        Write-Host "[quick-launch] Reusing the existing WebUI build."
        return
    }

    Write-Host "[quick-launch] Building the WebUI; the first launch takes a few minutes..."
    Remove-Item -Path $KeyFile -Force -ErrorAction SilentlyContinue
    Push-Location $WebUIDir
    try {
        & bun run $BuildScript
        if ($LASTEXITCODE -ne 0) {
            throw "[quick-launch] The WebUI build failed. Try 'bun install' in apps/, close other programs to free memory, or set TLDW_WEBUI_DEV=1 to use the dev server."
        }
    } finally {
        Pop-Location
    }
    Set-Content -Path $KeyFile -Value $BuildKey
}

# "<file><TAB><value>" for -Key exactly as `next build`/`next start` load it from the WebUI's
# .env.production.local, .env.local, .env.production and .env files: Next's own loader
# (@next/env, reached through next/package.json), so no venv is needed and parsing, $VAR
# expansion and shell precedence match Next. Prints only the key, and nothing when the shell
# sets it or no file does. --env-file stops bun from loading those files itself. The script
# uses only single quotes: Windows PowerShell 5.1 drops double quotes in native arguments.
$FrontendEnvReader = 'const { createRequire } = require(''node:module''); const key = process.argv[1]; const nextEnv = createRequire(require.resolve(''next/package.json'', { paths: [process.cwd()] }))(''@next/env''); const { parsedEnv, loadedEnvFiles } = nextEnv.loadEnvConfig(process.cwd(), false, { info() {}, error() {} }, true); const file = loadedEnvFiles.find((f) => f.env && Object.hasOwn(f.env, key)); if (parsedEnv && Object.hasOwn(parsedEnv, key) && file) process.stdout.write(file.path + ''\t'' + parsedEnv[key]);'
function Get-QuickLaunchFrontendEnv {
    param([string]$Key)

    $NoEnvFile = New-TemporaryFile
    $SavedEnv = Save-QuickLaunchEnv -Names @("NODE_ENV")
    Push-Location $WebUIDir
    try {
        $env:NODE_ENV = "production"
        $Found = & bun "--env-file=$($NoEnvFile.FullName)" -e $FrontendEnvReader $Key
        if ($LASTEXITCODE -ne 0 -or -not $Found) {
            return @("", "")
        }
        return "$Found" -split "`t", 2
    } finally {
        Pop-Location
        Restore-QuickLaunchEnv -Saved $SavedEnv
        Remove-Item -Path $NoEnvFile.FullName -Force -ErrorAction SilentlyContinue
    }
}

# NEXT_PUBLIC_API_URL in the WebUI's own env files (README tells users to copy
# .env.local.example) is a user override too: quickstart builds refuse it.
function Use-QuickLaunchFrontendEnvApiUrl {
    $File, $Value = Get-QuickLaunchFrontendEnv -Key "NEXT_PUBLIC_API_URL"
    if ($Value) {
        $env:NEXT_PUBLIC_API_URL = $Value
        Write-Host "[quick-launch] Using NEXT_PUBLIC_API_URL from apps/tldw-frontend/$File (the browser calls the API directly)."
    }
}

# Like the Docker WebUI, the WebUI server (never the browser bundle) turns SINGLE_USER_API_KEY
# into a cookie session for browsers on this computer, so nobody pastes a key; programs on this
# computer can sign in too, hence the loopback-only bind and the TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=0
# opt-out (shell, else env file, else the WebUI's .env*, else 1; the default would otherwise
# mask a .env* value, since Next never overrides a set variable). Set after the build so the
# key never reaches it. Unset settings still reach Next from the WebUI's .env* files.
function Set-QuickLaunchRuntimeSignIn {
    Set-QuickLaunchSetting -Name "AUTH_MODE"
    Set-QuickLaunchSetting -Name "SINGLE_USER_API_KEY"
    Set-QuickLaunchSetting -Name "SINGLE_USER_SESSION_COOKIE_NAME"
    $null, $FrontendFlag = Get-QuickLaunchFrontendEnv -Key "TLDW_WEBUI_EXPOSE_RUNTIME_AUTH"
    $FlagDefault = if ($FrontendFlag) { $FrontendFlag } else { "1" }
    Set-QuickLaunchSetting -Name "TLDW_WEBUI_EXPOSE_RUNTIME_AUTH" -Default $FlagDefault

    if ($env:TLDW_WEBUI_EXPOSE_RUNTIME_AUTH -ne "1") {
        Write-Host "[quick-launch] Automatic sign-in is off (TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=$($env:TLDW_WEBUI_EXPOSE_RUNTIME_AUTH)); paste your API key in the WebUI."
    } elseif (-not $env:SINGLE_USER_API_KEY) {
        if (-not (Test-Path $VenvPython)) {
            $Reason = "no Python virtualenv at $VenvDir to read $EnvFile"
        } elseif (-not (Test-Path $EnvFile)) {
            $Reason = "no env file at $EnvFile"
        } else {
            $Reason = "SINGLE_USER_API_KEY is not set in $EnvFile"
        }
        Write-Host "[quick-launch] Automatic sign-in is unavailable: $Reason. Paste your API key in the WebUI."
    } elseif ($Mode -eq "webui") {
        Write-Host "[quick-launch] If sign-in does not stick (for example in Safari), start the API with SESSION_COOKIE_SECURE=0, as 'all' mode does on loopback."
    }
}

function run_webui {
    Test-Bun
    Test-WebUIDirectory

    # Every $env: change below is undone on exit, so a later launch in this window starts clean.
    $SavedEnv = Save-QuickLaunchEnv -Names $WebUIEnvNames
    try {
        if ($HostAddress -eq "0.0.0.0" -and -not $env:NEXT_PUBLIC_API_URL) {
            Write-Host "[quick-launch] API is bound to 0.0.0.0; using 127.0.0.1 for local browser requests."
            Write-Host "[quick-launch] Set NEXT_PUBLIC_API_URL to your LAN URL for non-local browser clients."
        }

        if ($env:TLDW_WEBUI_DEV -eq "1") {
            # Developer opt-in: hot reload, but `next dev` holds several GB of RAM.
            $env:NEXT_PUBLIC_API_URL = Resolve-QuickLaunchApiUrl
            if ($script:ApiStarted) {
                Wait-QuickLaunchApi
            }
            Write-Host ""
            Write-Host "[quick-launch] Starting WebUI dev server (TLDW_WEBUI_DEV=1) on all interfaces, port $WebUIPort; open http://127.0.0.1:$WebUIPort"
            Write-Host "[quick-launch] Using API URL: $($env:NEXT_PUBLIC_API_URL)"
            Write-Host ""
            Set-Location $WebUIDir
            Invoke-CheckedNative -FilePath "bun" -Arguments @("run", "dev", "--", "-p", "$WebUIPort")
            return
        }

        if (-not $env:NEXT_PUBLIC_API_URL) {
            Use-QuickLaunchFrontendEnvApiUrl
        }
        $ApiUrl = Resolve-QuickLaunchApiUrl

        if ($env:NEXT_PUBLIC_API_URL) {
            # Advanced networking: the browser calls NEXT_PUBLIC_API_URL directly. build:dev is
            # still an optimized `next build`; only its networking profile differs from build:prod.
            $env:NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "advanced"
            Invoke-QuickLaunchWebUIBuild -BuildScript "build:dev" -ApiUrl $ApiUrl
            # LAN browsers cannot use a loopback API URL, so keep such a WebUI on loopback too.
            $WebUIHost = if (Test-QuickLaunchLoopbackHost -HostName (Get-QuickLaunchUrlHost -Url $ApiUrl)) { "127.0.0.1" } else { "0.0.0.0" }
            Write-Host "[quick-launch] Using API URL: $ApiUrl"
        } else {
            # Quickstart networking, as in the Docker WebUI: the browser talks only to the WebUI,
            # which proxies /api and /health to the backend origin baked in at build time.
            $env:NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
            $env:TLDW_INTERNAL_API_ORIGIN = $ApiUrl
            Invoke-QuickLaunchWebUIBuild -BuildScript "build:prod" -ApiUrl $ApiUrl
            Set-QuickLaunchRuntimeSignIn
            $WebUIHost = "127.0.0.1"
            Write-Host "[quick-launch] WebUI proxies API requests to: $ApiUrl"
        }

        if ($script:ApiStarted) {
            Wait-QuickLaunchApi
        }
        Write-Host ""
        Write-Host "[quick-launch] Starting WebUI on $WebUIHost`:$WebUIPort; open http://127.0.0.1:$WebUIPort"
        Write-Host "[quick-launch] Next.js warns that `"next start`" does not work with `"output: standalone`"; that warning is expected here."
        Write-Host ""
        Set-Location $WebUIDir
        Invoke-CheckedNative -FilePath "bun" -Arguments @("run", "start", "--", "-p", "$WebUIPort", "-H", $WebUIHost)
    } finally {
        Restore-QuickLaunchEnv -Saved $SavedEnv
    }
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
    # The API window inherits this environment; restore it once the window has started.
    $SavedEnv = Save-QuickLaunchEnv -Names @("SESSION_COOKIE_SECURE", "CSRF_ENABLED")
    try {
        if (Test-QuickLaunchLoopbackHost -HostName $HostAddress) {
            # Plain-http loopback, as in Dockerfiles/docker-compose.webui.yml: Secure cookies are
            # not reliably kept over http://127.0.0.1 (Safari), so drop Secure and keep CSRF on.
            # A value from the shell or the env file wins.
            Set-QuickLaunchSetting -Name "SESSION_COOKIE_SECURE" -Default "0"
            Set-QuickLaunchSetting -Name "CSRF_ENABLED" -Default "1"
        }
        Start-Process -FilePath $PsExe -ArgumentList $apiArgs | Out-Null
    } finally {
        Restore-QuickLaunchEnv -Saved $SavedEnv
    }
    # run_webui waits for the API's /health just before starting the WebUI server.
    $script:ApiStarted = $true
    run_webui
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not $HostAddress) {
    $HostAddress = if ($env:TLDW_HOST) { $env:TLDW_HOST } else { "127.0.0.1" }
}
# Browsers and the WebUI reach a 0.0.0.0-bound API through loopback.
$ApiUrlHost = if ($HostAddress -eq "0.0.0.0") { "127.0.0.1" } else { $HostAddress }
# Every variable run_webui may set; it restores them all on exit.
$WebUIEnvNames = @(
    "NEXT_PUBLIC_API_URL",
    "NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE",
    "TLDW_INTERNAL_API_ORIGIN",
    "AUTH_MODE",
    "SINGLE_USER_API_KEY",
    "SINGLE_USER_SESSION_COOKIE_NAME",
    "TLDW_WEBUI_EXPOSE_RUNTIME_AUTH"
)

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
