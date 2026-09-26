param([switch]$VerifyOnly)
$ErrorActionPreference = 'Stop'
# Compose inherits the process environment ahead of --env-file.
foreach ($name in @('TLDW_PROJECT_ID', 'TLDW_PUBLIC_PORT', 'SINGLE_USER_API_KEY', 'TLDW_GATEWAY_HOP_SECRET', 'SINGLE_USER_SESSION_COOKIE_NAME', 'CSRF_COOKIE_NAME', 'TLDW_BACKEND_IMAGE', 'TLDW_WEBUI_IMAGE', 'TLDW_GATEWAY_IMAGE')) {
    [Environment]::SetEnvironmentVariable($name, $null, 'Process')
}
$bundleDir = $PSScriptRoot
$controlImage = '__CONTROL_IMAGE_DIGEST__'
$trustedKeyId = '__TRUSTED_KEY_ID__'
if ($controlImage.Contains('__') -or $trustedKeyId.Contains('__')) {
    throw 'This source template must be filled with a signed release control image and key.'
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker is required to start this bundle.'
}
# The helper opens this machine's loopback URL, never a remote daemon's listener.
if ($env:DOCKER_CONTEXT) {
    $dockerEndpoint = (& docker context inspect --format '{{.Endpoints.docker.Host}}' $env:DOCKER_CONTEXT 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect the selected Docker context.' }
} elseif ($env:DOCKER_HOST) {
    $dockerEndpoint = $env:DOCKER_HOST
} else {
    $dockerContext = (& docker context show 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Unable to determine the selected Docker context.' }
    $dockerEndpoint = (& docker context inspect --format '{{.Endpoints.docker.Host}}' $dockerContext 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect the selected Docker context.' }
}
if ($dockerEndpoint -notmatch '^unix:///[^\r\n]+$' -and $dockerEndpoint -notmatch '^npipe:////\./pipe/[^\r\n/]+$') {
    throw 'This bundle requires local Docker through a Unix socket or Windows named pipe; remote/TCP daemons cannot serve its local browser URL.'
}
$architecture = (& docker info --format '{{.Architecture}}' 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Docker daemon is unavailable.' }
& docker compose version *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker Compose v2 is required.' }
switch ($architecture) {
    { $_ -in @('x86_64', 'amd64') } { $platform = 'linux/amd64'; break }
    { $_ -in @('aarch64', 'arm64') } { $platform = 'linux/arm64'; break }
    default { throw "Unsupported Docker architecture: $architecture" }
}

if ($env:TLDW_APP_STATE_DIR) {
    $stateRoot = $env:TLDW_APP_STATE_DIR
} else {
    $stateRoot = Join-Path $env:LOCALAPPDATA 'tldw\app'
}
New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null

$portArgs = @()
if (-not $VerifyOnly -and $env:TLDW_APP_PUBLIC_PORT) {
    if ($env:TLDW_APP_PUBLIC_PORT -notmatch '^\d{1,5}$' -or
        [int]$env:TLDW_APP_PUBLIC_PORT -lt 1 -or [int]$env:TLDW_APP_PUBLIC_PORT -gt 65535) {
        throw 'TLDW_APP_PUBLIC_PORT must be a valid decimal port.'
    }
    $portArgs = @('--public-port', $env:TLDW_APP_PUBLIC_PORT)
}
$runArgs = @(
    'run', '--rm', '--network', 'none', '--read-only', '--cap-drop', 'ALL',
    '--security-opt', 'no-new-privileges',
    '--mount', "type=bind,source=$bundleDir,target=/bundle,readonly",
    '--mount', "type=bind,source=$stateRoot,target=/state",
    $controlImage
)
$controlArgs = @(
    '--state', '/state/instance', '--manifest', '/bundle/manifest.json',
    '--signature', '/bundle/manifest.sig', '--bundle-root', '/bundle',
    '--platform', $platform, '--expected-signer', $trustedKeyId
) + $portArgs

& docker @runArgs 'verify' @controlArgs
if ($LASTEXITCODE -ne 0) { throw 'Bundle verification failed; no application containers were started.' }
if ($VerifyOnly) { return }
function Test-FirstPort([string]$RequestedPort) {
    $preflightId = ''
    try {
        $preflightId = (& docker create --network bridge --read-only --cap-drop ALL `
            --security-opt no-new-privileges -p "127.0.0.1:${RequestedPort}:8080" `
            --entrypoint python $controlImage -c 'import time; time.sleep(120)' 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) { $preflightId = ''; throw 'Unable to create Docker port preflight; no origin was saved.' }
        if ($preflightId -notmatch '^[a-f0-9]{64}$') {
            $preflightId = ''; throw 'Invalid preflight resource identity.'
        }
        & docker start $preflightId *> $null
        if ($LASTEXITCODE -ne 0) { return $null }
        if ($RequestedPort) { return $RequestedPort }
        $selectedPort = (& docker inspect --format '{{(index (index .NetworkSettings.Ports "8080/tcp") 0).HostPort}}' $preflightId | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $selectedPort -notmatch '^\d{1,5}$') { throw 'No available port could be confirmed.' }
        return $selectedPort
    } finally {
        if ($preflightId) {
            & docker rm -f $preflightId *> $null
            if ($LASTEXITCODE -ne 0) { throw "Port preflight cleanup failed. Recovery container: $preflightId; state: $stateRoot." }
        }
    }
}
if (-not (Test-Path (Join-Path $stateRoot 'instance'))) {
    $requestedPort = if ($env:TLDW_APP_PUBLIC_PORT) { $env:TLDW_APP_PUBLIC_PORT } else { '8080' }
    if (-not (Test-FirstPort $requestedPort)) {
        if ($env:TLDW_APP_PUBLIC_PORT) {
            throw "Public port $requestedPort is unavailable. Choose another TLDW_APP_PUBLIC_PORT and retry; no origin was saved."
        }
        $alternative = Test-FirstPort ''
        if ($alternative) { throw "Default port 8080 is unavailable. Available choice: set TLDW_APP_PUBLIC_PORT=$alternative and run start.ps1 again; no origin was saved." }
        throw 'No available Docker port choice could be confirmed; no origin was saved.'
    }
}
& docker @runArgs 'init' @controlArgs
if ($LASTEXITCODE -ne 0) { throw 'Instance initialization failed; no application containers were started.' }

$envFile = Join-Path $stateRoot 'instance\config.env'
if (-not (Test-Path $envFile -PathType Leaf)) {
    throw 'Instance configuration is missing after initialization.'
}
$values = @{}
foreach ($line in Get-Content -LiteralPath $envFile) {
    if ($line -match '^(TLDW_PROJECT_ID|TLDW_PUBLIC_PORT)=(.*)$') {
        $values[$Matches[1]] = $Matches[2]
    }
}
$projectId = $values['TLDW_PROJECT_ID']
$publicPort = $values['TLDW_PUBLIC_PORT']
if ($projectId -notmatch '^[a-zA-Z0-9_.-]+$' -or $publicPort -notmatch '^\d{1,5}$') {
    throw 'Instance configuration lacks a valid project ID or port.'
}
$composeArgs = @(
    'compose', '--project-name', $projectId,
    '--env-file', $envFile, '-f', (Join-Path $bundleDir 'compose.yaml')
)
& docker @composeArgs 'pull'
if ($LASTEXITCODE -ne 0) { throw 'Failed to pull the signed image references.' }
function Stop-FailedStart {
    & docker @composeArgs 'down' *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Application failed readiness and cleanup failed; services may still be running. Recovery state: $stateRoot. Retry stop.ps1."
    }
    throw "Application failed readiness; partial services were stopped. Data retained at $stateRoot."
}
& docker @composeArgs 'up' '-d' '--no-build' '--wait' '--wait-timeout' '600'
if ($LASTEXITCODE -ne 0) { Stop-FailedStart }
$containerIds = @(& docker @composeArgs 'ps' '-q' 'app' 'webui' 'gateway')
if ($LASTEXITCODE -ne 0 -or $containerIds.Count -ne 3) { Stop-FailedStart }
# Keep secret-bearing inspection in process memory and feed it only to control stdin.
$inspection = & docker inspect @containerIds "${projectId}_private"
if ($LASTEXITCODE -ne 0) { Stop-FailedStart }
$readyArgs = @(
    'run', '--rm', '-i', '--network', "${projectId}_private", '--read-only', '--cap-drop', 'ALL',
    '--security-opt', 'no-new-privileges',
    '--mount', "type=bind,source=$bundleDir,target=/bundle,readonly",
    '--mount', "type=bind,source=$stateRoot,target=/state,readonly", $controlImage
)
$inspection | & docker @readyArgs 'ready' @controlArgs
$readyExit = $LASTEXITCODE
$inspection = $null
if ($readyExit -ne 0) { Stop-FailedStart }
$browserUrl = "http://127.0.0.1:$publicPort/"
Write-Output "Open $browserUrl"
if ($env:TLDW_APP_NO_BROWSER -ne '1') {
    try { Start-Process $browserUrl } catch { Write-Warning 'Open the printed browser URL manually.' }
}
