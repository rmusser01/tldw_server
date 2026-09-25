$ErrorActionPreference = 'Stop'
$bundleDir = $PSScriptRoot
$controlImage = '__CONTROL_IMAGE_DIGEST__'
$trustedKeyId = '__TRUSTED_KEY_ID__'
if ($controlImage.Contains('__') -or $trustedKeyId.Contains('__')) {
    throw 'This source template must be filled with a signed release control image and key.'
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker is required to start this bundle.'
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
if ($env:TLDW_APP_PUBLIC_PORT) {
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
$occupied = Get-NetTCPConnection -LocalPort ([int]$publicPort) -State Listen -ErrorAction SilentlyContinue
if ($occupied) {
    $ownGateway = (& docker @composeArgs 'ps' '-q' 'gateway' | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $ownGateway) {
        throw "Public port $publicPort is already occupied."
    }
}
& docker @composeArgs 'pull'
if ($LASTEXITCODE -ne 0) { throw 'Failed to pull the signed image references.' }
& docker @composeArgs 'up' '-d' '--no-build'
if ($LASTEXITCODE -ne 0) { throw 'Failed to start the paired application.' }
Write-Output "Open http://127.0.0.1:$publicPort/"
