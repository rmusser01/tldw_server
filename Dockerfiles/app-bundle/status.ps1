$ErrorActionPreference = 'Stop'
# Compose inherits the process environment ahead of --env-file.
foreach ($name in @('TLDW_PROJECT_ID', 'TLDW_PUBLIC_PORT', 'SINGLE_USER_API_KEY', 'TLDW_GATEWAY_HOP_SECRET', 'SINGLE_USER_SESSION_COOKIE_NAME', 'CSRF_COOKIE_NAME', 'TLDW_BACKEND_IMAGE', 'TLDW_WEBUI_IMAGE', 'TLDW_GATEWAY_IMAGE')) {
    [Environment]::SetEnvironmentVariable($name, $null, 'Process')
}
$bundleDir = $PSScriptRoot
if ($env:TLDW_APP_STATE_DIR) {
    $stateRoot = $env:TLDW_APP_STATE_DIR
} else {
    $stateRoot = Join-Path $env:LOCALAPPDATA 'tldw\app'
}
& (Join-Path $bundleDir 'start.ps1') -VerifyOnly *> $null
$envFile = Join-Path $stateRoot 'instance\config.env'
if (-not (Test-Path $envFile -PathType Leaf)) { throw 'No initialized tldw instance was found.' }
$lines = Get-Content -LiteralPath $envFile
$projectId = (($lines | Where-Object { $_ -match '^TLDW_PROJECT_ID=' }) -replace '^TLDW_PROJECT_ID=', '')
$publicPort = (($lines | Where-Object { $_ -match '^TLDW_PUBLIC_PORT=' }) -replace '^TLDW_PUBLIC_PORT=', '')
if ($projectId -notmatch '^[a-zA-Z0-9_.-]+$' -or $publicPort -notmatch '^\d{1,5}$') {
    throw 'Instance configuration lacks a valid project ID or port.'
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker is required to inspect this bundle.' }
& docker compose --project-name $projectId --env-file $envFile -f (Join-Path $bundleDir 'compose.yaml') ps
if ($LASTEXITCODE -ne 0) { throw 'Could not read application service status.' }
Write-Output "Browser URL: http://127.0.0.1:$publicPort/"
