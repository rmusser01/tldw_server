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
$projectLine = Get-Content -LiteralPath $envFile | Where-Object { $_ -match '^TLDW_PROJECT_ID=' }
$projectId = ($projectLine -replace '^TLDW_PROJECT_ID=', '')
if ($projectId -notmatch '^[a-zA-Z0-9_.-]+$') { throw 'Instance project ID is invalid.' }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker is required to stop this bundle.' }
& docker compose --project-name $projectId --env-file $envFile -f (Join-Path $bundleDir 'compose.yaml') down
if ($LASTEXITCODE -ne 0) { throw 'Failed to stop the application.' }
Write-Output 'Application stopped. Persistent data and credentials were retained.'
