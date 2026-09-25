$ErrorActionPreference = 'Stop'
$bundleDir = $PSScriptRoot
if ($env:TLDW_APP_STATE_DIR) {
    $stateRoot = $env:TLDW_APP_STATE_DIR
} else {
    $stateRoot = Join-Path $env:LOCALAPPDATA 'tldw\app'
}
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
