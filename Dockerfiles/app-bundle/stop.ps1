$ErrorActionPreference = 'Stop'
$bundleDir = $PSScriptRoot
if ($env:TLDW_APP_STATE_DIR) {
    $stateRoot = $env:TLDW_APP_STATE_DIR
} else {
    $stateRoot = Join-Path $env:LOCALAPPDATA 'tldw\app'
}
$envFile = Join-Path $stateRoot 'instance\config.env'
if (-not (Test-Path $envFile -PathType Leaf)) { throw 'No initialized tldw instance was found.' }
$projectLine = Get-Content -LiteralPath $envFile | Where-Object { $_ -match '^TLDW_PROJECT_ID=' }
$projectId = ($projectLine -replace '^TLDW_PROJECT_ID=', '')
if ($projectId -notmatch '^[a-zA-Z0-9_.-]+$') { throw 'Instance project ID is invalid.' }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker is required to stop this bundle.' }
& docker compose --project-name $projectId --env-file $envFile -f (Join-Path $bundleDir 'compose.yaml') down
if ($LASTEXITCODE -ne 0) { throw 'Failed to stop the application.' }
Write-Output 'Application stopped. Persistent data and credentials were retained.'
