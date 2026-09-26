# Exercise Windows helper daemon selection without starting Docker containers.
$ErrorActionPreference = 'Stop'
$taskTestRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('task13343-context-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $taskTestRoot | Out-Null
$names = @('DOCKER_CONTEXT', 'DOCKER_HOST', 'TLDW_APP_STATE_DIR', 'TLDW_APP_PUBLIC_PORT',
    'TLDW_PROJECT_ID', 'TLDW_PUBLIC_PORT', 'SINGLE_USER_API_KEY', 'TLDW_GATEWAY_HOP_SECRET',
    'SINGLE_USER_SESSION_COOKIE_NAME', 'CSRF_COOKIE_NAME', 'TLDW_BACKEND_IMAGE', 'TLDW_WEBUI_IMAGE', 'TLDW_GATEWAY_IMAGE')
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
function global:docker {
    $global:Task13343DockerCalls += ,@($args)
    $global:LASTEXITCODE = 0
    if ($args[0] -eq 'context' -and $args[1] -eq 'show') { 'fixture-local' }
    if ($args[0] -eq 'context' -and $args[1] -eq 'inspect') { $global:Task13343DockerEndpoint }
    if ($args[0] -eq 'info') { 'x86_64' }
    if ($args[0] -eq 'compose' -and $args[1] -eq 'version') { 'Docker Compose version v2' }
}
try {
    $template = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot '../Dockerfiles/app-bundle/start.ps1')
    $template = $template.Replace('__CONTROL_IMAGE_DIGEST__', ('localhost:5000/tldw/control@sha256:' + ('a' * 64))).Replace('__TRUSTED_KEY_ID__', 'fixture-key')
    $helper = Join-Path $taskTestRoot 'start.ps1'
    Set-Content -LiteralPath $helper -Value $template -Encoding utf8
    $cases = @(
        @{ Endpoint='ssh://remote.invalid'; Selection='context'; Allowed=$false },
        @{ Endpoint='tcp://remote.invalid:2376'; Selection='context'; Allowed=$false },
        @{ Endpoint='ssh://remote.invalid'; Selection='host'; Allowed=$false },
        @{ Endpoint='tcp://127.0.0.1:2375'; Selection='host'; Allowed=$false },
        @{ Endpoint='unix:///fixture/docker.sock'; Selection='context'; Allowed=$true },
        @{ Endpoint='npipe:////./pipe/docker_engine'; Selection='default'; Allowed=$true },
        @{ Endpoint='npipe:////./pipe/docker_engine'; Selection='override'; Allowed=$true }
    )
    $index = 0
    foreach ($case in $cases) {
        $global:Task13343DockerCalls = @()
        $global:Task13343DockerEndpoint = $case.Endpoint
        $env:DOCKER_CONTEXT = if ($case.Selection -in @('context', 'override')) { 'fixture-choice' } else { $null }
        $env:DOCKER_HOST = if ($case.Selection -eq 'host') { $case.Endpoint } elseif ($case.Selection -eq 'override') { 'ssh://unused.invalid' } else { $null }
        $env:TLDW_APP_PUBLIC_PORT = $null
        $env:TLDW_APP_STATE_DIR = Join-Path $taskTestRoot ('state-' + $index)
        $failure = $null
        try { & $helper -VerifyOnly } catch { $failure = $_.Exception.Message }
        if ($case.Allowed) {
            if ($failure) { throw "Local daemon selection failed: $failure" }
            $verifications = @($global:Task13343DockerCalls | Where-Object { $_[0] -eq 'run' -and $_ -contains 'verify' })
            if ($verifications.Count -ne 1) { throw 'Local daemon did not reach signed verification.' }
        } else {
            if ($failure -notlike '*local Docker*') { throw 'Remote daemon was not refused explicitly.' }
            if (Test-Path -LiteralPath $env:TLDW_APP_STATE_DIR) { throw 'Remote daemon created application state.' }
            if (@($global:Task13343DockerCalls | Where-Object { $_[0] -in @('run', 'create', 'info', 'compose') }).Count -ne 0) {
                throw 'Remote daemon reached application or runtime operations.'
            }
        }
        $index++
    }
    Write-Output 'Windows daemon-selection regressions passed; actual Windows Docker runtime remains unqualified.'
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
    Remove-Item Function:\docker
    Remove-Variable Task13343DockerCalls, Task13343DockerEndpoint -Scope Global -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $taskTestRoot -Recurse -Force
}
