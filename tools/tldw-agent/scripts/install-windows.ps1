# Install tldw-agent native messaging host for Windows (Chrome/Firefox/Edge)
# Run this script as Administrator for system-wide installation,
# or as regular user for user-specific installation.

param(
    [string]$Browser = "all",  # "chrome", "firefox", "edge", or "all"
    [string]$ExtensionId = "your-extension-id-here"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AgentDir = Split-Path -Parent $ScriptDir
$BinaryName = "tldw-agent-host.exe"
$BinaryPath = Join-Path $AgentDir "bin\$BinaryName"

# Check if binary exists
if (-not (Test-Path $BinaryPath)) {
    Write-Host "Binary not found at: $BinaryPath"
    Write-Host "Please build the binary first using: go build -o bin\$BinaryName .\cmd\tldw-agent-host"
    exit 1
}

$BinaryAbsPath = (Resolve-Path $BinaryPath).Path
$ManifestName = "com.tldw.agent"

# Chrome/Edge manifest (uses allowed_origins)
$ChromeManifest = @{
    name = "com.tldw.agent"
    description = "tldw Agent Native Messaging Host - Local workspace tools for agentic coding assistance"
    path = $BinaryAbsPath
    type = "stdio"
    allowed_origins = @("chrome-extension://$ExtensionId/")
} | ConvertTo-Json

# Firefox manifest (uses allowed_extensions)
$FirefoxManifest = @{
    name = "com.tldw.agent"
    description = "tldw Agent Native Messaging Host - Local workspace tools for agentic coding assistance"
    path = $BinaryAbsPath
    type = "stdio"
    allowed_extensions = @("tldw-agent@tldw.io")
} | ConvertTo-Json

# Config directory
$ConfigDir = Join-Path $env:USERPROFILE ".tldw-agent"
if (-not (Test-Path $ConfigDir)) {
    New-Item -ItemType Directory -Path $ConfigDir | Out-Null
    Write-Host "Created config directory: $ConfigDir"
}

# Create default config
$ConfigFile = Join-Path $ConfigDir "config.yaml"
if (-not (Test-Path $ConfigFile)) {
    $DefaultConfig = @"
# tldw-agent configuration
server:
  llm_endpoint: "http://localhost:8000"
  api_key: ""

workspace:
  default_root: ""
  blocked_paths:
    - ".env"
    - "*.pem"
    - "*.key"
    - "**/node_modules/**"
    - "**/.git/objects/**"
  max_file_size_bytes: 10000000

execution:
  enabled: true
  timeout_ms: 30000
  shell: "powershell"
  network_allowed: false
  custom_commands: []

security:
  require_approval_for_writes: true
  require_approval_for_exec: true
  redact_secrets: true
"@
    Set-Content -Path $ConfigFile -Value $DefaultConfig
    Write-Host "Created config file: $ConfigFile"
}

# Manifest directory (in user's config)
$ManifestDir = Join-Path $ConfigDir "manifests"
if (-not (Test-Path $ManifestDir)) {
    New-Item -ItemType Directory -Path $ManifestDir | Out-Null
}

function Install-ChromeManifest {
    param([string]$RegistryPath)

    $ManifestPath = Join-Path $ManifestDir "chrome_$ManifestName.json"
    Set-Content -Path $ManifestPath -Value $ChromeManifest

    # Create registry key
    if (-not (Test-Path $RegistryPath)) {
        New-Item -Path $RegistryPath -Force | Out-Null
    }
    Set-ItemProperty -Path $RegistryPath -Name "(Default)" -Value $ManifestPath

    Write-Host "  Manifest: $ManifestPath"
    Write-Host "  Registry: $RegistryPath"
}

function Install-FirefoxManifest {
    $ManifestPath = Join-Path $ManifestDir "firefox_$ManifestName.json"
    Set-Content -Path $ManifestPath -Value $FirefoxManifest

    # Create registry key
    $RegistryPath = "HKCU:\Software\Mozilla\NativeMessagingHosts\$ManifestName"
    if (-not (Test-Path $RegistryPath)) {
        New-Item -Path $RegistryPath -Force | Out-Null
    }
    Set-ItemProperty -Path $RegistryPath -Name "(Default)" -Value $ManifestPath

    Write-Host "  Manifest: $ManifestPath"
    Write-Host "  Registry: $RegistryPath"
}

# Install based on browser selection
if ($Browser -eq "all" -or $Browser -eq "chrome") {
    Write-Host "`nInstalling for Google Chrome..."
    Install-ChromeManifest "HKCU:\Software\Google\Chrome\NativeMessagingHosts\$ManifestName"
}

if ($Browser -eq "all" -or $Browser -eq "edge") {
    Write-Host "`nInstalling for Microsoft Edge..."
    Install-ChromeManifest "HKCU:\Software\Microsoft\Edge\NativeMessagingHosts\$ManifestName"
}

if ($Browser -eq "all" -or $Browser -eq "firefox") {
    Write-Host "`nInstalling for Firefox..."
    Install-FirefoxManifest
}

Write-Host "`n=========================================="
Write-Host "Installation complete!"
Write-Host "=========================================="
Write-Host ""
Write-Host "IMPORTANT: Update the extension ID in the manifest files:"
Write-Host "  1. Get your extension ID from browser settings"
Write-Host "  2. Edit manifests in: $ManifestDir"
Write-Host "  3. Replace 'your-extension-id-here' with your actual extension ID"
Write-Host ""
Write-Host "Binary location: $BinaryAbsPath"
Write-Host "Config location: $ConfigFile"
Write-Host ""
Write-Host "To uninstall, run: .\install-windows.ps1 -Uninstall"
