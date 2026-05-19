#!/bin/bash
# Install tldw-agent native messaging host for Chrome/Chromium browsers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"

# Determine the binary name and path
BINARY_NAME="tldw-agent-host"
BINARY_PATH="$AGENT_DIR/bin/$BINARY_NAME"

# Build if needed
if [ ! -f "$BINARY_PATH" ]; then
    echo "Building tldw-agent-host..."
    cd "$AGENT_DIR"
    mkdir -p bin
    go build -o bin/$BINARY_NAME ./cmd/tldw-agent-host
fi

# Get absolute binary path
BINARY_ABS_PATH="$(cd "$(dirname "$BINARY_PATH")" && pwd)/$(basename "$BINARY_PATH")"

# Chrome extension ID (update this after publishing)
EXTENSION_ID="your-extension-id-here"

# Create manifest
MANIFEST_NAME="com.tldw.agent.json"
MANIFEST_CONTENT=$(cat <<EOF
{
  "name": "com.tldw.agent",
  "description": "tldw Agent Native Messaging Host - Local workspace tools for agentic coding assistance",
  "path": "$BINARY_ABS_PATH",
  "type": "stdio",
  "allowed_origins": [
    "chrome-extension://$EXTENSION_ID/"
  ]
}
EOF
)

# Determine manifest location based on OS
case "$(uname -s)" in
    Darwin)
        # macOS
        CHROME_MANIFEST_DIR="$HOME/Library/Application Support/Google/Chrome/NativeMessagingHosts"
        CHROMIUM_MANIFEST_DIR="$HOME/Library/Application Support/Chromium/NativeMessagingHosts"
        EDGE_MANIFEST_DIR="$HOME/Library/Application Support/Microsoft Edge/NativeMessagingHosts"
        ;;
    Linux)
        # Linux
        CHROME_MANIFEST_DIR="$HOME/.config/google-chrome/NativeMessagingHosts"
        CHROMIUM_MANIFEST_DIR="$HOME/.config/chromium/NativeMessagingHosts"
        EDGE_MANIFEST_DIR="$HOME/.config/microsoft-edge/NativeMessagingHosts"
        ;;
    *)
        echo "Unsupported OS: $(uname -s)"
        echo "For Windows, use install-windows.ps1"
        exit 1
        ;;
esac

# Install for Chrome
if [ -d "$(dirname "$CHROME_MANIFEST_DIR")" ]; then
    echo "Installing for Google Chrome..."
    mkdir -p "$CHROME_MANIFEST_DIR"
    echo "$MANIFEST_CONTENT" > "$CHROME_MANIFEST_DIR/$MANIFEST_NAME"
    echo "  Manifest installed to: $CHROME_MANIFEST_DIR/$MANIFEST_NAME"
fi

# Install for Chromium
if [ -d "$(dirname "$CHROMIUM_MANIFEST_DIR")" ]; then
    echo "Installing for Chromium..."
    mkdir -p "$CHROMIUM_MANIFEST_DIR"
    echo "$MANIFEST_CONTENT" > "$CHROMIUM_MANIFEST_DIR/$MANIFEST_NAME"
    echo "  Manifest installed to: $CHROMIUM_MANIFEST_DIR/$MANIFEST_NAME"
fi

# Install for Edge
if [ -d "$(dirname "$EDGE_MANIFEST_DIR")" ]; then
    echo "Installing for Microsoft Edge..."
    mkdir -p "$EDGE_MANIFEST_DIR"
    echo "$MANIFEST_CONTENT" > "$EDGE_MANIFEST_DIR/$MANIFEST_NAME"
    echo "  Manifest installed to: $EDGE_MANIFEST_DIR/$MANIFEST_NAME"
fi

# Create config directory
CONFIG_DIR="$HOME/.tldw-agent"
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Creating config directory..."
    mkdir -p "$CONFIG_DIR"
fi

# Create default config if not exists
CONFIG_FILE="$CONFIG_DIR/config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating default config..."
    cat > "$CONFIG_FILE" <<EOF
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
  shell: "auto"
  network_allowed: false
  custom_commands: []

security:
  require_approval_for_writes: true
  require_approval_for_exec: true
  redact_secrets: true
EOF
    echo "  Config created at: $CONFIG_FILE"
fi

echo ""
echo "Installation complete!"
echo ""
echo "IMPORTANT: Update the extension ID in the manifest files:"
echo "  1. Get your extension ID from chrome://extensions"
echo "  2. Edit the manifest files and replace 'your-extension-id-here'"
echo ""
echo "Binary location: $BINARY_ABS_PATH"
echo "Config location: $CONFIG_FILE"
