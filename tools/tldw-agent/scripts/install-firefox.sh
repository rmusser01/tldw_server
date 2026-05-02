#!/bin/bash
# Install tldw-agent native messaging host for Firefox

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

# Firefox extension ID (update this after publishing)
EXTENSION_ID="tldw-agent@tldw.io"

# Create manifest (Firefox uses a different format)
MANIFEST_NAME="com.tldw.agent.json"
MANIFEST_CONTENT=$(cat <<EOF
{
  "name": "com.tldw.agent",
  "description": "tldw Agent Native Messaging Host - Local workspace tools for agentic coding assistance",
  "path": "$BINARY_ABS_PATH",
  "type": "stdio",
  "allowed_extensions": [
    "$EXTENSION_ID"
  ]
}
EOF
)

# Determine manifest location based on OS
case "$(uname -s)" in
    Darwin)
        # macOS
        FIREFOX_MANIFEST_DIR="$HOME/Library/Application Support/Mozilla/NativeMessagingHosts"
        ;;
    Linux)
        # Linux
        FIREFOX_MANIFEST_DIR="$HOME/.mozilla/native-messaging-hosts"
        ;;
    *)
        echo "Unsupported OS: $(uname -s)"
        echo "For Windows, use install-windows.ps1"
        exit 1
        ;;
esac

# Install for Firefox
echo "Installing for Firefox..."
mkdir -p "$FIREFOX_MANIFEST_DIR"
echo "$MANIFEST_CONTENT" > "$FIREFOX_MANIFEST_DIR/$MANIFEST_NAME"
echo "  Manifest installed to: $FIREFOX_MANIFEST_DIR/$MANIFEST_NAME"

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
echo "IMPORTANT: Update the extension ID in the manifest file if needed:"
echo "  Default extension ID: $EXTENSION_ID"
echo "  Manifest location: $FIREFOX_MANIFEST_DIR/$MANIFEST_NAME"
echo ""
echo "Binary location: $BINARY_ABS_PATH"
echo "Config location: $CONFIG_FILE"
