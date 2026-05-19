#!/usr/bin/env bash
# setup_acp.sh — Download and install the tldw-agent-acp binary for the current platform.
#
# Usage:
#   ./Helper_Scripts/setup_acp.sh [--version VERSION] [--install-dir DIR]
#
# Environment overrides:
#   TLDW_AGENT_VERSION   — Version tag (default: latest)
#   TLDW_AGENT_INSTALL   — Install directory (default: ./bin)
#   GITHUB_TOKEN         — Optional token for private repos / rate limits

set -euo pipefail

BINARY_NAME="tldw-agent-acp"
GITHUB_OWNER="rmusser01"     # If the repo isn't public yet, build from source instead (see fallback instructions below)
GITHUB_REPO="tldw-agent"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
VERSION="${TLDW_AGENT_VERSION:-latest}"
INSTALL_DIR="${TLDW_AGENT_INSTALL:-./bin}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --install-dir) INSTALL_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--version VERSION] [--install-dir DIR]"
            echo ""
            echo "Downloads the tldw-agent-acp binary for the current platform."
            echo ""
            echo "Options:"
            echo "  --version VERSION   Release tag (default: latest)"
            echo "  --install-dir DIR   Installation directory (default: ./bin)"
            echo ""
            echo "Environment variables:"
            echo "  TLDW_AGENT_VERSION  Same as --version"
            echo "  TLDW_AGENT_INSTALL  Same as --install-dir"
            echo "  GITHUB_TOKEN        GitHub token for API requests"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Detect platform
# ---------------------------------------------------------------------------
detect_platform() {
    local os arch

    case "$(uname -s)" in
        Linux*)  os="linux" ;;
        Darwin*) os="darwin" ;;
        *)       echo "ERROR: Unsupported OS: $(uname -s)"; exit 1 ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64)  arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        *)             echo "ERROR: Unsupported architecture: $(uname -m)"; exit 1 ;;
    esac

    echo "${os}_${arch}"
}

PLATFORM="$(detect_platform)"
echo "==> Detected platform: ${PLATFORM}"

# ---------------------------------------------------------------------------
# Resolve version
# ---------------------------------------------------------------------------
auth_header() {
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        echo "Authorization: token ${GITHUB_TOKEN}"
    else
        echo "X-No-Auth: true"
    fi
}

if [[ "$VERSION" == "latest" ]]; then
    echo "==> Resolving latest release..."
    RELEASE_URL="https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/releases/latest"
    VERSION=$(curl -fsSL -H "$(auth_header)" "$RELEASE_URL" | grep '"tag_name"' | head -1 | sed 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
    if [[ -z "$VERSION" ]]; then
        echo "ERROR: Could not determine latest release version."
        echo ""
        echo "This may mean:"
        echo "  1. The tldw-agent repo is not yet public or has no published releases"
        echo "  2. GitHub API rate limit reached (set GITHUB_TOKEN to increase)"
        echo "  3. The repo URL is incorrect"
        echo ""
        echo "Alternative: Build from source (requires Go 1.22+):"
        echo "  git clone https://github.com/${GITHUB_OWNER}/${GITHUB_REPO}.git ../tldw-agent"
        echo "  cd ../tldw-agent && mkdir -p \"${INSTALL_DIR}\" && go build -o \"${INSTALL_DIR}/${BINARY_NAME}\" ./cmd/tldw-agent-acp"
        exit 1
    fi
    echo "==> Latest version: ${VERSION}"
fi

# ---------------------------------------------------------------------------
# Download binary
# ---------------------------------------------------------------------------
ASSET_NAME="${BINARY_NAME}_${PLATFORM}"
DOWNLOAD_URL="https://github.com/${GITHUB_OWNER}/${GITHUB_REPO}/releases/download/${VERSION}/${ASSET_NAME}"
CHECKSUM_URL="${DOWNLOAD_URL}.sha256"

mkdir -p "${INSTALL_DIR}"
TARGET="${INSTALL_DIR}/${BINARY_NAME}"

echo "==> Downloading ${ASSET_NAME} (${VERSION})..."

HTTP_CODE=$(curl -fsSL -w "%{http_code}" -H "$(auth_header)" -o "${TARGET}.tmp" "$DOWNLOAD_URL" 2>/dev/null || true)

if [[ ! -f "${TARGET}.tmp" ]] || [[ "${HTTP_CODE:-0}" != "200" ]]; then
    rm -f "${TARGET}.tmp"
    echo "ERROR: Download failed (HTTP ${HTTP_CODE:-???})."
    echo ""
    echo "The binary may not be published yet, or the repo may not be public."
    echo "Build from source instead:"
    echo "  1. Install Go 1.22+: https://go.dev/dl/"
    echo "  2. Clone the tldw-agent repo:"
    echo "     git clone https://github.com/${GITHUB_OWNER}/${GITHUB_REPO}.git ../tldw-agent"
    echo "  3. Build:"
    echo "     cd ../tldw-agent && go build -o ${INSTALL_DIR}/${BINARY_NAME} ./cmd/tldw-agent-acp"
    echo ""
    echo "Then configure in config.txt:"
    echo "  [ACP]"
    echo "  runner_binary_path = ${INSTALL_DIR}/${BINARY_NAME}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Verify checksum (if available)
# ---------------------------------------------------------------------------
echo "==> Verifying checksum..."
EXPECTED_SUM=$(curl -fsSL -H "$(auth_header)" "$CHECKSUM_URL" 2>/dev/null | awk '{print $1}' || true)

if [[ -n "$EXPECTED_SUM" ]]; then
    if command -v sha256sum &>/dev/null; then
        ACTUAL_SUM=$(sha256sum "${TARGET}.tmp" | awk '{print $1}')
    elif command -v shasum &>/dev/null; then
        ACTUAL_SUM=$(shasum -a 256 "${TARGET}.tmp" | awk '{print $1}')
    else
        echo "WARN: No sha256sum or shasum found — skipping checksum verification"
        ACTUAL_SUM="$EXPECTED_SUM"
    fi

    if [[ "$ACTUAL_SUM" != "$EXPECTED_SUM" ]]; then
        rm -f "${TARGET}.tmp"
        echo "ERROR: Checksum mismatch!"
        echo "  Expected: ${EXPECTED_SUM}"
        echo "  Got:      ${ACTUAL_SUM}"
        exit 1
    fi
    echo "==> Checksum verified"
else
    echo "WARN: No checksum file available — skipping verification"
fi

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
mv "${TARGET}.tmp" "${TARGET}"
chmod +x "${TARGET}"

echo "==> Installed: ${TARGET}"
echo ""
echo "Configure tldw_server to use this binary:"
echo ""
echo "  Option 1 — config.txt:"
echo "    [ACP]"
echo "    runner_binary_path = ${TARGET}"
echo ""
echo "  Option 2 — environment variable:"
echo "    export ACP_RUNNER_BINARY_PATH=${TARGET}"
echo ""

# Quick validation
if "${TARGET}" --version &>/dev/null 2>&1; then
    echo "==> Binary validated: $(${TARGET} --version 2>&1 || echo 'ok')"
else
    echo "==> Binary installed (could not verify --version flag, but file is executable)"
fi

echo "==> ACP setup complete!"
