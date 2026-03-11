#!/usr/bin/env bash
set -euo pipefail

ROOTFS="${1:-${TLDW_VZ_LINUX_IMAGE_ROOTFS:-}}"
if [[ -z "${ROOTFS}" ]]; then
  echo "rootfs path is required" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${IMAGE_DIR}/../.." && pwd)"
AGENT_DIR="${REPO_ROOT}/tools/tldw-agent"
TARGET_DIR="${ROOTFS}/usr/local/bin"
TARGET_BIN="${TARGET_DIR}/tldw-agent-guest"
SYSTEMD_DIR="${ROOTFS}/etc/systemd/system"
SYSTEMD_UNIT="${SYSTEMD_DIR}/tldw-agent-guest.service"
GOCACHE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tldw-agent-go.XXXXXX")"
trap 'rm -rf "${GOCACHE_DIR}"' EXIT

mkdir -p "${TARGET_DIR}"
mkdir -p "${SYSTEMD_DIR}"

(
  cd "${AGENT_DIR}"
  export GOCACHE="${GOCACHE_DIR}"
  go build -o "${TARGET_BIN}" ./cmd/tldw-agent-guest
)

chmod +x "${TARGET_BIN}"
install -m 0644 "${IMAGE_DIR}/systemd/tldw-agent-guest.service" "${SYSTEMD_UNIT}"
