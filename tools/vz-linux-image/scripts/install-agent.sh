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
WRAPPER_BIN="${TARGET_DIR}/tldw-agent-guest-wrapper"
SYSTEMD_DIR="${ROOTFS}/etc/systemd/system"
SYSTEMD_UNIT="${SYSTEMD_DIR}/tldw-agent-guest.service"
WORKSPACE_MOUNT_UNIT="${SYSTEMD_DIR}/workspace.mount"
WANTS_DIR="${SYSTEMD_DIR}/multi-user.target.wants"
GETTY_WANTS_DIR="${SYSTEMD_DIR}/getty.target.wants"
MODULES_LOAD_DIR="${ROOTFS}/etc/modules-load.d"
VSOCK_MODULES_FILE="${MODULES_LOAD_DIR}/vsock.conf"
INITRAMFS_MODULES_FILE="${ROOTFS}/etc/initramfs-tools/modules"
WORKSPACE_DIR="${ROOTFS}/workspace"
GOCACHE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tldw-agent-go.XXXXXX")"
trap 'rm -rf "${GOCACHE_DIR}"' EXIT

mkdir -p "${TARGET_DIR}"
mkdir -p "${SYSTEMD_DIR}"
mkdir -p "${WANTS_DIR}"
mkdir -p "${GETTY_WANTS_DIR}"
mkdir -p "${MODULES_LOAD_DIR}"
mkdir -p "${WORKSPACE_DIR}"
mkdir -p "$(dirname "${INITRAMFS_MODULES_FILE}")"

(
  cd "${AGENT_DIR}"
  export GOCACHE="${GOCACHE_DIR}"
  go build -o "${TARGET_BIN}" ./cmd/tldw-agent-guest
)

chmod +x "${TARGET_BIN}"
install -m 0755 "${IMAGE_DIR}/bin/tldw-agent-guest-wrapper" "${WRAPPER_BIN}"
install -m 0644 "${IMAGE_DIR}/systemd/tldw-agent-guest.service" "${SYSTEMD_UNIT}"
install -m 0644 "${IMAGE_DIR}/systemd/workspace.mount" "${WORKSPACE_MOUNT_UNIT}"

cat > "${VSOCK_MODULES_FILE}" <<'EOF'
vsock
vmw_vsock_virtio_transport
virtiofs
virtio_console
EOF

cat >> "${INITRAMFS_MODULES_FILE}" <<'EOF'
virtio_blk
virtio_console
virtio_mmio
virtio_pci
ext4
EOF

if [[ "$(uname -s)" == "Linux" && "$(id -u)" -eq 0 && -x "${ROOTFS}/usr/sbin/update-initramfs" ]]; then
  chroot "${ROOTFS}" update-initramfs -u -k all
fi

ln -sfn ../tldw-agent-guest.service "${WANTS_DIR}/tldw-agent-guest.service"
ln -sfn ../workspace.mount "${WANTS_DIR}/workspace.mount"
ln -sfn /lib/systemd/system/serial-getty@.service "${GETTY_WANTS_DIR}/serial-getty@ttyS0.service"
