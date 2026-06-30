#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="${1:-}"
if [[ -z "${BUNDLE_DIR}" ]]; then
  echo "bundle directory is required" >&2
  exit 1
fi

KERNEL_NAME="${TLDW_VZ_LINUX_BUNDLE_KERNEL_NAME:-kernel}"
ROOTFS_NAME="${TLDW_VZ_LINUX_BUNDLE_ROOTFS_NAME:-rootfs.img}"
INITRD_NAME="${TLDW_VZ_LINUX_BUNDLE_INITRD_NAME:-}"
GUEST_AGENT_PATH="${TLDW_VZ_LINUX_BUNDLE_GUEST_AGENT_PATH:-/usr/local/bin/tldw-agent-guest}"
WORKSPACE_MOUNT_TAG="${TLDW_VZ_LINUX_BUNDLE_WORKSPACE_TAG:-workspace}"
VSOCK_PORT="${TLDW_VZ_LINUX_BUNDLE_VSOCK_PORT:-1024}"

mkdir -p "${BUNDLE_DIR}"

{
  echo "{"
  echo "  \"bundle_version\": \"1\","
  echo "  \"boot_mode\": \"bundle\","
  echo "  \"kernel\": \"${KERNEL_NAME}\","
  if [[ -n "${INITRD_NAME}" ]]; then
    echo "  \"initrd\": \"${INITRD_NAME}\","
  fi
  echo "  \"rootfs\": \"${ROOTFS_NAME}\","
  echo "  \"guest_agent_path\": \"${GUEST_AGENT_PATH}\","
  echo "  \"workspace_mount_tag\": \"${WORKSPACE_MOUNT_TAG}\","
  echo "  \"vsock_port\": ${VSOCK_PORT}"
  echo "}"
} > "${BUNDLE_DIR}/manifest.json"
