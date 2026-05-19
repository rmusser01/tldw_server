#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="${1:-}"
if [[ -z "${BUNDLE_DIR}" ]]; then
  echo "bundle directory is required" >&2
  exit 1
fi

ROOTFS="${TLDW_VZ_LINUX_IMAGE_ROOTFS:-}"
KERNEL_SOURCE="${TLDW_VZ_LINUX_BUNDLE_KERNEL:-}"
ROOTFS_IMAGE_SOURCE="${TLDW_VZ_LINUX_BUNDLE_ROOTFS_IMAGE:-}"
INITRD_SOURCE="${TLDW_VZ_LINUX_BUNDLE_INITRD:-}"

if [[ -z "${ROOTFS}" ]]; then
  echo "TLDW_VZ_LINUX_IMAGE_ROOTFS is required" >&2
  exit 1
fi

require_source_file() {
  local label="$1"
  local path="$2"
  if [[ -z "${path}" ]]; then
    echo "${label} path is required" >&2
    exit 1
  fi
  if [[ ! -f "${path}" ]]; then
    echo "${label} file does not exist: ${path}" >&2
    exit 1
  fi
  if [[ ! -s "${path}" ]]; then
    echo "${label} file is empty: ${path}" >&2
    exit 1
  fi
}

require_source_file "kernel" "${KERNEL_SOURCE}"
require_source_file "rootfs image" "${ROOTFS_IMAGE_SOURCE}"
if [[ -n "${INITRD_SOURCE}" ]]; then
  require_source_file "initrd" "${INITRD_SOURCE}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/install-agent.sh" "${ROOTFS}"

mkdir -p "${BUNDLE_DIR}"
install -m 0644 "${KERNEL_SOURCE}" "${BUNDLE_DIR}/kernel"
install -m 0644 "${ROOTFS_IMAGE_SOURCE}" "${BUNDLE_DIR}/rootfs.img"

if [[ -n "${INITRD_SOURCE}" ]]; then
  install -m 0644 "${INITRD_SOURCE}" "${BUNDLE_DIR}/initrd"
  export TLDW_VZ_LINUX_BUNDLE_INITRD_NAME="initrd"
else
  unset TLDW_VZ_LINUX_BUNDLE_INITRD_NAME 2>/dev/null || true
fi

export TLDW_VZ_LINUX_BUNDLE_KERNEL_NAME="kernel"
export TLDW_VZ_LINUX_BUNDLE_ROOTFS_NAME="rootfs.img"
"${SCRIPT_DIR}/write-manifest.sh" "${BUNDLE_DIR}"
