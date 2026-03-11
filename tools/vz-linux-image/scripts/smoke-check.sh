#!/usr/bin/env bash
set -euo pipefail

ROOTFS="${TLDW_VZ_LINUX_IMAGE_ROOTFS:-}"
if [[ -z "${ROOTFS}" ]]; then
  echo "TLDW_VZ_LINUX_IMAGE_ROOTFS is required" >&2
  exit 1
fi

TARGET_BIN="${ROOTFS}/usr/local/bin/tldw-agent-guest"
if [[ ! -x "${TARGET_BIN}" ]]; then
  echo "missing guest agent binary at ${TARGET_BIN}" >&2
  exit 1
fi

echo "ok: ${TARGET_BIN}"
