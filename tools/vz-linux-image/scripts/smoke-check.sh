#!/usr/bin/env bash
set -euo pipefail

ROOTFS="${TLDW_VZ_LINUX_IMAGE_ROOTFS:-}"
if [[ -z "${ROOTFS}" ]]; then
  echo "TLDW_VZ_LINUX_IMAGE_ROOTFS is required" >&2
  exit 1
fi

TARGET_BIN="${ROOTFS}/usr/local/bin/tldw-agent-guest"
SYSTEMD_UNIT="${ROOTFS}/etc/systemd/system/tldw-agent-guest.service"
WORKSPACE_MOUNT_UNIT="${ROOTFS}/etc/systemd/system/workspace.mount"
WANTS_DIR="${ROOTFS}/etc/systemd/system/multi-user.target.wants"
WORKSPACE_DIR="${ROOTFS}/workspace"
if [[ ! -x "${TARGET_BIN}" ]]; then
  echo "missing guest agent binary at ${TARGET_BIN}" >&2
  exit 1
fi

if [[ ! -f "${SYSTEMD_UNIT}" ]]; then
  echo "missing guest agent systemd unit at ${SYSTEMD_UNIT}" >&2
  exit 1
fi

if [[ ! -f "${WORKSPACE_MOUNT_UNIT}" ]]; then
  echo "missing workspace mount unit at ${WORKSPACE_MOUNT_UNIT}" >&2
  exit 1
fi

if [[ ! -d "${WORKSPACE_DIR}" ]]; then
  echo "missing workspace directory at ${WORKSPACE_DIR}" >&2
  exit 1
fi

if [[ ! -L "${WANTS_DIR}/tldw-agent-guest.service" ]]; then
  echo "missing enabled guest service symlink at ${WANTS_DIR}/tldw-agent-guest.service" >&2
  exit 1
fi

if [[ ! -L "${WANTS_DIR}/workspace.mount" ]]; then
  echo "missing enabled workspace mount symlink at ${WANTS_DIR}/workspace.mount" >&2
  exit 1
fi

echo "ok: ${TARGET_BIN}"
echo "ok: ${SYSTEMD_UNIT}"
echo "ok: ${WORKSPACE_MOUNT_UNIT}"
echo "ok: ${WORKSPACE_DIR}"
