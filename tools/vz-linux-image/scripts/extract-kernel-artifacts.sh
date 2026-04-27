#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: extract-kernel-artifacts.sh [options]

Options:
  --rootfs <path>       Prepared rootfs directory (required)
  --output-dir <path>   Output directory for kernel/initrd (required)
  --dry-run             Print commands without executing them
  -h, --help            Show this help
EOF
}

emit_command() {
  printf '%s\n' "$*"
}

find_boot_artifact() {
  local boot_dir="$1"
  local pattern="$2"
  find "${boot_dir}" -maxdepth 1 -type f -name "${pattern}" | sort | tail -n 1
}

ROOTFS_DIR=""
OUTPUT_DIR=""
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --rootfs)
      ROOTFS_DIR="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${ROOTFS_DIR}" ]]; then
  echo "rootfs path is required" >&2
  usage >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "output directory is required" >&2
  usage >&2
  exit 1
fi

BOOT_DIR="${ROOTFS_DIR}/boot"
if [[ ! -d "${BOOT_DIR}" ]]; then
  echo "boot artifacts not found under: ${BOOT_DIR}" >&2
  exit 1
fi

KERNEL_SOURCE="$(find_boot_artifact "${BOOT_DIR}" 'vmlinuz-*')"
INITRD_SOURCE="$(find_boot_artifact "${BOOT_DIR}" 'initrd.img-*')"

if [[ -z "${KERNEL_SOURCE}" || -z "${INITRD_SOURCE}" ]]; then
  echo "boot artifacts not found under: ${BOOT_DIR}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  emit_command "install -m 0644 ${KERNEL_SOURCE} ${OUTPUT_DIR}/kernel"
  emit_command "install -m 0644 ${INITRD_SOURCE} ${OUTPUT_DIR}/initrd"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
install -m 0644 "${KERNEL_SOURCE}" "${OUTPUT_DIR}/kernel"
install -m 0644 "${INITRD_SOURCE}" "${OUTPUT_DIR}/initrd"
