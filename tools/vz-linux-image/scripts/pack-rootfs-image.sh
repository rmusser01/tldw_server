#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: pack-rootfs-image.sh [options]

Options:
  --rootfs <path>         Prepared rootfs directory (required)
  --output-image <path>   Output rootfs image path (required)
  --image-size <size>     ext4 image size passed to mke2fs (default: 2G)
  --label <label>         ext4 label (default: tldw-vz-linux-rootfs)
  --dry-run               Print commands without executing them
  -h, --help              Show this help
EOF
}

emit_command() {
  printf '%s\n' "$*"
}

ROOTFS_DIR=""
OUTPUT_IMAGE=""
IMAGE_SIZE="${TLDW_VZ_LINUX_ROOTFS_IMAGE_SIZE:-2G}"
IMAGE_LABEL="${TLDW_VZ_LINUX_ROOTFS_IMAGE_LABEL:-tldw-vz-linux-rootfs}"
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --rootfs)
      ROOTFS_DIR="${2:-}"
      shift 2
      ;;
    --output-image)
      OUTPUT_IMAGE="${2:-}"
      shift 2
      ;;
    --image-size)
      IMAGE_SIZE="${2:-}"
      shift 2
      ;;
    --label)
      IMAGE_LABEL="${2:-}"
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

if [[ ! -d "${ROOTFS_DIR}" ]]; then
  echo "rootfs directory does not exist: ${ROOTFS_DIR}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_IMAGE}" ]]; then
  echo "output image path is required" >&2
  usage >&2
  exit 1
fi

PACK_CMD=(
  mke2fs
  -t ext4
  -d "${ROOTFS_DIR}"
  -L "${IMAGE_LABEL}"
  "${OUTPUT_IMAGE}"
  "${IMAGE_SIZE}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  emit_command "${PACK_CMD[*]}"
  exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Linux host required for rootfs image packing" >&2
  exit 1
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "root privileges required for rootfs image packing" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_IMAGE}")"
"${PACK_CMD[@]}"
