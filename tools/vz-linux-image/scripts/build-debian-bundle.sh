#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/builder-defaults.sh"

usage() {
  cat <<'EOF'
Usage: build-debian-bundle.sh [options]

Options:
  --output-dir <path>     Output directory (required)
  --profile <name>        Package profile: minimal or debug (default: minimal)
  --suite <name>          Debian suite (default: bookworm)
  --mirror <url>          Debian mirror URL
  --clean                 Remove intermediate rootfs after bundle assembly
  --dry-run               Print commands without executing them
  -h, --help              Show this help
EOF
}

emit_command() {
  printf '%s\n' "$*"
}

resolve_packages() {
  local profile="${1:-minimal}"
  case "${profile}" in
    minimal)
      compose_package_profiles minimal
      ;;
    debug)
      compose_package_profiles minimal debug
      ;;
    *)
      echo "unsupported profile: ${profile}" >&2
      return 1
      ;;
  esac
}

write_build_metadata() {
  local output_path="$1"
  local profile="$2"
  local suite="$3"
  local rootfs_dir="$4"
  local rootfs_image="$5"
  local kernel_path="$6"
  local initrd_path="$7"
  local bundle_dir="$8"
  local package_json=""

  while IFS= read -r package; do
    [[ -n "${package}" ]] || continue
    if [[ -n "${package_json}" ]]; then
      package_json+=", "
    fi
    package_json+="\"${package}\""
  done < <(resolve_packages "${profile}")

  cat > "${output_path}" <<EOF
{
  "artifact_kind": "canonical_bundle",
  "suite": "${suite}",
  "profile": "${profile}",
  "architecture": "${TLDW_VZ_LINUX_BUILDER_ARCH}",
  "kernel_package": "${TLDW_VZ_LINUX_BUILDER_KERNEL_PACKAGE}",
  "packages": [${package_json}],
  "artifacts": {
    "rootfs_dir": "${rootfs_dir}",
    "rootfs_image": "${rootfs_image}",
    "kernel": "${kernel_path}",
    "initrd": "${initrd_path}",
    "bundle_dir": "${bundle_dir}"
  }
}
EOF
}

OUTPUT_DIR=""
PROFILE="minimal"
SUITE="${TLDW_VZ_LINUX_BUILDER_SUITE}"
MIRROR="${TLDW_VZ_LINUX_BUILDER_MIRROR:-http://deb.debian.org/debian}"
DRY_RUN=0
CLEAN=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --suite)
      SUITE="${2:-}"
      shift 2
      ;;
    --mirror)
      MIRROR="${2:-}"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
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

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "output directory is required" >&2
  usage >&2
  exit 1
fi

ROOTFS_DIR="${OUTPUT_DIR}/rootfs"
ROOTFS_IMAGE="${OUTPUT_DIR}/rootfs.img"
KERNEL_PATH="${OUTPUT_DIR}/kernel"
INITRD_PATH="${OUTPUT_DIR}/initrd"
BUNDLE_DIR="${OUTPUT_DIR}/bundle"
BUILD_INFO_PATH="${OUTPUT_DIR}/build-info.json"

mkdir -p "${OUTPUT_DIR}"
write_build_metadata "${BUILD_INFO_PATH}" "${PROFILE}" "${SUITE}" "${ROOTFS_DIR}" "${ROOTFS_IMAGE}" "${KERNEL_PATH}" "${INITRD_PATH}" "${BUNDLE_DIR}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  emit_command "rootfs/ -> ${ROOTFS_DIR}"
  emit_command "rootfs.img -> ${ROOTFS_IMAGE}"
  emit_command "kernel -> ${KERNEL_PATH}"
  emit_command "initrd -> ${INITRD_PATH}"
  emit_command "bundle/ -> ${BUNDLE_DIR}"
  emit_command "build-info.json -> ${BUILD_INFO_PATH}"
  emit_command "${SCRIPT_DIR}/build-debian-rootfs.sh --dry-run --profile ${PROFILE} --suite ${SUITE} --mirror ${MIRROR} --output-rootfs ${ROOTFS_DIR}"
  emit_command "${SCRIPT_DIR}/pack-rootfs-image.sh --dry-run --rootfs ${ROOTFS_DIR} --output-image ${ROOTFS_IMAGE}"
  emit_command "${SCRIPT_DIR}/extract-kernel-artifacts.sh --dry-run --rootfs ${ROOTFS_DIR} --output-dir ${OUTPUT_DIR}"
  emit_command "TLDW_VZ_LINUX_IMAGE_ROOTFS=${ROOTFS_DIR} TLDW_VZ_LINUX_BUNDLE_KERNEL=${KERNEL_PATH} TLDW_VZ_LINUX_BUNDLE_ROOTFS_IMAGE=${ROOTFS_IMAGE} TLDW_VZ_LINUX_BUNDLE_INITRD=${INITRD_PATH} ${SCRIPT_DIR}/build-bundle.sh ${BUNDLE_DIR}"
  exit 0
fi

"${SCRIPT_DIR}/build-debian-rootfs.sh" --profile "${PROFILE}" --suite "${SUITE}" --mirror "${MIRROR}" --output-rootfs "${ROOTFS_DIR}"
"${SCRIPT_DIR}/pack-rootfs-image.sh" --rootfs "${ROOTFS_DIR}" --output-image "${ROOTFS_IMAGE}"
"${SCRIPT_DIR}/extract-kernel-artifacts.sh" --rootfs "${ROOTFS_DIR}" --output-dir "${OUTPUT_DIR}"

TLDW_VZ_LINUX_IMAGE_ROOTFS="${ROOTFS_DIR}" \
TLDW_VZ_LINUX_BUNDLE_KERNEL="${KERNEL_PATH}" \
TLDW_VZ_LINUX_BUNDLE_ROOTFS_IMAGE="${ROOTFS_IMAGE}" \
TLDW_VZ_LINUX_BUNDLE_INITRD="${INITRD_PATH}" \
bash "${SCRIPT_DIR}/build-bundle.sh" "${BUNDLE_DIR}"

if [[ "${CLEAN}" -eq 1 ]]; then
  rm -rf "${ROOTFS_DIR}"
fi
