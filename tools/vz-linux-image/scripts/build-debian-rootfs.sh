#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/builder-defaults.sh"

usage() {
  cat <<'EOF'
Usage: build-debian-rootfs.sh [options]

Options:
  --output-rootfs <path>   Output rootfs directory (required)
  --profile <name>         Package profile: minimal or debug (default: minimal)
  --suite <name>           Debian suite (default: bookworm)
  --mirror <url>           Debian mirror URL
  --dry-run                Print commands without executing them
  -h, --help               Show this help
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

OUTPUT_ROOTFS=""
PROFILE="minimal"
SUITE="${TLDW_VZ_LINUX_BUILDER_SUITE}"
MIRROR="${TLDW_VZ_LINUX_BUILDER_MIRROR:-http://deb.debian.org/debian}"
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --output-rootfs)
      OUTPUT_ROOTFS="${2:-}"
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

if [[ -z "${OUTPUT_ROOTFS}" ]]; then
  echo "output rootfs is required" >&2
  usage >&2
  exit 1
fi

PACKAGE_LIST="$(resolve_packages "${PROFILE}")"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  emit_command "debootstrap --arch=${TLDW_VZ_LINUX_BUILDER_ARCH} ${SUITE} ${OUTPUT_ROOTFS} ${MIRROR}"
  emit_command "chroot ${OUTPUT_ROOTFS} apt-get update"
  emit_command "chroot ${OUTPUT_ROOTFS} env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ${PACKAGE_LIST//$'\n'/ }"
  emit_command "${SCRIPT_DIR}/install-agent.sh ${OUTPUT_ROOTFS}"
  exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Linux host required for Debian rootfs builds" >&2
  exit 1
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "root privileges required for Debian rootfs builds" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOTFS}"
debootstrap --arch="${TLDW_VZ_LINUX_BUILDER_ARCH}" "${SUITE}" "${OUTPUT_ROOTFS}" "${MIRROR}"
chroot "${OUTPUT_ROOTFS}" apt-get update
chroot "${OUTPUT_ROOTFS}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ${PACKAGE_LIST//$'\n'/ }
"${SCRIPT_DIR}/install-agent.sh" "${OUTPUT_ROOTFS}"
