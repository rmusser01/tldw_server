#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILES_DIR="${IMAGE_DIR}/profiles"

export TLDW_VZ_LINUX_BUILDER_SUITE="${TLDW_VZ_LINUX_BUILDER_SUITE:-bookworm}"
export TLDW_VZ_LINUX_BUILDER_ARCH="${TLDW_VZ_LINUX_BUILDER_ARCH:-arm64}"
export TLDW_VZ_LINUX_BUILDER_KERNEL_PACKAGE="${TLDW_VZ_LINUX_BUILDER_KERNEL_PACKAGE:-linux-image-arm64}"

_profile_path() {
  local profile_name="${1:-}"
  if [[ -z "${profile_name}" ]]; then
    echo "profile name is required" >&2
    return 1
  fi
  printf '%s/%s.packages\n' "${PROFILES_DIR}" "${profile_name}"
}

read_profile_packages() {
  local profile_name="${1:-}"
  local profile_path
  profile_path="$(_profile_path "${profile_name}")"
  if [[ ! -f "${profile_path}" ]]; then
    echo "profile file does not exist: ${profile_path}" >&2
    return 1
  fi

  grep -v '^[[:space:]]*#' "${profile_path}" | awk 'NF'
}

compose_package_profiles() {
  if [[ "$#" -eq 0 ]]; then
    echo "at least one profile is required" >&2
    return 1
  fi

  awk '!seen[$0]++' <(
    while [[ "$#" -gt 0 ]]; do
      read_profile_packages "$1"
      shift
    done
  )
}

print_builder_defaults() {
  cat <<EOF
TLDW_VZ_LINUX_BUILDER_SUITE=${TLDW_VZ_LINUX_BUILDER_SUITE}
TLDW_VZ_LINUX_BUILDER_ARCH=${TLDW_VZ_LINUX_BUILDER_ARCH}
TLDW_VZ_LINUX_BUILDER_KERNEL_PACKAGE=${TLDW_VZ_LINUX_BUILDER_KERNEL_PACKAGE}
EOF
}
