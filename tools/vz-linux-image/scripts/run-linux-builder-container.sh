#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${IMAGE_DIR}/../.." && pwd)"

CONTAINER_RUNTIME="${TLDW_VZ_LINUX_BUILDER_CONTAINER_RUNTIME:-docker}"
CONTAINER_IMAGE="${TLDW_VZ_LINUX_BUILDER_CONTAINER_IMAGE:-debian:bookworm}"
DRY_RUN=0
FORWARDED_ARGS=()

usage() {
  cat <<'EOF'
Usage: run-linux-builder-container.sh [options]

Options:
  --dry-run             Print the container command without executing it
  -h, --help            Show this help

All other arguments are forwarded to build-debian-bundle.sh.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      shift
      ;;
  esac
done

CONTAINER_CMD=(
  "${CONTAINER_RUNTIME}" run --rm
  -v "${REPO_ROOT}:${REPO_ROOT}"
  -w "${IMAGE_DIR}"
  "${CONTAINER_IMAGE}"
  bash "./scripts/build-debian-bundle.sh" "${FORWARDED_ARGS[@]}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%s\n' "${CONTAINER_CMD[*]}"
  exit 0
fi

"${CONTAINER_CMD[@]}"
