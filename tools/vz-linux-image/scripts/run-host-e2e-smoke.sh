#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${IMAGE_DIR}/../.." && pwd)"

BUNDLE_PATH=""
DEFAULT_RUNTIME_DIR="${TMPDIR:-/tmp}/tldw-vz-helper-e2e-$$"
SOCKET_PATH="${DEFAULT_RUNTIME_DIR}/helper.sock"
SERIAL_LOG_DIR="${DEFAULT_RUNTIME_DIR}/serial"
HELPER_PATH="${REPO_ROOT}/tools/macos-vz-helper/.build/debug/macos-vz-helper"
ENTITLEMENTS_PATH=""
PYTHON_BIN=""
DRY_RUN=0
SKIP_BUILD=0
SKIP_SIGN=0
INCLUDE_FAILURE_DRILLS=0
HELPER_PID=""
HELPER_PID_FILE=""

usage() {
  cat <<'EOF'
Usage: run-host-e2e-smoke.sh --bundle PATH [options]

Build/sign/start the macOS Virtualization.framework helper, run the helper
daemon smoke, run real vz_linux ephemeral execution, verify same-session VM
reuse, verify recovery diagnostics plus dry-run repair planning, and stop the
helper.

Options:
  --bundle PATH          Canonical vz_linux bundle directory (required)
  --socket PATH          Host-side helper AF_UNIX socket path
  --serial-log-dir PATH  Directory for helper VM serial logs
  --helper PATH          Helper binary path
  --entitlements PATH    Entitlements plist for ad hoc codesigning unless the helper is already signed
  --python PATH          Python executable for pytest
  --skip-build           Do not build the helper even if the binary is missing
  --skip-sign            Do not codesign the helper
  --include-failure-drills
                         Run manual-only host failure recovery drills after baseline smoke
  --dry-run              Print commands without starting helper or VMs
  -h, --help             Show this help
EOF
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "${option} requires a value" >&2
    exit 1
  fi
}

print_cmd() {
  printf '+'
  local arg
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  "$@"
}

die() {
  echo "$*" >&2
  exit 1
}

cleanup() {
  local pid
  pid="$(current_helper_pid)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
  if [[ -S "${SOCKET_PATH}" ]]; then
    rm -f "${SOCKET_PATH}"
  fi
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --bundle)
      require_value "$1" "${2:-}"
      BUNDLE_PATH="$2"
      shift 2
      ;;
    --socket)
      require_value "$1" "${2:-}"
      SOCKET_PATH="$2"
      shift 2
      ;;
    --serial-log-dir)
      require_value "$1" "${2:-}"
      SERIAL_LOG_DIR="$2"
      shift 2
      ;;
    --helper)
      require_value "$1" "${2:-}"
      HELPER_PATH="$2"
      shift 2
      ;;
    --entitlements)
      require_value "$1" "${2:-}"
      ENTITLEMENTS_PATH="$2"
      shift 2
      ;;
    --python)
      require_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-sign)
      SKIP_SIGN=1
      shift
      ;;
    --include-failure-drills)
      INCLUDE_FAILURE_DRILLS=1
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

if [[ -z "${BUNDLE_PATH}" ]]; then
  echo "--bundle is required" >&2
  usage >&2
  exit 1
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

validate_inputs() {
  [[ -d "${BUNDLE_PATH}" ]] || die "bundle directory does not exist: ${BUNDLE_PATH}"
  [[ -f "${BUNDLE_PATH}/kernel" ]] || die "bundle missing kernel: ${BUNDLE_PATH}/kernel"
  [[ -f "${BUNDLE_PATH}/rootfs.img" ]] || die "bundle missing rootfs.img: ${BUNDLE_PATH}/rootfs.img"
  if [[ -n "${ENTITLEMENTS_PATH}" ]]; then
    [[ -f "${ENTITLEMENTS_PATH}" ]] || die "entitlements file does not exist: ${ENTITLEMENTS_PATH}"
  fi
}

build_helper_if_needed() {
  if [[ "${SKIP_BUILD}" -eq 1 ]]; then
    return 0
  fi
  if [[ -x "${HELPER_PATH}" ]]; then
    return 0
  fi
  run_cmd swift build --package-path "${REPO_ROOT}/tools/macos-vz-helper" -c debug
}

sign_helper_if_requested() {
  if [[ -z "${ENTITLEMENTS_PATH}" || "${SKIP_SIGN}" -eq 1 ]]; then
    return 0
  fi
  run_cmd codesign --force --sign - --entitlements "${ENTITLEMENTS_PATH}" "${HELPER_PATH}"
}

require_helper_binary() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  [[ -x "${HELPER_PATH}" ]] || die "helper binary is not executable: ${HELPER_PATH}"
}

prepare_socket_path() {
  if [[ -d "${SOCKET_PATH}" ]]; then
    die "helper socket path is a directory: ${SOCKET_PATH}"
  fi
  if [[ -L "${SOCKET_PATH}" ]]; then
    die "helper socket path already exists and is not a UNIX socket: ${SOCKET_PATH}"
  fi
  if [[ -S "${SOCKET_PATH}" ]]; then
    rm -f "${SOCKET_PATH}"
    return 0
  fi
  if [[ -e "${SOCKET_PATH}" ]]; then
    die "helper socket path already exists and is not a UNIX socket: ${SOCKET_PATH}"
  fi
}

stat_owner() {
  stat -f '%u' "$1" 2>/dev/null || stat -c '%u' "$1"
}

stat_mode() {
  stat -f '%Lp' "$1" 2>/dev/null || stat -c '%a' "$1"
}

directory_is_owner_private() {
  local directory="$1"
  local owner
  local mode
  [[ -d "${directory}" ]] || return 1
  owner="$(stat_owner "${directory}")" || return 1
  mode="$(stat_mode "${directory}")" || return 1
  [[ "${owner}" == "$(id -u)" ]] || return 1
  [[ $((8#${mode} & 077)) -eq 0 ]]
}

prepare_private_socket_parent() {
  local socket_dir
  socket_dir="$(dirname "${SOCKET_PATH}")"
  if [[ -L "${socket_dir}" ]]; then
    die "helper socket directory is a symlink: ${socket_dir}"
  fi
  if [[ -e "${socket_dir}" && ! -d "${socket_dir}" ]]; then
    die "helper socket directory is not a directory: ${socket_dir}"
  fi
  if [[ ! -e "${socket_dir}" ]]; then
    mkdir -p "${socket_dir}"
    chmod 700 "${socket_dir}"
  fi
  if ! directory_is_owner_private "${socket_dir}"; then
    die "helper socket directory must be owner-only: ${socket_dir}"
  fi
}

prepare_private_serial_log_dir() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  if [[ -L "${SERIAL_LOG_DIR}" ]]; then
    die "serial log directory is a symlink: ${SERIAL_LOG_DIR}"
  fi
  if [[ -e "${SERIAL_LOG_DIR}" && ! -d "${SERIAL_LOG_DIR}" ]]; then
    die "serial log path is not a directory: ${SERIAL_LOG_DIR}"
  fi
  if [[ -d "${SERIAL_LOG_DIR}" ]]; then
    if ! directory_is_owner_private "${SERIAL_LOG_DIR}"; then
      die "serial log directory must be owner-only: ${SERIAL_LOG_DIR}"
    fi
    return 0
  fi
  mkdir -p "${SERIAL_LOG_DIR}"
  chmod 700 "${SERIAL_LOG_DIR}"
  if ! directory_is_owner_private "${SERIAL_LOG_DIR}"; then
    die "serial log directory must be owner-only: ${SERIAL_LOG_DIR}"
  fi
}

prepare_runtime_paths() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  prepare_private_socket_parent
  prepare_socket_path
  prepare_private_serial_log_dir
}

helper_pid_file_path() {
  local socket_dir
  socket_dir="$(dirname "${SOCKET_PATH}")"
  printf '%s/helper.pid' "${socket_dir}"
}

record_helper_pid() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  HELPER_PID_FILE="$(helper_pid_file_path)"
  printf '%s\n' "${HELPER_PID}" > "${HELPER_PID_FILE}"
  chmod 600 "${HELPER_PID_FILE}" 2>/dev/null || true
}

current_helper_pid() {
  local candidate=""
  local pid_file="${HELPER_PID_FILE:-$(helper_pid_file_path)}"
  if [[ -f "${pid_file}" && ! -L "${pid_file}" ]]; then
    candidate="$(tr -d '[:space:]' < "${pid_file}" 2>/dev/null || true)"
    if [[ "${candidate}" =~ ^[1-9][0-9]*$ ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi
  printf '%s\n' "${HELPER_PID}"
}

run_helper_daemon_smoke() {
  run_cmd env \
    TLDW_SANDBOX_MACOS_HELPER_DAEMON_SMOKE=1 \
    TLDW_SANDBOX_VZ_LINUX_BUNDLE_SMOKE=1 \
    TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH="${BUNDLE_PATH}" \
    TLDW_SANDBOX_MACOS_HELPER_BINARY="${HELPER_PATH}" \
    TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR="${SERIAL_LOG_DIR}" \
    "${PYTHON_BIN}" -m pytest \
    "${REPO_ROOT}/tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py" \
    -q -rs
}

start_helper_for_real_e2e() {
  print_cmd env \
    TLDW_SANDBOX_MACOS_HELPER_SOCKET="${SOCKET_PATH}" \
    TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR="${SERIAL_LOG_DIR}" \
    "${HELPER_PATH}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  prepare_private_socket_parent
  prepare_socket_path
  env \
    TLDW_SANDBOX_MACOS_HELPER_SOCKET="${SOCKET_PATH}" \
    TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR="${SERIAL_LOG_DIR}" \
    "${HELPER_PATH}" \
    > "${SERIAL_LOG_DIR}/helper.stdout.log" \
    2> "${SERIAL_LOG_DIR}/helper.stderr.log" &
  HELPER_PID="$!"
  record_helper_pid
}

wait_for_helper_socket() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  # Test harnesses may run where AF_UNIX socket binding is denied.
  if [[ "${TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT:-0}" == "1" ]]; then
    return 0
  fi
  local deadline=$((SECONDS + 10))
  while [[ "${SECONDS}" -lt "${deadline}" ]]; do
    if [[ -S "${SOCKET_PATH}" ]]; then
      return 0
    fi
    if [[ -n "${HELPER_PID}" ]] && ! kill -0 "${HELPER_PID}" 2>/dev/null; then
      die "helper daemon exited before creating socket; see ${SERIAL_LOG_DIR}/helper.stderr.log"
    fi
    sleep 0.1
  done
  die "helper daemon did not create socket within timeout: ${SOCKET_PATH}"
}

run_real_vz_linux_pytest() {
  run_cmd env \
    TEST_MODE=0 \
    TLDW_SANDBOX_VZ_LINUX_E2E=1 \
    TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE="${BUNDLE_PATH}" \
    TLDW_SANDBOX_MACOS_HELPER_SOCKET="${SOCKET_PATH}" \
    SANDBOX_ENABLE_EXECUTION=1 \
    SANDBOX_BACKGROUND_EXECUTION=0 \
    "${PYTHON_BIN}" -m pytest \
    "${REPO_ROOT}/tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py" \
    "$@"
}

run_real_vz_linux_host_smoke() {
  run_real_vz_linux_pytest -m vz_linux_host_smoke -q -rs
}

run_real_vz_linux_failure_drills() {
  local helper_pid_file
  helper_pid_file="$(helper_pid_file_path)"
  run_cmd env \
    TEST_MODE=0 \
    TLDW_SANDBOX_VZ_LINUX_E2E=1 \
    TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE="${BUNDLE_PATH}" \
    TLDW_SANDBOX_MACOS_HELPER_SOCKET="${SOCKET_PATH}" \
    TLDW_SANDBOX_MACOS_HELPER_BINARY="${HELPER_PATH}" \
    TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR="${SERIAL_LOG_DIR}" \
    TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED=1 \
    TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_PID_FILE="${helper_pid_file}" \
    SANDBOX_ENABLE_EXECUTION=1 \
    SANDBOX_BACKGROUND_EXECUTION=0 \
    "${PYTHON_BIN}" -m pytest \
    "${REPO_ROOT}/tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py" \
    -m vz_linux_host_failure_drill -q -rs
}

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run: printing host smoke commands without starting helper or VMs"
fi

trap cleanup EXIT INT TERM
validate_inputs
cd "${REPO_ROOT}"
build_helper_if_needed
sign_helper_if_requested
require_helper_binary
prepare_runtime_paths
run_helper_daemon_smoke
start_helper_for_real_e2e
wait_for_helper_socket
run_real_vz_linux_host_smoke
if [[ "${INCLUDE_FAILURE_DRILLS}" -eq 1 ]]; then
  run_real_vz_linux_failure_drills
fi
