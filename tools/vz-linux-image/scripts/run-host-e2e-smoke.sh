#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${IMAGE_DIR}/../.." && pwd)"

SOURCE_BUNDLE_PATH=""
BUNDLE_PATH=""
DEFAULT_RUNTIME_ROOT="${TLDW_HOST_E2E_SMOKE_RUNTIME_ROOT:-/tmp}"
DEFAULT_RUNTIME_ROOT="${DEFAULT_RUNTIME_ROOT%/}"
DEFAULT_RUNTIME_DIR="${DEFAULT_RUNTIME_ROOT}/tvz-e2e-$$"
SOCKET_PATH="${DEFAULT_RUNTIME_DIR}/helper.sock"
SERIAL_LOG_DIR="${DEFAULT_RUNTIME_DIR}/serial"
IMAGE_STORE_ROOT=""
SMOKE_RUN_ID="host-smoke-$$"
PREPARE_SMOKE_BUNDLE_SCRIPT="${SCRIPT_DIR}/prepare-smoke-bundle.py"
EVIDENCE_DIR=""
HELPER_PATH="${REPO_ROOT}/tools/macos-vz-helper/.build/debug/macos-vz-helper"
ENTITLEMENTS_PATH=""
PYTHON_BIN=""
DRY_RUN=0
SKIP_BUILD=0
SKIP_SIGN=0
INCLUDE_FAILURE_DRILLS=0
HELPER_PID=""
HELPER_PID_FILE=""
EVIDENCE_FINALIZED=0
PHASE_RECORDS=()
EVIDENCE_FILE_NAMES=(
  "host-smoke-evidence.json"
  "source-bundle-hashes-before.txt"
  "source-bundle-hashes-after.txt"
  "run-bundle-hashes.txt"
  "runtime-paths.txt"
  "cleanup-status.txt"
)

usage() {
  cat <<'EOF'
Usage: run-host-e2e-smoke.sh --bundle PATH [options]

Build/sign/start the macOS Virtualization.framework helper, run the helper
daemon smoke, run real vz_linux ephemeral execution, verify same-session VM
reuse, verify recovery diagnostics plus dry-run repair planning, and stop the
helper.

Options:
  --bundle PATH          Canonical source vz_linux bundle directory (required);
                         VM stages use a disposable image-store run bundle
  --image-store-root PATH
                         Private image-store root for disposable smoke bundles
  --smoke-run-id ID      Image-store run id for the disposable smoke bundle
  --evidence-dir PATH    Directory for structured smoke evidence
                         (default: socket runtime directory/evidence)
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
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  if [[ -z "${HELPER_PID}" && -z "${HELPER_PID_FILE}" ]]; then
    return 0
  fi
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
      SOURCE_BUNDLE_PATH="$2"
      shift 2
      ;;
    --image-store-root)
      require_value "$1" "${2:-}"
      IMAGE_STORE_ROOT="$2"
      shift 2
      ;;
    --smoke-run-id)
      require_value "$1" "${2:-}"
      SMOKE_RUN_ID="$2"
      shift 2
      ;;
    --evidence-dir)
      require_value "$1" "${2:-}"
      EVIDENCE_DIR="$2"
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

if [[ -z "${SOURCE_BUNDLE_PATH}" ]]; then
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

if [[ -z "${IMAGE_STORE_ROOT}" ]]; then
  IMAGE_STORE_ROOT="$(dirname "${SOCKET_PATH}")/image-store"
fi

if [[ -z "${EVIDENCE_DIR}" ]]; then
  EVIDENCE_DIR="$(dirname "${SOCKET_PATH}")/evidence"
fi

validate_inputs() {
  [[ -d "${SOURCE_BUNDLE_PATH}" ]] || die "bundle directory does not exist: ${SOURCE_BUNDLE_PATH}"
  [[ -f "${SOURCE_BUNDLE_PATH}/kernel" ]] || die "bundle missing kernel: ${SOURCE_BUNDLE_PATH}/kernel"
  [[ -f "${SOURCE_BUNDLE_PATH}/rootfs.img" ]] || die "bundle missing rootfs.img: ${SOURCE_BUNDLE_PATH}/rootfs.img"
  if [[ -n "${ENTITLEMENTS_PATH}" ]]; then
    [[ -f "${ENTITLEMENTS_PATH}" ]] || die "entitlements file does not exist: ${ENTITLEMENTS_PATH}"
  fi
}

prepare_smoke_bundle() {
  local prepare_args=(
    "${PYTHON_BIN}" "${PREPARE_SMOKE_BUNDLE_SCRIPT}"
    --source-bundle "${SOURCE_BUNDLE_PATH}" \
    --store-root "${IMAGE_STORE_ROOT}" \
    --run-id "${SMOKE_RUN_ID}"
  )
  local prepare_status
  print_cmd "${prepare_args[@]}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    BUNDLE_PATH="$("${prepare_args[@]}" --print-path-only)"
    prepare_status="$?"
    if [[ "${prepare_status}" -ne 0 ]]; then
      return "${prepare_status}"
    fi
    return 0
  fi
  BUNDLE_PATH="$("${prepare_args[@]}")"
  prepare_status="$?"
  if [[ "${prepare_status}" -ne 0 ]]; then
    return "${prepare_status}"
  fi
  [[ -d "${BUNDLE_PATH}" ]] || die "prepared smoke bundle directory missing: ${BUNDLE_PATH}"
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

socket_probe_state() {
  local socket_path="$1"
  [[ -S "${socket_path}" ]] || return 1
  "${PYTHON_BIN}" -c '
import errno
import socket
import sys

try:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(0.2)
        client.connect(sys.argv[1])
except OSError as exc:
    if exc.errno in {errno.ECONNREFUSED, errno.ENOENT}:
        raise SystemExit(1)
    raise SystemExit(2)
raise SystemExit(0)
' "${socket_path}"
}

prepare_socket_path() {
  local socket_probe_status

  if [[ -d "${SOCKET_PATH}" ]]; then
    die "helper socket path is a directory: ${SOCKET_PATH}"
  fi
  if [[ -L "${SOCKET_PATH}" ]]; then
    die "helper socket path already exists and is not a UNIX socket: ${SOCKET_PATH}"
  fi
  if [[ -S "${SOCKET_PATH}" ]]; then
    if socket_probe_state "${SOCKET_PATH}"; then
      die "helper socket path is already in use: ${SOCKET_PATH}"
    else
      socket_probe_status=$?
      if [[ "${socket_probe_status}" -ne 1 ]]; then
        die "helper socket path could not be safely probed; refusing to remove: ${SOCKET_PATH}"
      fi
    fi
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

prepare_private_evidence_dir() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    if [[ -L "${EVIDENCE_DIR}" ]]; then
      die "evidence directory is a symlink: ${EVIDENCE_DIR}"
    fi
    if [[ -e "${EVIDENCE_DIR}" && ! -d "${EVIDENCE_DIR}" ]]; then
      die "evidence path is not a directory: ${EVIDENCE_DIR}"
    fi
    if [[ -d "${EVIDENCE_DIR}" ]] && ! directory_is_owner_private "${EVIDENCE_DIR}"; then
      die "evidence directory must be owner-only: ${EVIDENCE_DIR}"
    fi
    return 0
  fi
  if [[ -L "${EVIDENCE_DIR}" ]]; then
    die "evidence directory is a symlink: ${EVIDENCE_DIR}"
  fi
  if [[ -e "${EVIDENCE_DIR}" && ! -d "${EVIDENCE_DIR}" ]]; then
    die "evidence path is not a directory: ${EVIDENCE_DIR}"
  fi
  if [[ -d "${EVIDENCE_DIR}" ]]; then
    if ! directory_is_owner_private "${EVIDENCE_DIR}"; then
      die "evidence directory must be owner-only: ${EVIDENCE_DIR}"
    fi
    return 0
  fi
  mkdir -p -m 700 "${EVIDENCE_DIR}"
  chmod 700 "${EVIDENCE_DIR}"
  if ! directory_is_owner_private "${EVIDENCE_DIR}"; then
    die "evidence directory must be owner-only: ${EVIDENCE_DIR}"
  fi
}

print_evidence_plan() {
  echo "evidence directory: ${EVIDENCE_DIR}"
  local evidence_file
  for evidence_file in "${EVIDENCE_FILE_NAMES[@]}"; do
    echo "evidence file: ${EVIDENCE_DIR}/${evidence_file}"
  done
  print_evidence_env_hint
}

print_evidence_env_hint() {
  if [[ -n "${EVIDENCE_DIR:-}" ]]; then
    printf 'export TLDW_SANDBOX_VZ_EVIDENCE_DIR=%q\n' "${EVIDENCE_DIR}"
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
  local pid_file
  pid_file="$(helper_pid_file_path)"
  (umask 077; printf '%s\n' "${HELPER_PID}" > "${pid_file}")
  chmod 600 "${pid_file}" 2>/dev/null || true
  HELPER_PID_FILE="${pid_file}"
}

current_helper_pid() {
  local candidate=""
  if [[ -n "${HELPER_PID_FILE}" && -f "${HELPER_PID_FILE}" && ! -L "${HELPER_PID_FILE}" ]]; then
    candidate="$(tr -d '[:space:]' < "${HELPER_PID_FILE}" 2>/dev/null || true)"
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

timestamp_utc() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

mark_phase_started() {
  local phase="$1"
  PHASE_RECORDS+=("${phase}|started|0|$(timestamp_utc)")
}

mark_phase_ok() {
  local phase="$1"
  PHASE_RECORDS+=("${phase}|ok|0|$(timestamp_utc)")
}

mark_phase_failed() {
  local phase="$1"
  local status="$2"
  PHASE_RECORDS+=("${phase}|failed|${status}|$(timestamp_utc)")
}

run_phase() {
  local phase="$1"
  shift
  local status
  mark_phase_started "${phase}"
  set +e
  "$@"
  status="$?"
  set -e
  if [[ "${status}" -eq 0 ]]; then
    mark_phase_ok "${phase}"
  else
    mark_phase_failed "${phase}" "${status}"
  fi
  return "${status}"
}

phase_records_text() {
  local record
  for record in "${PHASE_RECORDS[@]}"; do
    printf '%s\n' "${record}"
  done
}

evidence_python_supports_stdlib() {
  local candidate="$1"
  [[ -n "${candidate}" && -x "${candidate}" ]] || return 1
  "${candidate}" -c 'import hashlib, json, pathlib' >/dev/null 2>&1
}

evidence_python_bin() {
  local candidate
  candidate="$(command -v python3 2>/dev/null || true)"
  if evidence_python_supports_stdlib "${candidate}"; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  candidate="$(command -v python 2>/dev/null || true)"
  if evidence_python_supports_stdlib "${candidate}"; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  if evidence_python_supports_stdlib "${PYTHON_BIN}"; then
    printf '%s\n' "${PYTHON_BIN}"
    return 0
  fi
  return 1
}

hash_bundle_files() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  local bundle_root="$1"
  local output_path="$2"
  local evidence_python
  if [[ -z "${bundle_root}" ]]; then
    mkdir -p -m 700 "$(dirname "${output_path}")"
    {
      printf '# sha256  relative_path\n'
      printf '# missing: <empty bundle path>\n'
    } > "${output_path}"
    chmod 600 "${output_path}"
    return 0
  fi
  evidence_python="$(evidence_python_bin)" || return 1
  EVIDENCE_HASH_ROOT="${bundle_root}" \
    EVIDENCE_HASH_OUTPUT="${output_path}" \
    "${evidence_python}" - <<'PY'
from __future__ import annotations

import hashlib
import os
from pathlib import Path

root = Path(os.environ["EVIDENCE_HASH_ROOT"])
output = Path(os.environ["EVIDENCE_HASH_OUTPUT"])
output.parent.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

with output.open("w", encoding="utf-8") as handle:
    handle.write("# sha256  relative_path\n")
    if not root.is_dir():
        handle.write(f"# missing: {root}\n")
        raise SystemExit(0)
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        digest = sha256_file(path)
        handle.write(f"{digest}  {path.relative_to(root)}\n")
PY
  chmod 600 "${output_path}"
}

write_runtime_paths() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  local evidence_python
  evidence_python="$(evidence_python_bin)" || return 1
  EVIDENCE_RUNTIME_OUTPUT="${EVIDENCE_DIR}/runtime-paths.txt" \
    EVIDENCE_SOURCE_BUNDLE="${SOURCE_BUNDLE_PATH}" \
    EVIDENCE_RUN_BUNDLE="${BUNDLE_PATH}" \
    EVIDENCE_IMAGE_STORE_ROOT="${IMAGE_STORE_ROOT}" \
    EVIDENCE_SOCKET_PATH="${SOCKET_PATH}" \
    EVIDENCE_SERIAL_LOG_DIR="${SERIAL_LOG_DIR}" \
    EVIDENCE_DIR_PATH="${EVIDENCE_DIR}" \
    EVIDENCE_HELPER_PATH="${HELPER_PATH}" \
    EVIDENCE_HELPER_PID_FILE="${HELPER_PID_FILE:-$(helper_pid_file_path)}" \
    "${evidence_python}" - <<'PY'
from __future__ import annotations

import os
import stat
from pathlib import Path

output = Path(os.environ["EVIDENCE_RUNTIME_OUTPUT"])
paths = {
    "source_bundle_path": os.environ["EVIDENCE_SOURCE_BUNDLE"],
    "run_bundle_path": os.environ["EVIDENCE_RUN_BUNDLE"],
    "image_store_root": os.environ["EVIDENCE_IMAGE_STORE_ROOT"],
    "socket_path": os.environ["EVIDENCE_SOCKET_PATH"],
    "serial_log_dir": os.environ["EVIDENCE_SERIAL_LOG_DIR"],
    "evidence_dir": os.environ["EVIDENCE_DIR_PATH"],
    "helper_path": os.environ["EVIDENCE_HELPER_PATH"],
    "helper_pid_file": os.environ["EVIDENCE_HELPER_PID_FILE"],
}

def metadata(path_text: str) -> str:
    path = Path(path_text)
    try:
        st = path.lstat()
    except OSError:
        return "exists=false"
    mode = stat.S_IMODE(st.st_mode)
    return f"exists=true owner={st.st_uid} mode={mode:o} size={st.st_size}"

with output.open("w", encoding="utf-8") as handle:
    for key, value in paths.items():
        handle.write(f"{key}={value}\n")
        if value:
            handle.write(f"{key}_metadata={metadata(value)}\n")
PY
  chmod 600 "${EVIDENCE_DIR}/runtime-paths.txt"
}

CLEANUP_HELPER_PID=""
CLEANUP_HELPER_RUNNING="false"
CLEANUP_SOCKET_PRESENT="false"
CLEANUP_STATUS="0"

write_cleanup_status() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  local final_exit="$1"
  local cleanup_status="$2"
  local pid
  pid="$(current_helper_pid)"
  CLEANUP_HELPER_PID="${pid}"
  CLEANUP_STATUS="${cleanup_status}"
  CLEANUP_HELPER_RUNNING="false"
  CLEANUP_SOCKET_PRESENT="false"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    CLEANUP_HELPER_RUNNING="true"
  fi
  if [[ -S "${SOCKET_PATH}" ]]; then
    CLEANUP_SOCKET_PRESENT="true"
  fi
  {
    printf 'final_exit_code=%s\n' "${final_exit}"
    printf 'cleanup_status=%s\n' "${cleanup_status}"
    printf 'helper_pid=%s\n' "${pid}"
    printf 'helper_running_after_cleanup=%s\n' "${CLEANUP_HELPER_RUNNING}"
    printf 'socket_path=%s\n' "${SOCKET_PATH}"
    printf 'socket_present_after_cleanup=%s\n' "${CLEANUP_SOCKET_PRESENT}"
  } > "${EVIDENCE_DIR}/cleanup-status.txt"
  chmod 600 "${EVIDENCE_DIR}/cleanup-status.txt"
}

write_json_evidence() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  local final_exit="$1"
  local phase_records="${2:-$(phase_records_text)}"
  local evidence_python
  evidence_python="$(evidence_python_bin)" || return 1
  EVIDENCE_JSON_OUTPUT="${EVIDENCE_DIR}/host-smoke-evidence.json" \
    EVIDENCE_CREATED_AT="$(timestamp_utc)" \
    EVIDENCE_PHASE_RECORDS="${phase_records}" \
    EVIDENCE_SOURCE_BUNDLE="${SOURCE_BUNDLE_PATH}" \
    EVIDENCE_RUN_BUNDLE="${BUNDLE_PATH}" \
    EVIDENCE_IMAGE_STORE_ROOT="${IMAGE_STORE_ROOT}" \
    EVIDENCE_SMOKE_RUN_ID="${SMOKE_RUN_ID}" \
    EVIDENCE_SOCKET_PATH="${SOCKET_PATH}" \
    EVIDENCE_SERIAL_LOG_DIR="${SERIAL_LOG_DIR}" \
    EVIDENCE_DIR_PATH="${EVIDENCE_DIR}" \
    EVIDENCE_HELPER_PATH="${HELPER_PATH}" \
    EVIDENCE_HELPER_PID_FILE="${HELPER_PID_FILE:-$(helper_pid_file_path)}" \
    EVIDENCE_SKIP_BUILD="${SKIP_BUILD}" \
    EVIDENCE_SKIP_SIGN="${SKIP_SIGN}" \
    EVIDENCE_INCLUDE_FAILURE_DRILLS="${INCLUDE_FAILURE_DRILLS}" \
    EVIDENCE_FINAL_EXIT="${final_exit}" \
    EVIDENCE_CLEANUP_STATUS="${CLEANUP_STATUS}" \
    EVIDENCE_CLEANUP_HELPER_PID="${CLEANUP_HELPER_PID}" \
    EVIDENCE_CLEANUP_HELPER_RUNNING="${CLEANUP_HELPER_RUNNING}" \
    EVIDENCE_CLEANUP_SOCKET_PRESENT="${CLEANUP_SOCKET_PRESENT}" \
    "${evidence_python}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def bool_from_flag(value: str) -> bool:
    return value in {"1", "true", "TRUE", "True", "yes", "YES", "Yes"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    digest = sha256_file(path)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest}


def phase_payload(records_text: str) -> dict[str, dict[str, Any]]:
    phases: dict[str, dict[str, Any]] = {}
    for record in records_text.splitlines():
        parts = record.split("|", 3)
        if len(parts) != 4:
            continue
        phase, status, exit_code, timestamp = parts
        try:
            parsed_exit = int(exit_code)
        except ValueError:
            parsed_exit = 0
        phases[phase] = {"status": status, "exit_code": parsed_exit, "timestamp": timestamp}
    return phases


evidence_dir = Path(os.environ["EVIDENCE_DIR_PATH"])
serial_dir = Path(os.environ["EVIDENCE_SERIAL_LOG_DIR"])
evidence_files = {
    name: str(evidence_dir / name)
    for name in (
        "host-smoke-evidence.json",
        "source-bundle-hashes-before.txt",
        "source-bundle-hashes-after.txt",
        "run-bundle-hashes.txt",
        "runtime-paths.txt",
        "cleanup-status.txt",
    )
}
logs = []
if serial_dir.is_dir():
    for path in sorted(serial_dir.glob("*.log")):
        if path.is_file() and not path.is_symlink():
            try:
                logs.append(file_metadata(path))
            except OSError:
                pass

payload = {
    "schema_version": 1,
    "created_at": os.environ["EVIDENCE_CREATED_AT"],
    "source_bundle_path": os.environ["EVIDENCE_SOURCE_BUNDLE"],
    "run_bundle_path": os.environ["EVIDENCE_RUN_BUNDLE"],
    "image_store_root": os.environ["EVIDENCE_IMAGE_STORE_ROOT"],
    "smoke_run_id": os.environ["EVIDENCE_SMOKE_RUN_ID"],
    "socket_path": os.environ["EVIDENCE_SOCKET_PATH"],
    "serial_log_dir": os.environ["EVIDENCE_SERIAL_LOG_DIR"],
    "evidence_dir": os.environ["EVIDENCE_DIR_PATH"],
    "helper_path": os.environ["EVIDENCE_HELPER_PATH"],
    "helper_pid_file": os.environ["EVIDENCE_HELPER_PID_FILE"],
    "skip_build": bool_from_flag(os.environ["EVIDENCE_SKIP_BUILD"]),
    "skip_sign": bool_from_flag(os.environ["EVIDENCE_SKIP_SIGN"]),
    "include_failure_drills": bool_from_flag(os.environ["EVIDENCE_INCLUDE_FAILURE_DRILLS"]),
    "final_exit_code": int(os.environ["EVIDENCE_FINAL_EXIT"]),
    "phases": phase_payload(os.environ.get("EVIDENCE_PHASE_RECORDS", "")),
    "cleanup": {
        "status": int(os.environ["EVIDENCE_CLEANUP_STATUS"]),
        "helper_pid": os.environ["EVIDENCE_CLEANUP_HELPER_PID"],
        "helper_running_after_cleanup": bool_from_flag(os.environ["EVIDENCE_CLEANUP_HELPER_RUNNING"]),
        "socket_present_after_cleanup": bool_from_flag(os.environ["EVIDENCE_CLEANUP_SOCKET_PRESENT"]),
    },
    "evidence_files": evidence_files,
    "log_artifacts": logs,
}

output = Path(os.environ["EVIDENCE_JSON_OUTPUT"])
temporary_output = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary_output.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary_output.chmod(0o600)
os.replace(temporary_output, output)
PY
}

finalize_evidence() {
  local final_exit="$1"
  local cleanup_status="$2"
  local status=0
  local old_umask
  if [[ "${DRY_RUN}" -eq 1 || "${EVIDENCE_FINALIZED}" -eq 1 ]]; then
    return 0
  fi
  EVIDENCE_FINALIZED=1
  old_umask="$(umask)"
  umask 077
  mark_phase_started "evidence_finalize"
  prepare_private_evidence_dir || status="$?"
  if [[ "${status}" -eq 0 ]]; then
    hash_bundle_files "${SOURCE_BUNDLE_PATH}" "${EVIDENCE_DIR}/source-bundle-hashes-after.txt" || status="$?"
  fi
  if [[ "${status}" -eq 0 ]]; then
    hash_bundle_files "${BUNDLE_PATH}" "${EVIDENCE_DIR}/run-bundle-hashes.txt" || status="$?"
  fi
  if [[ "${status}" -eq 0 ]]; then
    write_runtime_paths || status="$?"
  fi
  if [[ "${status}" -eq 0 ]]; then
    write_cleanup_status "${final_exit}" "${cleanup_status}" || status="$?"
  fi
  if [[ "${status}" -eq 0 ]]; then
    local evidence_success_record
    evidence_success_record="evidence_finalize|ok|0|$(timestamp_utc)"
    write_json_evidence "${final_exit}" "$(phase_records_text; printf '%s\n' "${evidence_success_record}")" || status="$?"
    if [[ "${status}" -eq 0 ]]; then
      PHASE_RECORDS+=("${evidence_success_record}")
      print_evidence_env_hint
    fi
  fi
  if [[ "${status}" -ne 0 ]]; then
    mark_phase_failed "evidence_finalize" "${status}"
  fi
  umask "${old_umask}"
  return "${status}"
}

finalize() {
  local status="$?"
  local cleanup_status=0
  local evidence_status=0
  trap - EXIT INT TERM
  set +e
  mark_phase_started "cleanup"
  cleanup
  cleanup_status="$?"
  if [[ "${cleanup_status}" -eq 0 ]]; then
    mark_phase_ok "cleanup"
  else
    mark_phase_failed "cleanup" "${cleanup_status}"
  fi
  finalize_evidence "${status}" "${cleanup_status}"
  evidence_status="$?"
  if [[ "${status}" -eq 0 && "${evidence_status}" -ne 0 ]]; then
    exit "${evidence_status}"
  fi
  exit "${status}"
}

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run: printing host smoke commands without starting helper or VMs"
fi

trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_phase validate_inputs validate_inputs
cd "${REPO_ROOT}"
run_phase build_helper build_helper_if_needed
run_phase sign_helper sign_helper_if_requested
run_phase require_helper_binary require_helper_binary
run_phase prepare_runtime_paths prepare_runtime_paths
run_phase prepare_evidence_dir prepare_private_evidence_dir
if [[ "${DRY_RUN}" -eq 1 ]]; then
  print_evidence_plan
fi
run_phase source_hash_before hash_bundle_files "${SOURCE_BUNDLE_PATH}" "${EVIDENCE_DIR}/source-bundle-hashes-before.txt"
run_phase prepare_smoke_bundle prepare_smoke_bundle
run_phase helper_daemon_smoke run_helper_daemon_smoke
run_phase start_helper start_helper_for_real_e2e
run_phase wait_for_helper_socket wait_for_helper_socket
run_phase real_host_smoke run_real_vz_linux_host_smoke
if [[ "${INCLUDE_FAILURE_DRILLS}" -eq 1 ]]; then
  run_phase failure_drills run_real_vz_linux_failure_drills
fi
