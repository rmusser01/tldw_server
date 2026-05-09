# VZ Linux Helper Restart Recovery Drill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a manual host-gated `vz_linux` helper restart recovery drill that proves stale same-session VM control state is cleared after the macOS helper process is restarted.

**Architecture:** Keep `run-host-e2e-smoke.sh` as the lower-level helper lifecycle owner and add a private restart lease only when failure drills are enabled. The real-host pytest drill uses that lease to stop the current helper, start a replacement helper on the same socket, update the pid file for shell cleanup, then run a second same-session command and assert a new VM is used. `vz-helperctl.py smoke` remains the preferred operator wrapper and must forward `--include-failure-drills`.

**Tech Stack:** Bash, Python 3.11, pytest, macOS AF_UNIX sockets, existing `MacOSVirtualizationHelperClient`, existing sandbox service/session APIs.

---

## Scope Check

This is one focused PR. It touches one operator shell script, one operator wrapper, one real-host pytest module, focused tests, and operator docs. It must not add launchd management, host reboot automation, network changes, helper protocol changes, or broad repair mutation.

## File Map

- Modify: `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`
  - Owns lower-level helper lifecycle and restart lease env for failure drills.
- Modify: `tools/vz-linux-image/tests/test_host_e2e_smoke_script.py`
  - Verifies dry-run lease visibility, default non-visibility, and cleanup of replacement helper PID.
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
  - Adds `smoke --include-failure-drills` pass-through to the lower-level script.
- Modify: `tools/macos-vz-helper/tests/test_vz_helperctl.py`
  - Verifies wrapper dry-run pass-through.
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`
  - Adds restart-lease validation helpers and the real host-gated helper restart drill.
- Modify: `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
  - Adds helper restart recovery as a manual failure-drill criterion.
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
  - Documents that manual failure drills now cover stale VM termination and helper restart recovery.
- Optional Modify: `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
  - Only if the current gaps language needs to distinguish helper restart drill coverage from host reboot/destructive repair gaps.
- Modify: `backlog/tasks/task-150 - Add-manual-VZ-Linux-helper-restart-recovery-drill.md`
  - Track implementation notes and verification.

## Task 1: Add Smoke Script Restart Lease

**Files:**
- Modify: `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`
- Modify: `tools/vz-linux-image/tests/test_host_e2e_smoke_script.py`

- [ ] **Step 1: Write failing dry-run tests for restart lease env**

Add to `tools/vz-linux-image/tests/test_host_e2e_smoke_script.py`:

```python
def test_host_e2e_smoke_script_default_dry_run_omits_restart_lease(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"

    result = _run_smoke_script(
        "--dry-run",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert "TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED" not in result.stdout
    assert "TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_PID_FILE" not in result.stdout


def test_host_e2e_smoke_script_failure_drill_dry_run_includes_restart_lease(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"

    result = _run_smoke_script(
        "--dry-run",
        "--include-failure-drills",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert "TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED=1" in result.stdout
    assert "TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_PID_FILE=" in result.stdout
    assert "/helper.pid" in result.stdout
    assert f"TLDW_SANDBOX_MACOS_HELPER_BINARY={helper}" in result.stdout
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tools/vz-linux-image/tests/test_host_e2e_smoke_script.py::test_host_e2e_smoke_script_default_dry_run_omits_restart_lease \
  tools/vz-linux-image/tests/test_host_e2e_smoke_script.py::test_host_e2e_smoke_script_failure_drill_dry_run_includes_restart_lease \
  -q
```

Expected: first test passes or remains neutral; second test fails because restart lease env is not printed.

- [ ] **Step 3: Implement minimal restart lease env**

In `run-host-e2e-smoke.sh`:

```bash
HELPER_PID=""
HELPER_PID_FILE=""

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
```

After `HELPER_PID="$!"` in `start_helper_for_real_e2e`, call `record_helper_pid`.

Change `run_real_vz_linux_failure_drills` to:

```bash
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
```

Keep `run_real_vz_linux_pytest` unchanged for baseline smoke so restart lease env is not present by default.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tools/vz-linux-image/tests/test_host_e2e_smoke_script.py::test_host_e2e_smoke_script_default_dry_run_omits_restart_lease \
  tools/vz-linux-image/tests/test_host_e2e_smoke_script.py::test_host_e2e_smoke_script_failure_drill_dry_run_includes_restart_lease \
  -q
```

Expected: both pass.

- [ ] **Step 5: Write failing cleanup test for replacement helper PID**

Add a test that proves cleanup follows the updated pid file:

```python
def test_host_e2e_smoke_script_cleanup_uses_replacement_helper_pid(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    marker = tmp_path / "replacement_pid.txt"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import os, subprocess, sys, time\n"
        "if sys.argv[1:3] != ['-m', 'pytest']:\n"
        "    sys.exit(2)\n"
        "if 'vz_linux_host_failure_drill' in sys.argv:\n"
        "    pid_file = os.environ['TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_PID_FILE']\n"
        "    marker = os.environ['TLDW_TEST_REPLACEMENT_PID_MARKER']\n"
        "    proc = subprocess.Popen([os.environ['TLDW_SANDBOX_MACOS_HELPER_BINARY']])\n"
        "    open(pid_file, 'w', encoding='utf-8').write(str(proc.pid) + '\\n')\n"
        "    open(marker, 'w', encoding='utf-8').write(str(proc.pid) + '\\n')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    fake_helper.write_text(
        "#!/usr/bin/env python3\n"
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    result = _run_smoke_script(
        "--bundle", str(bundle),
        "--helper", str(fake_helper),
        "--python", str(fake_python),
        "--skip-build",
        "--skip-sign",
        "--include-failure-drills",
        env_overrides={
            "TMPDIR": str(tmp_dir),
            "TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1",
            "TLDW_TEST_REPLACEMENT_PID_MARKER": str(marker),
        },
    )

    assert result.returncode == 0, result.stderr
    replacement_pid = int(marker.read_text(encoding="utf-8").strip())
    with pytest.raises(ProcessLookupError):
        os.kill(replacement_pid, 0)
```

If `os.kill(pid, 0)` behavior is platform-sensitive in this environment, adjust the assertion to poll briefly and still require `ProcessLookupError` as the success condition.

- [ ] **Step 6: Run cleanup test to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tools/vz-linux-image/tests/test_host_e2e_smoke_script.py::test_host_e2e_smoke_script_cleanup_uses_replacement_helper_pid \
  -q
```

Expected: FAIL because cleanup only tracks the original shell `HELPER_PID`.

- [ ] **Step 7: Implement cleanup pid-file handoff**

In `cleanup`, prefer the current pid file:

```bash
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
```

Keep pid-file deletion out of cleanup for this PR unless a test proves it is needed; the runtime directory is already temporary and useful for failure logs.

- [ ] **Step 8: Run script tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q
bash -n tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
```

Expected: existing script tests pass with the expected skip; bash syntax check exits 0.

- [ ] **Step 9: Commit Task 1**

```bash
git add tools/vz-linux-image/scripts/run-host-e2e-smoke.sh \
  tools/vz-linux-image/tests/test_host_e2e_smoke_script.py
git commit -m "test(sandbox): add VZ helper restart lease wiring"
```

## Task 2: Forward Failure Drills Through `vz-helperctl smoke`

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Modify: `tools/macos-vz-helper/tests/test_vz_helperctl.py`

- [ ] **Step 1: Write failing wrapper pass-through test**

Add near existing smoke dry-run tests:

```python
def test_smoke_dry_run_forwards_failure_drills(tmp_path, capsys):
    helperctl = load_helperctl()
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    code = helperctl.main([
        "smoke",
        "--dry-run",
        "--bundle",
        str(bundle),
        "--include-failure-drills",
    ])

    captured = capsys.readouterr()
    CASE.assertEqual(code, 0)
    CASE.assertIn("--include-failure-drills", captured.out)
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tools/macos-vz-helper/tests/test_vz_helperctl.py::test_smoke_dry_run_forwards_failure_drills -q
```

Expected: FAIL because the parser does not accept the flag.

- [ ] **Step 3: Implement pass-through**

In `smoke_helper`, add parameter:

```python
include_failure_drills: bool = False,
```

Append:

```python
if include_failure_drills:
    argv.append("--include-failure-drills")
```

In `_smoke_command`, pass `include_failure_drills=args.include_failure_drills`.

In `build_parser`, add:

```python
smoke.add_argument("--include-failure-drills", action="store_true")
```

- [ ] **Step 4: Run wrapper tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tools/macos-vz-helper/tests/test_vz_helperctl.py::test_smoke_dry_run_forwards_failure_drills \
  tools/macos-vz-helper/tests/test_vz_helperctl.py::test_smoke_dry_run_delegates_to_host_smoke_script \
  tools/macos-vz-helper/tests/test_vz_helperctl.py::test_helperctl_executable_smoke_dry_run_works \
  -q
```

Expected: all pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add tools/macos-vz-helper/scripts/vz-helperctl.py \
  tools/macos-vz-helper/tests/test_vz_helperctl.py
git commit -m "feat(sandbox): forward helper failure drills from helperctl"
```

## Task 3: Add Real-Host Helper Restart Drill

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`

- [ ] **Step 1: Add helper restart lease model and validation tests**

Add imports:

```python
import dataclasses
import signal
import socket
import subprocess
import time
```

Add helper types/functions near `_require_vz_linux_real_host_e2e`:

```python
@dataclasses.dataclass(frozen=True)
class _HelperRestartLease:
    helper_path: Path
    socket_path: Path
    serial_log_dir: Path
    pid_file: Path


def _require_helper_restart_lease() -> _HelperRestartLease:
    if not is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED")):
        pytest.skip("Set TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED=1 to enable helper restart drill")
    helper_text = str(os.getenv("TLDW_SANDBOX_MACOS_HELPER_BINARY") or "").strip()
    socket_text = str(os.getenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET") or "").strip()
    serial_log_text = str(os.getenv("TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR") or "").strip()
    pid_file_text = str(os.getenv("TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_PID_FILE") or "").strip()
    if not helper_text or not socket_text or not serial_log_text or not pid_file_text:
        pytest.skip("helper restart drill requires helper binary, socket, serial log dir, and pid file env")
    return _HelperRestartLease(
        helper_path=Path(helper_text).expanduser(),
        socket_path=Path(socket_text).expanduser(),
        serial_log_dir=Path(serial_log_text).expanduser(),
        pid_file=Path(pid_file_text).expanduser(),
    )
```

Add unit tests:

```python
def test_helper_restart_lease_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED", raising=False)
    with pytest.raises(pytest.skip.Exception, match="HELPER_RESTART_ALLOWED"):
        _require_helper_restart_lease()


def test_helper_restart_pid_file_rejects_symlink(tmp_path: Path) -> None:
    helper = tmp_path / "macos-vz-helper"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o755)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    target = runtime_dir / "target.pid"
    target.write_text("1234\n", encoding="utf-8")
    pid_file = runtime_dir / "helper.pid"
    pid_file.symlink_to(target)
    lease = _HelperRestartLease(helper, runtime_dir / "helper.sock", runtime_dir / "serial", pid_file)

    with pytest.raises(pytest.fail.Exception, match="pid file"):
        _read_valid_restart_pid(lease, process_lookup=lambda _pid: str(helper))
```

The second test requires `_read_valid_restart_pid` to be defined in the next step.

- [ ] **Step 2: Run validation tests to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py::test_helper_restart_lease_requires_explicit_opt_in \
  tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py::test_helper_restart_pid_file_rejects_symlink \
  -q
```

Expected: opt-in test may pass once helper exists; symlink test fails because `_read_valid_restart_pid` does not exist.

- [ ] **Step 3: Implement pid/process validation helpers**

Add:

```python
def _lookup_process_command(pid: int) -> str | None:
    completed = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    command = completed.stdout.strip()
    return command or None


def _read_valid_restart_pid(
    lease: _HelperRestartLease,
    *,
    process_lookup=_lookup_process_command,
) -> int:
    socket_dir = lease.socket_path.parent.resolve()
    try:
        pid_parent = lease.pid_file.parent.resolve()
    except OSError as exc:
        pytest.fail(f"helper restart pid file parent is invalid: {exc}")
    if pid_parent != socket_dir:
        pytest.fail("helper restart pid file must be inside the private socket directory")
    try:
        stat_result = lease.pid_file.lstat()
    except OSError as exc:
        pytest.fail(f"helper restart pid file is unavailable: {exc}")
    if not lease.pid_file.is_file() or lease.pid_file.is_symlink():
        pytest.fail("helper restart pid file must be a regular non-symlink file")
    if stat_result.st_mode & 0o077:
        pytest.fail("helper restart pid file must be owner-only")
    raw_pid = lease.pid_file.read_text(encoding="utf-8").strip()
    if not raw_pid.isdigit() or int(raw_pid) <= 0:
        pytest.fail("helper restart pid file does not contain a positive PID")
    pid = int(raw_pid)
    command = process_lookup(pid)
    if command is None:
        pytest.skip("helper process exited before restart drill could stop it")
    if str(lease.helper_path) not in command and lease.helper_path.name not in command:
        pytest.fail("helper restart pid file points at a non-helper process")
    return pid
```

- [ ] **Step 4: Run validation tests to verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py::test_helper_restart_lease_requires_explicit_opt_in \
  tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py::test_helper_restart_pid_file_rejects_symlink \
  -q
```

Expected: both pass.

- [ ] **Step 5: Add restart helper function with unit guards**

Add helper:

```python
def _wait_for_helper_socket_unavailable(socket_path: Path, timeout_sec: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not socket_path.exists():
            return
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(0.2)
                client.connect(str(socket_path))
        except OSError:
            return
        time.sleep(0.05)
    pytest.fail(f"helper socket remained available after helper stop: {socket_path}")


def _wait_for_helper_ping(helper, timeout_sec: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            helper.ping()
            return
        except Exception as exc:  # intentionally broad for test harness readiness polling
            last_error = exc
            time.sleep(0.1)
    pytest.fail(f"replacement helper did not answer ping: {last_error}")
```

Add `_restart_helper_for_drill` that:

- reads the valid old pid
- sends SIGTERM with `os.kill(pid, signal.SIGTERM)`
- waits for old socket unavailable
- starts replacement via `subprocess.Popen([str(lease.helper_path)], env=env, stdout=..., stderr=...)`
- writes replacement pid to lease pid file with `0o600`
- waits for ping
- returns replacement process

Use `TLDW_SANDBOX_MACOS_HELPER_SOCKET`, `TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR`, and existing protocol env if needed.

- [ ] **Step 6: Add real helper restart drill test**

Add:

```python
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
@pytest.mark.vz_linux_host_failure_drill
def test_vz_linux_real_session_recreates_vm_after_helper_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_image = _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)
    lease = _require_helper_restart_lease()
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", raising=False)

    service = SandboxService()
    helper = VZLinuxRunner.helper_client_cls()
    session_id: str | None = None
    destroyed = False
    try:
        session = service.create_session(
            user_id="e2e-user",
            spec=SessionSpec(runtime=RuntimeType.vz_linux, base_image=base_image, network_policy="deny_all"),
            spec_version="1.0",
            idem_key=None,
            raw_body={"spec_version": "1.0", "runtime": "vz_linux", "base_image": base_image},
        )
        session_id = session.id
        first = service.start_run_scaffold(
            user_id="e2e-user",
            spec=RunSpec(
                session_id=session.id,
                runtime=RuntimeType.vz_linux,
                base_image=base_image,
                command=["/bin/echo", "restart-drill-first"],
                network_policy="deny_all",
            ),
            spec_version="1.0",
            idem_key=None,
            raw_body={"session_id": session.id, "runtime": "vz_linux"},
        )
        control_after_first = service._orch.get_vz_session_control(session.id)
        _expect(first.phase == RunPhase.completed, f"Expected first run completed, got {first.phase!r}")
        _expect(isinstance(control_after_first, dict), "Expected VZ session control after first run")
        first_vm_id = str(control_after_first.get("vm_id") or "").strip()
        _expect(bool(first_vm_id), f"Expected first VM id, got {control_after_first!r}")
        _expect(helper.get_vm_status(first_vm_id).healthy, f"Expected first VM healthy before restart: {first_vm_id!r}")

        _restart_helper_for_drill(lease)
        helper_after_restart = VZLinuxRunner.helper_client_cls()
        status_after_restart = helper_after_restart.get_vm_status(first_vm_id)
        _expect(not bool(status_after_restart.healthy), f"Expected old VM stale after helper restart: {status_after_restart!r}")

        second = service.start_run_scaffold(
            user_id="e2e-user",
            spec=RunSpec(
                session_id=session.id,
                runtime=RuntimeType.vz_linux,
                base_image=base_image,
                command=["/bin/echo", "restart-drill-second"],
                network_policy="deny_all",
            ),
            spec_version="1.0",
            idem_key=None,
            raw_body={"session_id": session.id, "runtime": "vz_linux"},
        )
        control_after_second = service._orch.get_vz_session_control(session.id)
        _expect(second.phase == RunPhase.completed, f"Expected second run completed, got {second.phase!r}")
        _expect(isinstance(control_after_second, dict), "Expected VZ session control after second run")
        second_vm_id = str(control_after_second.get("vm_id") or "").strip()
        _expect(bool(second_vm_id), f"Expected second VM id, got {control_after_second!r}")
        _expect(second_vm_id != first_vm_id, f"Expected helper restart to force fresh VM, got {first_vm_id!r}")
        _expect(service.destroy_session(session.id) is True, "Expected session destruction to succeed")
        destroyed = True
    finally:
        if session_id and not destroyed:
            service.destroy_session(session_id)
```

Adjust exact helper implementation if `HelperVMStatusReply` field access differs; existing tests use `getattr(..., "healthy", False)` if needed.

- [ ] **Step 7: Run host-gated marker locally to verify guarded skip**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py \
  -m vz_linux_host_failure_drill -q -rs
```

Expected locally without real env: selected failure drills skip with clear opt-in/base-image reasons, not collection errors.

- [ ] **Step 8: Commit Task 3**

```bash
git add tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py
git commit -m "test(sandbox): add VZ helper restart recovery drill"
```

## Task 4: Update Operator Docs And Policy

**Files:**
- Modify: `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Optional Modify: `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- Modify: `backlog/tasks/task-150 - Add-manual-VZ-Linux-helper-restart-recovery-drill.md`

- [ ] **Step 1: Update failure criteria policy**

In `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`, extend blocking regression criteria:

```markdown
- a manually requested helper restart drill cannot replace stale session-control
  VM state after the helper process is stopped and restarted through the smoke
  harness restart lease
```

- [ ] **Step 2: Update operator notes**

In `Docs/Sandbox/macos-runtime-operator-notes.md`, update Real Host E2E Smoke to say failure drills include:

- stale session VM after helper-side termination
- helper restart recovery through the smoke harness restart lease

Keep host reboot and launchd restart listed as manual/operator-only gaps.

- [ ] **Step 3: Update inventory only if wording is stale**

If `Docs/Sandbox/sandbox-runtime-capability-inventory.md` still says helper crash/restart remains wholly manual after this PR, narrow it to host reboot/destructive repair/stale socket/stuck readiness. Do not claim broad helper crash coverage beyond this explicit restart drill.

- [ ] **Step 4: Update Backlog task notes**

Record touched files, verification commands, host-gated skip status, and any unrun real VM drill reason.

- [ ] **Step 5: Commit Task 4**

```bash
git add Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md \
  Docs/Sandbox/macos-runtime-operator-notes.md \
  Docs/Sandbox/sandbox-runtime-capability-inventory.md \
  "backlog/tasks/task-150 - Add-manual-VZ-Linux-helper-restart-recovery-drill.md"
git commit -m "docs(sandbox): document helper restart recovery drill"
```

If inventory is not changed, omit it from `git add`.

## Task 5: Final Verification And PR Prep

**Files:**
- All touched files from Tasks 1-4.

- [ ] **Step 1: Run focused Python/shell tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tools/vz-linux-image/tests/test_host_e2e_smoke_script.py \
  tools/macos-vz-helper/tests/test_vz_helperctl.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py \
  -m "not vz_linux_host_smoke and not vz_linux_host_failure_drill" \
  -q
bash -n tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile \
  tools/macos-vz-helper/scripts/vz-helperctl.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py
```

Expected: focused tests pass, shell syntax exits 0, py_compile exits 0.

- [ ] **Step 2: Run host-gated drill selection locally**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py \
  -m vz_linux_host_failure_drill -q -rs
```

Expected without prepared real-host env: skips with clear reasons. On a prepared Apple silicon host, run the full operator command:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py smoke \
  --bundle /path/to/canonical/bundle \
  --entitlements /path/to/helper.entitlements \
  --include-failure-drills
```

- [ ] **Step 3: Run Bandit and diff checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit \
  -r tools/macos-vz-helper/scripts/vz-helperctl.py \
     tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py \
  -s B101 -f json -o /tmp/bandit_vz_helper_restart_recovery.json
git diff --check
```

Expected: Bandit reports zero new findings after excluding pytest `assert` noise; diff check has no output.

- [ ] **Step 4: Rebase and inspect final diff**

Run:

```bash
git fetch origin dev
git rebase origin/dev
git diff --stat origin/dev...HEAD
git log --oneline origin/dev..HEAD
```

Expected: clean rebase, diff limited to planned files.

- [ ] **Step 5: Update Backlog final summary**

Set TASK-150 acceptance criteria and DoD checked only after verification passes. Include any real-host drill that was skipped locally because a prepared host or bundle was not available.

- [ ] **Step 6: Push and create/update PR**

```bash
git push -u origin codex/sandbox-helper-restart-recovery-drills
gh pr create --base dev --head codex/sandbox-helper-restart-recovery-drills \
  --title "Add manual VZ helper restart recovery drill" \
  --body-file /tmp/vz-helper-restart-recovery-pr.md
```

PR body must include:

- summary of manual helper restart drill and restart lease
- host-independent tests run
- host-gated drill status and exact prepared-host command
- explicit note that host reboot, launchd bootstrap, networking, and broad repair generalization remain out of scope
