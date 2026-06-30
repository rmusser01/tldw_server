# VZ Helper Lifecycle Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an operator-first lifecycle command for the macOS `vz_linux` helper, harden helper socket startup safety, and document stable helper paths without introducing automatic service installation.

**Architecture:** Keep the Python sandbox service as runtime authority and add a separate operator CLI under `tools/macos-vz-helper/scripts/` for helper process lifecycle. Harden the Swift helper so direct launches and launchd launches enforce the same socket safety guarantees as the wrapper. Reuse the existing host E2E smoke script instead of duplicating real VM execution logic.

**Tech Stack:** Python 3 standard library, pytest, Swift Package Manager, Swift Testing, macOS `codesign`/`launchd` plist formats, existing `MacOSVirtualizationHelperClient`.

---

## Source References

- Spec: `Docs/superpowers/specs/2026-04-29-vz-helper-lifecycle-hardening-design.md`
- Doctrine: `Docs/Sandbox/sandbox-architecture-doctrine.md`
- Helper client: `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
- Swift helper entrypoint: `tools/macos-vz-helper/Sources/main.swift`
- Swift socket server: `tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift`
- Existing host smoke script: `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`
- Existing smoke tests: `tools/vz-linux-image/tests/test_host_e2e_smoke_script.py`
- Existing Swift socket tests: `tools/macos-vz-helper/Tests/UnixSocketServerTests.swift`

## File Structure

- Modify `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
  - Export one public expected helper protocol constant for the lifecycle command.
- Modify `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`
  - Cover the public protocol constant and default client behavior.
- Modify `tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift`
  - Add helper-side socket path safety before unlinking.
- Modify `tools/macos-vz-helper/Tests/UnixSocketServerTests.swift`
  - Cover symlink/non-socket refusal and stale socket removal.
- Create `tools/macos-vz-helper/scripts/vz-helperctl.py`
  - Operator CLI for `check`, `build`, `sign`, `start`, `status`, `stop`, `plist`, and `smoke`.
- Create `tools/macos-vz-helper/tests/test_vz_helperctl.py`
  - Portable unit tests for CLI defaults, dry-runs, path checks, pid checks, plist generation, and smoke delegation.
- Modify `tools/macos-vz-helper/README.md`
  - Document helper lifecycle command and safe launchd plist generation.
- Modify `Docs/Sandbox/macos-runtime-operator-notes.md`
  - Update operator workflow from ad hoc helper startup to `vz-helperctl`.
- Modify `tldw_Server_API/app/core/Sandbox/README.md`
  - Mention the managed helper lifecycle command and remaining non-goals.

## Task 1: Expose Helper Protocol Version Source Of Truth

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
- Modify: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`

- [ ] **Step 1: Write failing tests for the public protocol constant**

Add to `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`:

```python
def test_helper_client_exports_expected_protocol_version() -> None:
    from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
        EXPECTED_HELPER_PROTOCOL_VERSION,
    )

    assert EXPECTED_HELPER_PROTOCOL_VERSION == "1"


def test_helper_client_default_uses_expected_protocol_version(monkeypatch) -> None:
    from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
        EXPECTED_HELPER_PROTOCOL_VERSION,
    )

    requests = _install_fake_helper_socket(
        monkeypatch,
        {
            "ping": {
                "protocol_version": EXPECTED_HELPER_PROTOCOL_VERSION,
                "helper_version": "0.1.0",
                "status": "ok",
                "details": {"transport": "unix"},
            }
        },
    )

    MacOSVirtualizationHelperClient().ping()

    assert requests[0]["protocol_version"] == EXPECTED_HELPER_PROTOCOL_VERSION
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py::test_helper_client_exports_expected_protocol_version \
  tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py::test_helper_client_default_uses_expected_protocol_version \
  -v
```

Expected: FAIL because `EXPECTED_HELPER_PROTOCOL_VERSION` does not exist.

- [ ] **Step 3: Implement the public constant**

In `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`, replace the private default assignment:

```python
_DEFAULT_PROTOCOL_VERSION = "1"
```

with:

```python
EXPECTED_HELPER_PROTOCOL_VERSION = "1"
_DEFAULT_PROTOCOL_VERSION = EXPECTED_HELPER_PROTOCOL_VERSION
```

Do not change wire behavior in this task.

- [ ] **Step 4: Run tests and verify they pass**

Run the same targeted pytest command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py \
  tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py
git commit -m "refactor(sandbox): expose macos helper protocol version"
```

## Task 2: Harden Swift Helper Socket Startup Safety

**Files:**
- Modify: `tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift`
- Modify: `tools/macos-vz-helper/Tests/UnixSocketServerTests.swift`

- [ ] **Step 1: Write failing Swift tests for unsafe socket paths**

Add tests to `tools/macos-vz-helper/Tests/UnixSocketServerTests.swift`:

```swift
@Test func unixSocketServerRefusesExistingRegularFileSocketPath() throws {
    let dir = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("macos-vz-helper-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }

    let socket = dir.appendingPathComponent("helper.sock")
    try "do not remove".write(to: socket, atomically: true, encoding: .utf8)
    let server = UnixSocketServer(socketPath: socket.path, service: HelperService())

    #expect(throws: UnixSocketServerError.self) {
        try server.start()
    }
    #expect(FileManager.default.fileExists(atPath: socket.path))
}

@Test func unixSocketServerRefusesSymlinkSocketPath() throws {
    let dir = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("macos-vz-helper-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }

    let target = dir.appendingPathComponent("target")
    let socket = dir.appendingPathComponent("helper.sock")
    try "target".write(to: target, atomically: true, encoding: .utf8)
    try FileManager.default.createSymbolicLink(atPath: socket.path, withDestinationPath: target.path)

    let server = UnixSocketServer(socketPath: socket.path, service: HelperService())

    #expect(throws: UnixSocketServerError.self) {
        try server.start()
    }
    #expect(FileManager.default.fileExists(atPath: socket.path))
}
```

Also add a stale socket test if AF_UNIX binding is available:

```swift
@Test func unixSocketServerRemovesExistingSocketPath() throws {
    let socketPath = "/tmp/macos-vz-helper-\(UUID().uuidString.prefix(8)).sock"
    let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { return }
    defer {
        close(fd)
        unlink(socketPath)
    }

    try bindSocketForTest(fd: fd, path: socketPath)
    let server = UnixSocketServer(socketPath: socketPath, service: HelperService())
    try server.start()
    defer { server.stop() }

    #expect(FileManager.default.fileExists(atPath: socketPath))
}
```

Reuse or extract the existing `sockaddr_un` setup from `sendSocketRequest` into a local test helper.

- [ ] **Step 2: Run Swift tests and verify they fail**

Run:

```bash
swift test --package-path tools/macos-vz-helper --filter UnixSocketServerTests
```

Expected: FAIL because `UnixSocketServer.start()` currently unlinks any existing socket path.

- [ ] **Step 3: Implement helper-side socket safety**

In `tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift`, add explicit errors:

```swift
case unsafeSocketPath(String)
case existingSocketPathIsNotSocket(String)
```

Add a private method:

```swift
private func prepareSocketPath() throws {
    let parent = URL(fileURLWithPath: socketPath).deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)

    var statBuffer = stat()
    let result = lstat(socketPath, &statBuffer)
    if result != 0 {
        if errno == ENOENT {
            return
        }
        throw UnixSocketServerError.unsafeSocketPath(socketPath)
    }

    let mode = statBuffer.st_mode & S_IFMT
    if mode == S_IFLNK {
        throw UnixSocketServerError.unsafeSocketPath(socketPath)
    }
    if mode != S_IFSOCK {
        throw UnixSocketServerError.existingSocketPathIsNotSocket(socketPath)
    }
    unlink(socketPath)
}
```

Then replace the direct directory creation plus `unlink(socketPath)` in `start()` with:

```swift
try prepareSocketPath()
```

Keep the existing cleanup unlink on bind failure and stop.

- [ ] **Step 4: Run Swift tests and verify they pass**

Run:

```bash
swift test --package-path tools/macos-vz-helper --filter UnixSocketServerTests
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add \
  tools/macos-vz-helper/Sources/Server/UnixSocketServer.swift \
  tools/macos-vz-helper/Tests/UnixSocketServerTests.swift
git commit -m "fix(sandbox): harden macos helper socket startup"
```

## Task 3: Add `vz-helperctl` Core Checks And Plist Generation

**Files:**
- Create: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Create: `tools/macos-vz-helper/tests/test_vz_helperctl.py`

- [ ] **Step 1: Write failing tests for CLI defaults and path safety**

Create `tools/macos-vz-helper/tests/test_vz_helperctl.py`.

Use `importlib.util.spec_from_file_location` to import the script despite the hyphenated parent directory:

```python
from __future__ import annotations

import importlib.util
import os
import stat
import sys
from pathlib import Path

HELPERCTL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "vz-helperctl.py"


def _load_helperctl():
    spec = importlib.util.spec_from_file_location("vz_helperctl", HELPERCTL_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["vz_helperctl"] = module
    spec.loader.exec_module(module)
    return module


def test_default_paths_are_user_owned(monkeypatch, tmp_path: Path) -> None:
    helperctl = _load_helperctl()
    monkeypatch.setenv("HOME", str(tmp_path))

    paths = helperctl.default_paths()

    assert paths.socket_path == tmp_path / "Library/Application Support/tldw/sandbox/macos-vz-helper/helper.sock"
    assert paths.pid_file == tmp_path / "Library/Application Support/tldw/sandbox/macos-vz-helper/helper.pid"
    assert paths.log_dir == tmp_path / "Library/Logs/tldw/macos-vz-helper"


def test_validate_socket_path_refuses_symlink(tmp_path: Path) -> None:
    helperctl = _load_helperctl()
    target = tmp_path / "target"
    socket_path = tmp_path / "helper.sock"
    target.write_text("target", encoding="utf-8")
    socket_path.symlink_to(target)

    result = helperctl.validate_socket_path(socket_path)

    assert result.ok is False
    assert result.reason == "helper_socket_unsafe"


def test_validate_socket_path_refuses_regular_file(tmp_path: Path) -> None:
    helperctl = _load_helperctl()
    socket_path = tmp_path / "helper.sock"
    socket_path.write_text("do not remove", encoding="utf-8")

    result = helperctl.validate_socket_path(socket_path)

    assert result.ok is False
    assert result.reason == "helper_socket_unsafe"
    assert socket_path.read_text(encoding="utf-8") == "do not remove"
```

- [ ] **Step 2: Write failing tests for plist generation**

Add:

```python
def test_render_launchd_plist_contains_helper_paths(tmp_path: Path) -> None:
    helperctl = _load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    log_dir = tmp_path / "logs"

    payload = helperctl.render_launchd_plist(
        helper_path=helper,
        socket_path=socket_path,
        log_dir=log_dir,
        label="org.tldw.macos-vz-helper",
    )

    assert "org.tldw.macos-vz-helper" in payload
    assert str(helper) in payload
    assert "TLDW_SANDBOX_MACOS_HELPER_SOCKET" in payload
    assert str(socket_path) in payload
    assert "<key>KeepAlive</key>" in payload
    assert "<false/>" in payload
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/macos-vz-helper/tests/test_vz_helperctl.py -v
```

Expected: FAIL because `vz-helperctl.py` does not exist.

- [ ] **Step 4: Implement minimal `vz-helperctl.py` core**

Create `tools/macos-vz-helper/scripts/vz-helperctl.py` with:

- dataclasses `CheckResult` and `HelperPaths`
- `default_paths()`
- `validate_socket_path(path: Path) -> CheckResult`
- `ensure_private_dir(path: Path, *, dry_run: bool = False) -> CheckResult`
- `render_launchd_plist(...) -> str`
- an `argparse` CLI with `check` and `plist` subcommands wired first

Implementation details:

```python
EXPECTED_HELPER_PROTOCOL_VERSION = _load_expected_protocol_version()
REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER_PACKAGE_DIR = REPO_ROOT / "tools" / "macos-vz-helper"
DEFAULT_HELPER = HELPER_PACKAGE_DIR / ".build" / "debug" / "macos-vz-helper"
```

`_load_expected_protocol_version()` should import:

```python
from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
    EXPECTED_HELPER_PROTOCOL_VERSION,
)
```

and return `"1"` only as a defensive fallback for CLI bootstrap failures.

Path validation should use `Path.is_symlink()`, `Path.exists()`, and `stat.S_ISSOCK(path.lstat().st_mode)`.

The plist renderer should use `plistlib.dumps` to avoid malformed XML:

```python
plistlib.dumps(
    {
        "Label": label,
        "ProgramArguments": [str(helper_path)],
        "EnvironmentVariables": {
            "TLDW_SANDBOX_MACOS_HELPER_SOCKET": str(socket_path),
            "TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR": str(log_dir / "serial"),
        },
        "StandardOutPath": str(log_dir / "helper.stdout.log"),
        "StandardErrorPath": str(log_dir / "helper.stderr.log"),
        "KeepAlive": False,
        "RunAtLoad": False,
    },
    sort_keys=True,
).decode("utf-8")
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/macos-vz-helper/tests/test_vz_helperctl.py -v
```

Expected: PASS for the new core/plist tests.

- [ ] **Step 6: Commit Task 3**

```bash
git add \
  tools/macos-vz-helper/scripts/vz-helperctl.py \
  tools/macos-vz-helper/tests/test_vz_helperctl.py
git commit -m "feat(sandbox): add macos helper lifecycle checks"
```

## Task 4: Add Build, Sign, Status, Start, And Stop Behavior

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Modify: `tools/macos-vz-helper/tests/test_vz_helperctl.py`

- [x] **Step 1: Write failing tests for dry-run build and sign**

Add tests:

```python
def test_build_dry_run_prints_swift_command(capsys) -> None:
    helperctl = _load_helperctl()

    code = helperctl.main(["build", "--dry-run"])

    captured = capsys.readouterr()
    assert code == 0
    assert "swift build --package-path" in captured.out


def test_sign_requires_entitlements(tmp_path: Path, capsys) -> None:
    helperctl = _load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o755)

    code = helperctl.main(["sign", "--helper", str(helper), "--dry-run"])

    captured = capsys.readouterr()
    assert code != 0
    assert "helper_entitlements_missing" in captured.err
```

- [x] **Step 2: Write failing tests for pid mismatch and failed-start cleanup**

Use fake process hooks rather than spawning real helpers:

```python
def test_pid_file_mismatch_is_rejected(tmp_path: Path) -> None:
    helperctl = _load_helperctl()
    pid_file = tmp_path / "helper.pid"
    pid_file.write_text("12345", encoding="utf-8")

    result = helperctl.validate_pid_file(
        pid_file=pid_file,
        expected_helper=tmp_path / "macos-vz-helper",
        process_lookup=lambda pid: helperctl.ProcessInfo(pid=pid, command="/bin/other"),
    )

    assert result.ok is False
    assert result.reason == "helper_pid_process_mismatch"


def test_start_cleans_up_just_started_process_on_ping_failure(tmp_path: Path) -> None:
    helperctl = _load_helperctl()
    killed: list[int] = []
    pid_file = tmp_path / "helper.pid"

    code = helperctl.start_helper(
        helper_path=tmp_path / "macos-vz-helper",
        socket_path=tmp_path / "runtime" / "helper.sock",
        pid_file=pid_file,
        log_dir=tmp_path / "logs",
        dry_run=False,
        process_starter=lambda *_args, **_kwargs: helperctl.StartedProcess(pid=1234),
        ping_checker=lambda *_args, **_kwargs: helperctl.CheckResult(False, "helper_ping_failed"),
        process_killer=lambda pid: killed.append(pid),
    )

    assert code.reason == "helper_ping_failed"
    assert killed == [1234]
    assert not pid_file.exists()
```

Adjust exact helper function names if the implementation uses a different small seam, but keep injected process hooks so tests do not launch real helpers.

- [x] **Step 3: Implement lifecycle process helpers**

Add:

- `run_command(argv, *, dry_run=False, env=None) -> int`
- `build_helper(...)`
- `sign_helper(...)`
- `read_codesign_entitlements(...)`
- `compare_entitlements(...)`
- `validate_pid_file(...)`
- `start_helper(...)`
- `status_helper(...)`
- `stop_helper(...)`

Implementation requirements:

- `build --dry-run` prints the SwiftPM command and exits 0.
- `sign` requires `--entitlements`.
- `check` distinguishes unsigned, unreadable, matching, and mismatching entitlements on macOS.
- `start` refuses a live expected helper with `helper_already_running`.
- `start` cleans up only the process it started if ping/protocol validation fails.
- `stop` validates pid ownership before sending terminate.
- `status` never starts or stops anything.

- [x] **Step 4: Run lifecycle CLI tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tools/macos-vz-helper/tests/test_vz_helperctl.py -v
```

Expected: PASS.

- [x] **Step 5: Commit Task 4**

```bash
git add \
  tools/macos-vz-helper/scripts/vz-helperctl.py \
  tools/macos-vz-helper/tests/test_vz_helperctl.py
git commit -m "feat(sandbox): manage macos helper process lifecycle"
```

## Task 5: Add Smoke Delegation And Documentation

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Modify: `tools/macos-vz-helper/tests/test_vz_helperctl.py`
- Modify: `tools/macos-vz-helper/README.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`

- [x] **Step 1: Write failing smoke delegation test**

Add:

```python
def test_smoke_dry_run_delegates_to_host_smoke_script(tmp_path: Path, capsys) -> None:
    helperctl = _load_helperctl()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    helper = tmp_path / "macos-vz-helper"
    entitlements = tmp_path / "helper.entitlements"
    entitlements.write_text("<plist/>", encoding="utf-8")

    code = helperctl.main(
        [
            "smoke",
            "--dry-run",
            "--bundle",
            str(bundle),
            "--helper",
            str(helper),
            "--entitlements",
            str(entitlements),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "run-host-e2e-smoke.sh" in captured.out
    assert f"--bundle {bundle}" in captured.out
    assert f"--helper {helper}" in captured.out
```

- [x] **Step 2: Implement `smoke` delegation**

Implement `smoke` as command construction around:

```text
tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
```

Pass:

- `--bundle`
- `--socket`
- `--serial-log-dir`
- `--helper`
- `--entitlements`
- `--python`
- `--dry-run`

Do not duplicate helper daemon smoke or real host E2E pytest logic.

- [x] **Step 3: Update helper README**

Add a "Managed helper lifecycle" section to `tools/macos-vz-helper/README.md`:

````markdown
## Managed Helper Lifecycle

Use `tools/macos-vz-helper/scripts/vz-helperctl.py` for local operator workflows:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py check
tools/macos-vz-helper/scripts/vz-helperctl.py build
tools/macos-vz-helper/scripts/vz-helperctl.py start
tools/macos-vz-helper/scripts/vz-helperctl.py status
tools/macos-vz-helper/scripts/vz-helperctl.py stop
tools/macos-vz-helper/scripts/vz-helperctl.py plist --dry-run
```

The command uses stable user-owned defaults under `~/Library/Application Support/tldw/sandbox/macos-vz-helper/` and `~/Library/Logs/tldw/macos-vz-helper/`.
It does not install launchd services or auto-upgrade helpers.
````

- [x] **Step 4: Update operator docs**

In `Docs/Sandbox/macos-runtime-operator-notes.md`, update "Real Host E2E Smoke" to show:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py check
tools/macos-vz-helper/scripts/vz-helperctl.py smoke \
  --bundle /path/to/canonical/bundle \
  --entitlements /path/to/helper.entitlements
```

Keep the existing direct `run-host-e2e-smoke.sh` command as the lower-level fallback.

- [x] **Step 5: Update sandbox README**

In `tldw_Server_API/app/core/Sandbox/README.md`, add one bullet under macOS scaffolding/current limitations explaining:

- `vz-helperctl.py` is the preferred operator helper lifecycle command.
- It generates launchd plist scaffolding but does not install/load services.

- [x] **Step 6: Run smoke/documentation tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tools/macos-vz-helper/tests/test_vz_helperctl.py \
  tools/vz-linux-image/tests/test_host_e2e_smoke_script.py \
  -v
```

Expected: PASS.

- [x] **Step 7: Commit Task 5**

```bash
git add \
  tools/macos-vz-helper/scripts/vz-helperctl.py \
  tools/macos-vz-helper/tests/test_vz_helperctl.py \
  tools/macos-vz-helper/README.md \
  Docs/Sandbox/macos-runtime-operator-notes.md \
  tldw_Server_API/app/core/Sandbox/README.md
git commit -m "docs(sandbox): document managed macos helper lifecycle"
```

## Task 6: Final Verification And PR Readiness

**Files:**
- No new files expected.

- [ ] **Step 1: Run focused Python tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tools/macos-vz-helper/tests/test_vz_helperctl.py \
  tools/vz-linux-image/tests/test_host_e2e_smoke_script.py \
  tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run Swift helper tests**

Run:

```bash
swift test --package-path tools/macos-vz-helper --filter UnixSocketServerTests
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched Python production files**

Run:

```bash
source .venv/bin/activate && python -m bandit \
  tools/macos-vz-helper/scripts/vz-helperctl.py \
  tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py \
  -f json -o /tmp/bandit_vz_helper_lifecycle.json
```

Expected: command exits 0 and JSON `results` length is 0.

- [ ] **Step 4: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Run dry-run operator commands manually**

Run:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py check --json
tools/macos-vz-helper/scripts/vz-helperctl.py build --dry-run
tools/macos-vz-helper/scripts/vz-helperctl.py plist --dry-run
tools/macos-vz-helper/scripts/vz-helperctl.py smoke --dry-run --bundle /tmp/nonexistent-bundle
```

Expected:

- `check --json` emits JSON and returns non-zero only if required host prerequisites are missing.
- `build --dry-run` prints SwiftPM command.
- `plist --dry-run` prints plist XML.
- `smoke --dry-run` prints delegated smoke command. If bundle validation intentionally runs before dry-run, document that behavior and use a temporary minimal bundle for the dry-run.

- [ ] **Step 6: Commit any final fixes**

If final verification found issues, fix them and commit:

```bash
git add <changed-files>
git commit -m "fix(sandbox): polish macos helper lifecycle workflow"
```

- [ ] **Step 7: Prepare PR summary**

Summarize:

- helper-side socket startup hardening
- `vz-helperctl.py` lifecycle command
- launchd plist dry-run generation
- entitlement/protocol/status checks
- docs and verification

Do not claim real VM E2E passed unless it was run on a prepared Apple silicon host with a real bundle.
