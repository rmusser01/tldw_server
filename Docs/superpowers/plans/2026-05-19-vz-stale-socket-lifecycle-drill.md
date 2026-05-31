# VZ Stale Socket Lifecycle Drill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded stale-socket lifecycle drill/check for the macOS VZ helper that proves stale Unix socket recovery is safe, diagnosable, and fail-closed.

**Architecture:** Reuse the existing helper lifecycle primitives in `vz-helperctl.py` and `UnixSocketServer.swift`; do not add a second socket cleanup implementation. The operator-facing surface should be a manual `vz-helperctl.py stale-socket-drill` command that validates private directories, creates a controlled stale Unix socket only under that private runtime directory, starts the helper through the normal `start_helper()` path, verifies recovery, and reports evidence-friendly results. Swift helper coverage should remain focused on direct server socket-path safety because direct helper launch bypasses `vz-helperctl`.

**Tech Stack:** Python 3.11 pytest for `vz-helperctl.py`; SwiftPM `swift test` for `tools/macos-vz-helper`; Markdown operator docs and prepared-host evidence tracker.

---

## Scope Notes

- This is the first implementation slice from `Docs/superpowers/specs/2026-05-18-vz-linux-lifecycle-drill-gaps-design.md`.
- Keep this host-independent by default. It may bind local Unix sockets, but it must not boot a VM or require Virtualization.framework execution.
- Do not expand PR, push, scheduled, or destructive workflow triggers.
- Do not add a broad cleanup command. The drill may only create/remove the socket it created or an identity-verified stale socket accepted by existing lifecycle code.

## File Structure

- Modify `tools/macos-vz-helper/scripts/vz-helperctl.py`.
  Add a small `stale_socket_drill()` function and CLI command that composes existing validation/start/status helpers.
- Modify `tools/macos-vz-helper/Tests/test_vz_helperctl.py`.
  Add host-independent tests for the new command and for fail-closed path shapes.
- Modify `tools/macos-vz-helper/Tests/UnixSocketServerTests.swift`.
  Add one targeted Swift test for stale socket identity/replacement protection if current coverage is not explicit enough after review.
- Modify `tools/macos-vz-helper/README.md`.
  Document manual stale socket drill usage and evidence to capture.
- Modify `Docs/Sandbox/vz-linux-prepared-host-evidence.md`.
  Link the stale socket drill/check as the next evidence item without changing workflow triggers.
- Modify `backlog/tasks/task-433 - Implement-VZ-stale-socket-lifecycle-drill.md`.
  Track implementation notes, verification, and final summary.

## Task 1: Add Python Failing Tests For The Manual Drill

**Files:**
- Test: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`

- [ ] **Step 1: Add a test for safe stale socket recovery through the normal start path**

Add a test near the existing `start_helper` socket tests:

```python
def test_stale_socket_drill_recovers_controlled_stale_socket(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    runtime_dir = tmp_path / "runtime"
    socket_path = runtime_dir / "helper.sock"
    pid_file = runtime_dir / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)

    created = []

    def fake_socket_creator(path):
        created.append(path)
        runtime_dir.mkdir(mode=0o700, exist_ok=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(path))

    result = helperctl.stale_socket_drill(
        helper,
        socket_path,
        pid_file,
        log_dir,
        socket_creator=fake_socket_creator,
        starter=lambda *args, **kwargs: helperctl.CheckResult(True),
        status_collector=lambda *args, **kwargs: [("ping", helperctl.CheckResult(True, reason="helper_ping_ok"))],
    )

    CASE.assertEqual(created, [socket_path])
    CASE.assertEqual(result[-1], ("stale_socket_drill", helperctl.CheckResult(True)))
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py::test_stale_socket_drill_recovers_controlled_stale_socket -q
```

Expected: fail with `AttributeError: module 'vz_helperctl' has no attribute 'stale_socket_drill'`.

- [ ] **Step 3: Add tests for fail-closed path shapes**

Add focused tests that assert the drill refuses:

```python
@pytest.mark.parametrize("shape", ["symlink", "regular_file", "directory", "unsafe_parent"])
def test_stale_socket_drill_refuses_unsafe_socket_shapes(tmp_path, shape):
    ...
```

Each test should verify:

- The command returns `helper_socket_unsafe` or `helper_directory_not_private`.
- The unsafe path is still present after the drill.
- `starter` is not called.

- [ ] **Step 4: Run those tests and verify they fail for the missing function**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q -k stale_socket_drill
```

Expected: fail because `stale_socket_drill()` and the CLI command do not exist yet.

## Task 2: Implement The Python Drill By Composing Existing Helpers

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Test: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`

- [ ] **Step 1: Add a tiny controlled stale socket creator**

Add a helper near `socket_accepts_connection()`:

```python
def create_stale_unix_socket(path: Path) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
```

Do not call `listen()`. This creates a Unix socket path that is safe for the normal start path to identify as stale.

- [ ] **Step 2: Add `stale_socket_drill()`**

Implement a function with injectable collaborators:

```python
def stale_socket_drill(
    helper_path: Path,
    socket_path: Path,
    pid_file: Path,
    log_dir: Path,
    *,
    dry_run: bool = False,
    socket_creator: Callable[[Path], None] = create_stale_unix_socket,
    starter: Callable[..., CheckResult] = start_helper,
    status_collector: Callable[..., list[tuple[str, CheckResult]]] = collect_status_results,
) -> list[tuple[str, CheckResult]]:
    ...
```

The implementation should:

- Validate helper binary with `validate_helper_binary()`.
- Validate socket path with `validate_socket_path()`.
- Ensure `socket_path.parent`, `pid_file.parent`, `log_dir`, and `log_dir / "serial"` are private using `ensure_private_dir()`.
- Refuse existing active or unsafe sockets using existing `validate_socket_path()` plus `socket_accepts_connection()`.
- In dry-run mode, return only validation/check results and do not create a socket.
- Create the stale socket with `socket_creator(socket_path)` only after validation succeeds.
- Call `start_helper()` so actual cleanup remains identity-based in existing code.
- Append post-start `collect_status_results()` entries.
- Append final `("stale_socket_drill", CheckResult(...))`.

- [ ] **Step 3: Add the CLI command**

Add `_stale_socket_drill_command(args)` and register `stale-socket-drill` in `build_parser()` with:

- `--helper` / `--helper-path`
- `--socket` / `--socket-path`
- `--pid-file`
- `--log-dir`
- `--dry-run`
- `--json`

Use `_print_results()` and return nonzero if any result is not ok.

- [ ] **Step 4: Run focused Python tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q -k stale_socket_drill
```

Expected: pass.

- [ ] **Step 5: Run full helperctl Python tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q
```

Expected: `134 passed, 1 skipped` or equivalent current count plus new tests.

## Task 3: Add Or Confirm Swift Direct-Launch Socket Safety Coverage

**Files:**
- Modify: `tools/macos-vz-helper/Tests/UnixSocketServerTests.swift`

- [ ] **Step 1: Review current coverage before adding tests**

Confirm current tests cover:

- existing regular file is refused and preserved
- symlink socket path is refused and preserved
- stale Unix socket is removed and replaced
- active Unix socket is refused
- replacement path is not removed on stop
- unsafe parent directories are refused

- [ ] **Step 2: Add only missing direct-launch coverage**

If current coverage is missing identity-race protection during stale socket unlink, add a test that binds a socket, swaps the path before start can unlink, and expects `UnixSocketServerError` with the replacement preserved. If current coverage already proves this sufficiently, skip this step and record the skip in `TASK-433`.

- [ ] **Step 3: Run Swift tests**

Run:

```bash
swift test
```

Expected: all macOS helper Swift tests pass. If sandboxed SwiftPM cannot write module caches, rerun with host permissions and record that reason.

## Task 4: Document Manual Operator Usage And Evidence

**Files:**
- Modify: `tools/macos-vz-helper/README.md`
- Modify: `Docs/Sandbox/vz-linux-prepared-host-evidence.md`
- Modify: `backlog/tasks/task-433 - Implement-VZ-stale-socket-lifecycle-drill.md`

- [ ] **Step 1: Add README usage**

Add a short section near the helper lifecycle/operator commands:

```bash
runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/tldw-vz-stale-socket.XXXXXX")"
chmod 700 "${runtime_dir}"
trap 'rm -rf "${runtime_dir}"' EXIT

python tools/macos-vz-helper/scripts/vz-helperctl.py stale-socket-drill \
  --helper tools/macos-vz-helper/.build/debug/macos-vz-helper \
  --socket "${runtime_dir}/helper.sock" \
  --pid-file "${runtime_dir}/helper.pid" \
  --log-dir "${runtime_dir}/logs"
```

Include a note that the command is manual/operator-only and should not be wired into normal CI.

- [ ] **Step 2: Update evidence tracker**

In `Docs/Sandbox/vz-linux-prepared-host-evidence.md`, update the stale socket gap row to reference the new `stale-socket-drill` command and evidence fields:

- runtime dir mode
- socket path result
- command output
- helper stdout/stderr paths
- skip reason if not run

- [ ] **Step 3: Update task notes**

Record:

- implementation choices
- any Swift coverage skip rationale
- verification commands and results
- known skips, especially no real VM boot in this slice

## Task 5: Final Verification And Commit

**Files:**
- All touched files

- [ ] **Step 1: Run focused verification**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q
swift test
git diff --check
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -q tools/macos-vz-helper/scripts/vz-helperctl.py tools/macos-vz-helper/Tests/test_vz_helperctl.py
```

Expected:

- Python helperctl tests pass.
- Swift tests pass.
- `git diff --check` has no output.
- Bandit has no new findings.

- [ ] **Step 2: Self-review the diff**

Check:

- No broad deletion primitive was introduced.
- All socket unlink behavior remains identity-based or inside Swift direct-launch `lstat` checks.
- The command is manual-only and not referenced by normal CI workflows.
- Docs do not imply a VM smoke was run.

- [ ] **Step 3: Commit**

Run:

```bash
git add tools/macos-vz-helper/scripts/vz-helperctl.py \
  tools/macos-vz-helper/Tests/test_vz_helperctl.py \
  tools/macos-vz-helper/Tests/UnixSocketServerTests.swift \
  tools/macos-vz-helper/README.md \
  Docs/Sandbox/vz-linux-prepared-host-evidence.md \
  'backlog/tasks/task-433 - Implement-VZ-stale-socket-lifecycle-drill.md'
git commit -m "Add VZ stale socket lifecycle drill"
```

Expected: one focused commit on `codex/vz-stale-socket-drill`.
