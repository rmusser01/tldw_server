# VZ Helper Launchd Validation Drill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit operator-owned `vz-helperctl.py launchd-drill` command that validates launchd helper supervision without changing the default direct-helper smoke path.

**Architecture:** Keep the drill inside `tools/macos-vz-helper/scripts/vz-helperctl.py` so it reuses the existing launchd argv, plist, path-safety, status, and output conventions. Add portable tests with injected runners for launchctl/status/smoke behavior, then document the local operator flow and host-gated boundaries. Do not add workflow integration in this first implementation.

**Tech Stack:** Python 3 CLI, `argparse`, existing helperctl `CheckResult` output, pytest, macOS `launchctl` command construction, existing VZ Linux host-gated pytest smoke contract.

---

## Source Inputs

- Spec: `Docs/superpowers/specs/2026-05-15-vz-helper-launchd-validation-drill-design.md`
- Existing launchd plan: `Docs/superpowers/plans/2026-05-13-vz-helper-launchd-operator.md`
- Helper CLI: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Helper CLI tests: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`
- Operator docs: `tools/macos-vz-helper/README.md`
- macOS operator notes: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Host-gated policy: `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
- Tracker: `https://github.com/rmusser01/tldw_server/issues/1442`

## File Map

- Modify `tools/macos-vz-helper/scripts/vz-helperctl.py`
  - Add launchd drill defaults, launchd loaded-state check, external-helper smoke runner, drill orchestration, CLI parser, and JSON/human output.
- Modify `tools/macos-vz-helper/Tests/test_vz_helperctl.py`
  - Add portable unit tests for dry-run, sequencing, cleanup, loaded-label guard, missing pid-file launchd mode, JSON shape, and external-helper smoke command construction.
- Modify `tools/macos-vz-helper/README.md`
  - Document local launchd drill usage, isolated labels, and cleanup expectations.
- Modify `Docs/Sandbox/macos-runtime-operator-notes.md`
  - Add operator guidance for the launchd validation drill and clarify that it is distinct from `restart-drill`.
- Modify `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
  - Document expected skip/blocking semantics for manual launchd validation without enabling it in scheduled CI.
- Modify `backlog/tasks/task-364 - Plan-VZ-helper-launchd-validation-drill-implementation.md`
  - Track planning completion.

## Implementation Constraints

- Default drill label must be isolated, for example `org.tldw.macos-vz-helper.drill.<pid>`.
- Default drill plist path must live under a private runtime directory, not `~/Library/LaunchAgents`.
- If the selected launchd target is already loaded before bootstrap, fail with `launchd_service_already_loaded` and do not bootout it.
- Only bootout a service target that this drill successfully bootstrapped.
- Missing helperctl pid file is acceptable in launchd mode when launchd status plus helper ping/protocol are healthy.
- Do not start a second helper during VM smoke. Use the host-gated pytest smoke contract against the launchd-managed socket.
- Do not add `run-host-e2e-smoke.sh --use-launchd` or workflow integration in this first PR.
- Do not hide build/sign inside the drill. Require a prepared helper and report clear preflight failures.

---

### Task 1: Drill Defaults And Launchd Loaded Guard

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Test: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`

- [ ] **Step 1: Write failing tests for drill defaults**

Add tests near existing launchd tests:

```python
def test_launchd_drill_defaults_use_private_runtime_label_and_plist(monkeypatch, tmp_path):
    helperctl = load_helperctl()
    monkeypatch.setenv("HOME", str(tmp_path))

    paths = helperctl.default_paths()
    label = helperctl.default_launchd_drill_label(pid=12345)
    plist_path = helperctl.default_launchd_drill_plist_path(paths, label)

    CASE.assertEqual(label, "org.tldw.macos-vz-helper.drill.12345")
    CASE.assertEqual(
        plist_path,
        paths.socket_path.parent / "launchd-drill" / "org.tldw.macos-vz-helper.drill.12345.plist",
    )
```

- [ ] **Step 2: Write failing tests for loaded-service guard**

```python
def test_launchd_service_loaded_returns_true_for_launchctl_print_zero():
    helperctl = load_helperctl()

    result = helperctl.launchd_service_loaded(
        "org.tldw.test",
        uid=501,
        command_runner=lambda argv, **kwargs: 0,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=True, reason="launchd_service_loaded"))


def test_launchd_service_loaded_returns_false_for_launchctl_print_nonzero():
    helperctl = load_helperctl()

    result = helperctl.launchd_service_loaded(
        "org.tldw.test",
        uid=501,
        command_runner=lambda argv, **kwargs: 3,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=True, reason="launchd_service_absent"))
```

- [ ] **Step 3: Run focused tests and verify failure**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -k 'launchd_drill_defaults or launchd_service_loaded' -q
```

Expected: FAIL because the new helper functions do not exist.

- [ ] **Step 4: Implement defaults and loaded guard**

Add near the existing launchd helpers:

```python
def default_launchd_drill_label(*, pid: int | None = None) -> str:
    suffix = os.getpid() if pid is None else pid
    return f"{DEFAULT_LAUNCHD_LABEL}.drill.{suffix}"


def default_launchd_drill_plist_path(paths: HelperPaths, label: str) -> Path:
    return paths.socket_path.parent / "launchd-drill" / f"{label}.plist"


def launchd_service_loaded(
    label: str,
    *,
    uid: int | None = None,
    dry_run: bool = False,
    command_runner: Callable[..., int] | None = None,
) -> CheckResult:
    runner = command_runner or run_command
    argv = launchd_argv("status", label=label, uid=uid)
    if dry_run:
        code = runner(argv, dry_run=True)
        return CheckResult(ok=True, reason="dry_run" if code == 0 else "launchd_service_absent")
    if _is_default_run_command(runner) and shutil.which("launchctl") is None:
        return CheckResult(ok=False, reason="launchd_launchctl_unavailable")
    code = runner(argv, dry_run=False)
    if code == 0:
        return CheckResult(ok=True, reason="launchd_service_loaded", message=launchd_service_target(label, uid=uid))
    if code == 127:
        return CheckResult(ok=False, reason="launchd_launchctl_unavailable")
    return CheckResult(ok=True, reason="launchd_service_absent", message=launchd_service_target(label, uid=uid))
```

- [ ] **Step 5: Run focused tests and verify pass**

Run the same focused pytest command.

- [ ] **Step 6: Commit**

```bash
git add tools/macos-vz-helper/scripts/vz-helperctl.py tools/macos-vz-helper/Tests/test_vz_helperctl.py
git commit -m "test(sandbox): add launchd drill defaults"
```

---

### Task 2: Launchd Drill Orchestration And Cleanup

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Test: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`

- [ ] **Step 1: Write failing tests for drill sequencing**

Add tests that inject launchd and ping runners:

```python
def test_launchd_drill_runs_bootstrap_status_kickstart_ping_bootout(tmp_path):
    helperctl = load_helperctl()
    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    private_root.chmod(0o700)
    helper = private_root / "macos-vz-helper"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    socket_path = private_root / "runtime" / "helper.sock"
    log_dir = private_root / "logs"
    plist_path = private_root / "launchd-drill" / "org.tldw.test.plist"
    calls: list[list[str]] = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return 3 if argv[:2] == ["launchctl", "print"] and len(calls) == 1 else 0

    results = helperctl.run_launchd_drill(
        helper_path=helper,
        socket_path=socket_path,
        log_dir=log_dir,
        plist_path=plist_path,
        label="org.tldw.test",
        uid=501,
        write_plist=True,
        create_dirs=True,
        launchd_runner=runner,
        ping_checker=lambda path: helperctl.PingState(helperctl.CheckResult(ok=True), "1", "test"),
        smoke_runner=None,
    )

    by_name = dict(results)
    CASE.assertTrue(all(result.ok for _, result in results))
    CASE.assertEqual(by_name["launchd_bootstrap"].reason, "ok")
    CASE.assertEqual(by_name["launchd_bootout"].reason, "ok")
    CASE.assertIn(["launchctl", "bootstrap", "gui/501", str(plist_path)], calls)
    CASE.assertIn(["launchctl", "kickstart", "-k", "gui/501/org.tldw.test"], calls)
    CASE.assertIn(["launchctl", "bootout", "gui/501/org.tldw.test"], calls)
```

- [ ] **Step 2: Write failing tests for cleanup and loaded-label protection**

Cover:

```python
def test_launchd_drill_refuses_already_loaded_service_without_bootout(...):
    # First launchctl print returns 0.
    # Assert result contains launchd_preflight with launchd_service_already_loaded.
    # Assert no bootstrap or bootout call was made.

def test_launchd_drill_bootouts_after_kickstart_success_and_ping_failure(...):
    # Preflight print returns nonzero, bootstrap and kickstart return 0,
    # ping returns helper_ping_failed.
    # Assert bootout is still attempted and primary failure is preserved.

def test_launchd_drill_preserves_primary_failure_when_bootout_fails(...):
    # Ping fails and bootout returns nonzero.
    # Assert primary result reason is helper_ping_failed and cleanup result records launchd_bootout_failed.
```

- [ ] **Step 3: Run focused tests and verify failure**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -k 'launchd_drill' -q
```

Expected: FAIL because `run_launchd_drill` does not exist.

- [ ] **Step 4: Implement `run_launchd_drill`**

Add a function returning `list[tuple[str, CheckResult]]`. Keep it small and procedural:

```python
def run_launchd_drill(
    *,
    helper_path: Path,
    socket_path: Path,
    log_dir: Path,
    plist_path: Path,
    label: str,
    uid: int | None = None,
    write_plist: bool = False,
    create_dirs: bool = False,
    dry_run: bool = False,
    ping_checker: Callable[[Path], CheckResult | PingState] = ping_helper_state,
    launchd_runner: Callable[..., int] | None = None,
    smoke_runner: Callable[[], CheckResult] | None = None,
) -> list[tuple[str, CheckResult]]:
    results: list[tuple[str, CheckResult]] = []
    bootstrapped = False

    preflight = launchd_service_loaded(label, uid=uid, dry_run=dry_run, command_runner=launchd_runner)
    if preflight.reason == "launchd_service_loaded":
        results.append(("launchd_preflight", CheckResult(False, "launchd_service_already_loaded", preflight.message)))
        return results
    results.append(("launchd_preflight", preflight))
    if not preflight.ok:
        return results

    bootstrap = run_launchd_action(
        "bootstrap",
        label=label,
        plist_path=plist_path,
        helper_path=helper_path,
        socket_path=socket_path,
        log_dir=log_dir,
        uid=uid,
        dry_run=dry_run,
        write_plist=write_plist,
        create_dirs=create_dirs,
        command_runner=launchd_runner,
    )
    results.append(("launchd_bootstrap", bootstrap))
    if not bootstrap.ok:
        return results
    bootstrapped = not dry_run

    try:
        # status, kickstart, wait_for_ping, optional smoke
        ...
    finally:
        if bootstrapped:
            bootout = run_launchd_action(...)
            results.append(("launchd_bootout", bootout))
```

Use `wait_for_ping(socket_path, ping_checker=ping_checker)` after kickstart. Do not require a helperctl pid file.

- [ ] **Step 5: Run focused tests and verify pass**

Run the focused launchd-drill pytest selection.

- [ ] **Step 6: Commit**

```bash
git add tools/macos-vz-helper/scripts/vz-helperctl.py tools/macos-vz-helper/Tests/test_vz_helperctl.py
git commit -m "feat(sandbox): add launchd drill orchestration"
```

---

### Task 3: External-Helper VZ Smoke Runner

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Test: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`

- [ ] **Step 1: Write failing tests for external smoke command construction**

```python
def test_run_vz_linux_host_smoke_uses_existing_helper_socket(tmp_path):
    helperctl = load_helperctl()
    calls: list[tuple[list[str], dict[str, str] | None]] = []
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    socket_path = tmp_path / "helper.sock"

    result = helperctl.run_vz_linux_host_smoke(
        bundle_path=bundle,
        socket_path=socket_path,
        python_path=Path("/usr/bin/python3"),
        command_runner=lambda argv, **kwargs: calls.append((argv, kwargs.get("env"))) or 0,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=True))
    argv, env = calls[0]
    CASE.assertEqual(argv[:3], ["/usr/bin/python3", "-m", "pytest"])
    CASE.assertEqual(env["TLDW_SANDBOX_MACOS_HELPER_SOCKET"], str(socket_path))
    CASE.assertEqual(env["TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE"], str(bundle))
    CASE.assertEqual(env["SANDBOX_BACKGROUND_EXECUTION"], "0")
```

- [ ] **Step 2: Write failing test for smoke failure reason**

```python
def test_run_vz_linux_host_smoke_reports_failure(tmp_path):
    helperctl = load_helperctl()

    result = helperctl.run_vz_linux_host_smoke(
        bundle_path=tmp_path / "bundle",
        socket_path=tmp_path / "helper.sock",
        python_path=Path("/usr/bin/python3"),
        command_runner=lambda argv, **kwargs: 2,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="vz_linux_smoke_failed", message="2"))
```

- [ ] **Step 3: Run focused tests and verify failure**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -k 'vz_linux_host_smoke' -q
```

Expected: FAIL because `run_vz_linux_host_smoke` does not exist.

- [ ] **Step 4: Implement smoke runner**

Add:

```python
def run_vz_linux_host_smoke(
    *,
    bundle_path: Path,
    socket_path: Path,
    python_path: Path | None = None,
    dry_run: bool = False,
    command_runner: Callable[..., int] | None = None,
) -> CheckResult:
    runner = command_runner or run_command
    python_bin = python_path or Path(sys.executable)
    env = os.environ.copy()
    env.update(
        {
            "TEST_MODE": "0",
            "TLDW_SANDBOX_VZ_LINUX_E2E": "1",
            "TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE": str(bundle_path),
            "TLDW_SANDBOX_MACOS_HELPER_SOCKET": str(socket_path),
            "SANDBOX_ENABLE_EXECUTION": "1",
            "SANDBOX_BACKGROUND_EXECUTION": "0",
        }
    )
    argv = [
        str(python_bin),
        "-m",
        "pytest",
        str(REPO_ROOT / "tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py"),
        "-m",
        "vz_linux_host_smoke",
        "-q",
        "-rs",
    ]
    code = runner(argv, dry_run=dry_run, env=env)
    if code == 0:
        return CheckResult(ok=True, reason="dry_run" if dry_run else "ok")
    return CheckResult(ok=False, reason="vz_linux_smoke_failed", message=str(code))
```

- [ ] **Step 5: Wire optional smoke into `run_launchd_drill`**

If `bundle_path` is supplied, call `run_vz_linux_host_smoke(...)` after helper ping succeeds. If no bundle is supplied or `--skip-smoke` is set, append a `vz_linux_smoke` result with `ok=True, reason="skipped"` or omit the step consistently; tests should assert the chosen behavior.

- [ ] **Step 6: Run focused tests and verify pass**

Run the `vz_linux_host_smoke or launchd_drill` focused selection.

- [ ] **Step 7: Commit**

```bash
git add tools/macos-vz-helper/scripts/vz-helperctl.py tools/macos-vz-helper/Tests/test_vz_helperctl.py
git commit -m "feat(sandbox): run VZ smoke through launchd helper"
```

---

### Task 4: CLI Parser And Output

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Test: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`

- [ ] **Step 1: Write failing CLI tests**

Cover:

```python
def test_launchd_drill_cli_dry_run_uses_isolated_label(monkeypatch, tmp_path, capsys):
    helperctl = load_helperctl()
    monkeypatch.setenv("HOME", str(tmp_path))

    code = helperctl.main(["launchd-drill", "--dry-run", "--skip-smoke"])

    output = capsys.readouterr().out
    CASE.assertEqual(code, 0)
    CASE.assertIn("org.tldw.macos-vz-helper.drill.", output)
    CASE.assertIn("launchd_preflight", output)


def test_launchd_drill_cli_json_outputs_steps(monkeypatch, tmp_path, capsys):
    # Monkeypatch run_launchd_drill to return deterministic CheckResult steps.
    # Assert JSON includes launchd_preflight, launchd_bootstrap, launchd_bootout.
```

Also test `--bundle` with `--skip-smoke` conflict if you decide to reject that combination. If accepted, document and test that `--skip-smoke` wins.

- [ ] **Step 2: Run CLI tests and verify failure**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -k 'launchd_drill_cli' -q
```

Expected: FAIL because the parser lacks `launchd-drill`.

- [ ] **Step 3: Implement `_launchd_drill_command` and parser**

Add parser:

```python
launchd_drill = subparsers.add_parser("launchd-drill", help="validate launchd-managed helper lifecycle")
launchd_drill.add_argument("--bundle")
launchd_drill.add_argument("--helper", "--helper-path", dest="helper_path")
launchd_drill.add_argument("--socket", "--socket-path", dest="socket_path")
launchd_drill.add_argument("--log-dir")
launchd_drill.add_argument("--plist-output")
launchd_drill.add_argument("--label")
launchd_drill.add_argument("--uid", type=int)
launchd_drill.add_argument("--python")
launchd_drill.add_argument("--write-plist", action="store_true")
launchd_drill.add_argument("--create-dirs", action="store_true")
launchd_drill.add_argument("--skip-smoke", action="store_true")
launchd_drill.add_argument("--dry-run", action="store_true")
launchd_drill.add_argument("--json", action="store_true")
launchd_drill.set_defaults(func=_launchd_drill_command)
```

In `_launchd_drill_command`, resolve defaults as:

```python
paths = default_paths()
label = args.label or default_launchd_drill_label()
plist_path = Path(args.plist_output) if args.plist_output else default_launchd_drill_plist_path(paths, label)
```

Use `_print_results(results, as_json=args.json)` and return `0` only when every result is ok.

- [ ] **Step 4: Run focused and full helperctl tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -k 'launchd_drill' -q
python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q
```

Expected: focused and full helperctl tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/macos-vz-helper/scripts/vz-helperctl.py tools/macos-vz-helper/Tests/test_vz_helperctl.py
git commit -m "feat(sandbox): expose launchd drill CLI"
```

---

### Task 5: Operator Docs And Acceptance Policy

**Files:**
- Modify: `tools/macos-vz-helper/README.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
- Modify: `backlog/tasks/task-364 - Plan-VZ-helper-launchd-validation-drill-implementation.md` only if completing this planning PR before implementation

- [ ] **Step 1: Update helper README**

Add a section after the existing `launchd` command examples:

````markdown
### Launchd Validation Drill

Use `launchd-drill` when you want to validate the launchd-managed helper path
without making launchd the default smoke lifecycle:

```bash
runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/tldw-vz-launchd-drill.XXXXXX")"
chmod 700 "$runtime_dir"
trap 'tools/macos-vz-helper/scripts/vz-helperctl.py launchd bootout --label "$label" --plist-output "$runtime_dir/${label}.plist" >/dev/null 2>&1 || true; rm -rf "$runtime_dir"' EXIT

label="org.tldw.macos-vz-helper.drill.$$"
tools/macos-vz-helper/scripts/vz-helperctl.py launchd-drill \
  --socket "$runtime_dir/helper.sock" \
  --log-dir "$runtime_dir/logs" \
  --plist-output "$runtime_dir/${label}.plist" \
  --label "$label" \
  --write-plist \
  --create-dirs \
  --skip-smoke
```

Pass `--bundle /path/to/canonical/bundle` to run the real VZ Linux host smoke
through the launchd-managed helper. The drill uses an isolated label by default
and refuses to take over a service that was loaded before the drill started.
````

Adjust the trap snippet if the final CLI can do a safer cleanup command.

- [ ] **Step 2: Update macOS operator notes**

Clarify:

- `restart-drill` validates direct helperctl-managed helper lifecycle.
- `launchd-drill` validates LaunchAgent bootstrap/kickstart/bootout and optional real `vz_linux` smoke.
- Host reboot remains manual and out of scheduled CI.
- The drill is opt-in and should use isolated labels/private plist paths.

- [ ] **Step 3: Update host-gated policy**

Add:

- launchd drill is an expected skip unless manually requested on a prepared runner.
- launchd drill failure is blocking only when the runner is explicitly configured for LaunchAgent validation and helper/template readiness passes.
- scheduled workflow should not enable it by default in this PR.

- [ ] **Step 4: Run docs checks**

Run:

```bash
git diff --check
rg -n "launchd-drill|Launchd Validation Drill|external-helper" tools/macos-vz-helper/README.md Docs/Sandbox/macos-runtime-operator-notes.md Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
```

Expected: no whitespace errors; all docs include the new drill references.

- [ ] **Step 5: Commit docs**

```bash
git add tools/macos-vz-helper/README.md Docs/Sandbox/macos-runtime-operator-notes.md Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
git commit -m "docs(sandbox): document launchd validation drill"
```

---

### Task 6: Final Verification And PR Prep

**Files:**
- Modify: `backlog/tasks/task-364 - Plan-VZ-helper-launchd-validation-drill-implementation.md` if this planning branch is being finalized.
- Otherwise, for the implementation PR, update the implementation Backlog task created for that PR.

- [ ] **Step 1: Run full helperctl tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q
```

Expected: all helperctl tests pass. At the time this plan was written, the current baseline around helperctl was `101 passed, 1 skipped`; update the expected count if the baseline has changed.

- [ ] **Step 2: Run docs whitespace check**

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 3: Run Bandit on touched Python script**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit -r tools/macos-vz-helper/scripts/vz-helperctl.py -f json -o /tmp/bandit_vz_launchd_drill.json
```

Expected: no new findings in `vz-helperctl.py`. If Bandit reports existing subprocess warnings, verify they match the existing direct-argv/no-shell pattern and document them in the task notes.

- [ ] **Step 4: Optional real-host manual smoke**

Only on a prepared Apple silicon host with a signed helper and canonical bundle:

```bash
runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/tldw-vz-launchd-drill.XXXXXX")"
chmod 700 "$runtime_dir"
label="org.tldw.macos-vz-helper.drill.$$"

tools/macos-vz-helper/scripts/vz-helperctl.py launchd-drill \
  --bundle /path/to/canonical/bundle \
  --socket "$runtime_dir/helper.sock" \
  --log-dir "$runtime_dir/logs" \
  --plist-output "$runtime_dir/${label}.plist" \
  --label "$label" \
  --write-plist \
  --create-dirs
```

Expected: launchd bootstrap/status/kickstart/ping/smoke/bootout all report ok. If this is not run, document it as a host-gated skip.

- [ ] **Step 5: Update Backlog task final summary**

Record:

- tests run and result
- Bandit result
- whether real launchd host smoke was run or skipped
- any known limitations

- [ ] **Step 6: Create PR against `dev`**

```bash
git status --short --branch
git push -u origin codex/vz-launchd-validation-drill
gh pr create --base dev --head codex/vz-launchd-validation-drill --title "Add VZ helper launchd validation drill" --draft
```

The PR body should explain that the default direct-helper smoke path remains unchanged and launchd validation is opt-in/operator-owned.
