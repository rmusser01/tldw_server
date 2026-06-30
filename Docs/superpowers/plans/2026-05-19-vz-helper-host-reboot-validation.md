# VZ Helper Host Reboot Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an operator-owned `vz-helperctl.py host-reboot-drill pre/post` workflow that records bounded pre-reboot evidence and validates helper readiness, diagnostics guidance, and optional real smoke after a manual host reboot.

**Architecture:** Keep host reboot outside repo automation. Implement a thin helperctl evidence layer that reuses existing private path validation, helper ping/status, launchd drill, and restored-helper smoke seams. Treat diagnostics and repair as operator-reviewed steps, with dry-run repair remaining explicit and external to the drill unless a later authenticated API option is designed.

**Tech Stack:** Python 3 helper CLI (`tools/macos-vz-helper/scripts/vz-helperctl.py`), pytest helperctl tests, existing `run_vz_linux_host_smoke`, Markdown operator docs, Backlog TASK-438.

---

## File Map

- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
  - Add host-reboot evidence helpers.
  - Extend helper ping state to preserve bounded helper `details` from the
    existing helper protocol response.
  - Add `host-reboot-drill pre` and `host-reboot-drill post` CLI paths.
  - Reuse existing path hardening, ping, launchd, and restored-helper smoke helpers.
- Modify: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`
  - Add portable tests for evidence directory safety, manifest shape, pre/post behavior, smoke targeting, and JSON output.
- Modify: `tools/macos-vz-helper/README.md`
  - Document the operator command and durable evidence directory requirements.
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
  - Add the manual host reboot validation procedure.
- Modify: `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
  - Clarify host reboot validation remains manual and record expected skip/blocking criteria.
- Modify: `backlog/tasks/task-438 - Design-VZ-helper-host-reboot-validation-procedure.md`
  - Record implementation progress, verification, and final summary.

## Task 1: Evidence Directory And Manifest Helpers

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Test: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`

- [ ] **Step 1: Add failing tests for evidence directory safety**

Add tests near the launchd/helperctl tests:

```python
def test_host_reboot_evidence_dir_rejects_world_readable(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    evidence.chmod(0o755)

    result = helperctl.ensure_host_reboot_evidence_dir(evidence, create=False)

    CASE.assertEqual(result.reason, "host_reboot_evidence_dir_not_private")
    CASE.assertFalse(result.ok)
```

Also add a volatile path test by monkeypatching `helperctl.VOLATILE_EVIDENCE_ROOTS` to include a temp directory:

```python
def test_host_reboot_evidence_dir_rejects_volatile_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volatile = tmp_path / "tmp"
    evidence = volatile / "drill"
    monkeypatch.setattr(helperctl, "VOLATILE_EVIDENCE_ROOTS", (volatile,))

    result = helperctl.ensure_host_reboot_evidence_dir(evidence, create=True)

    CASE.assertEqual(result.reason, "host_reboot_evidence_dir_volatile")
    CASE.assertFalse(result.ok)
```

Add a ping details regression because the host-reboot manifest needs helper
generation details already returned by the Swift helper protocol:

```python
def test_ping_helper_state_preserves_helper_details(tmp_path: Path) -> None:
    class FakeReply:
        protocol_version = "1"
        helper_version = "test-helper"
        details = {
            "helper_instance_id": "before",
            "helper_started_at": "2026-05-19T00:00:00Z",
            "ignored_number": 1,
        }

    class FakeClient:
        def ping(self) -> FakeReply:
            return FakeReply()

    state = helperctl.ping_helper_state(
        tmp_path / "helper.sock",
        client_factory=lambda path: FakeClient(),
    )

    CASE.assertEqual(state.details["helper_instance_id"], "before")
    CASE.assertEqual(state.details["helper_started_at"], "2026-05-19T00:00:00Z")
    CASE.assertNotIn("ignored_number", state.details)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py \
  -k "host_reboot_evidence_dir" -q
```

Expected: fail because the helper functions/constants and ping detail field do
not exist.

- [ ] **Step 3: Implement minimal evidence directory helper**

Extend `PingState` first:

```python
@dataclass(frozen=True)
class PingState:
    result: CheckResult
    protocol_version: str = ""
    helper_version: str = ""
    details: dict[str, str] | None = None
```

Then normalize helper details in `ping_helper_state()` from either the
client-factory reply or raw helper JSON:

```python
def _string_details(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, str):
            output[key] = item
    return output
```

Use `details={}` or `details=_string_details(...)` when constructing
`PingState`; do not keep non-string values.

Add constants and helper functions in `vz-helperctl.py` near the other path helpers:

```python
VOLATILE_EVIDENCE_ROOTS = tuple(
    Path(path).resolve()
    for path in ("/tmp", "/private/tmp", os.getenv("TMPDIR") or "")
    if path
)


def _is_under_path(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    except RuntimeError:
        return False
    return True


def ensure_host_reboot_evidence_dir(
    evidence_dir: Path,
    *,
    create: bool = False,
    allow_volatile: bool = False,
) -> CheckResult:
    if not allow_volatile:
        for root in VOLATILE_EVIDENCE_ROOTS:
            if _is_under_path(evidence_dir, root):
                return CheckResult(False, "host_reboot_evidence_dir_volatile", str(evidence_dir))
    if not evidence_dir.exists():
        if not create:
            return CheckResult(False, "host_reboot_evidence_dir_missing", str(evidence_dir))
        evidence_dir.mkdir(mode=0o700, parents=True)
        evidence_dir.chmod(0o700)
    result = ensure_private_dir(evidence_dir, dry_run=False)
    if not result.ok:
        return CheckResult(False, "host_reboot_evidence_dir_not_private", result.message or str(evidence_dir))
    return CheckResult(True, "host_reboot_evidence_dir_ok", str(evidence_dir))
```

- [ ] **Step 4: Run focused tests**

Run the same pytest command. Expected: pass.

- [x] **Step 5: Commit**

```bash
git add tools/macos-vz-helper/scripts/vz-helperctl.py tools/macos-vz-helper/Tests/test_vz_helperctl.py
git commit -m "feat(vz): add host reboot evidence directory checks"
```

## Task 2: Pre-Reboot Manifest

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Test: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`

- [x] **Step 1: Add failing tests for `host-reboot-drill pre`**

Test that dry-run/pre mode writes bounded manifest data when explicitly allowed
to create the evidence directory:

```python
def test_host_reboot_pre_writes_bounded_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "durable" / "drill"
    monkeypatch.setattr(helperctl, "VOLATILE_EVIDENCE_ROOTS", ())

    result = helperctl.run_host_reboot_pre(
        evidence_dir=evidence,
        bundle_path=tmp_path / "bundle",
        helper_mode="direct",
        socket_path=tmp_path / "helper.sock",
        log_dir=tmp_path / "logs",
        create_evidence_dir=True,
        ping_checker=lambda path: helperctl.PingState(
            result=helperctl.CheckResult(True),
            protocol_version="1",
            helper_version="test",
            details={
                "helper_instance_id": "before",
                "helper_started_at": "2026-05-19T00:00:00Z",
            },
        ),
    )

    CASE.assertTrue(result.ok)
    payload = json.loads((evidence / "host-reboot-pre.json").read_text())
    CASE.assertEqual(payload["phase"], "pre")
    CASE.assertEqual(payload["helper_mode"], "direct")
    CASE.assertNotIn("environment", payload)
```

- [x] **Step 2: Run test to verify failure**

Run:

```bash
python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py \
  -k "host_reboot_pre" -q
```

Expected: fail because `run_host_reboot_pre` does not exist.

- [x] **Step 3: Implement pre-phase manifest writer**

Add:

- `HOST_REBOOT_PRE_MANIFEST = "host-reboot-pre.json"`
- `HOST_REBOOT_POST_MANIFEST = "host-reboot-post.json"`
- `write_json_private(path: Path, payload: Mapping[str, Any]) -> CheckResult`
- `run_host_reboot_pre(...) -> CheckResult`
- `ping_state_payload(state: PingState) -> dict[str, Any]`

Manifest payload should include only:

- `phase`
- `created_at`
- `hostname`
- `helper_mode`
- `bundle_path`
- `helper_path`
- `socket_path`
- `log_dir`
- `serial_log_dir`
- `launchd_label`
- `launchd_plist_path`
- `helper_ping_ok`
- `helper_ping_reason`
- `helper_protocol_version`
- `helper_version`
- `helper_details` from `PingState.details or {}`

Do not include raw env vars, stdout/stderr, serial log contents, or workspace
paths beyond configured helper/log/socket/bundle paths.

- [x] **Step 4: Run focused tests**

Run the same `host_reboot_pre` pytest selection. Expected: pass.

- [x] **Step 5: Commit**

```bash
git add tools/macos-vz-helper/scripts/vz-helperctl.py tools/macos-vz-helper/Tests/test_vz_helperctl.py
git commit -m "feat(vz): record host reboot preflight evidence"
```

## Task 3: Post-Reboot Manifest And Generation Drift

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Test: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`

- [x] **Step 1: Add failing post-phase tests**

Add tests for:

- missing pre manifest returns `host_reboot_pre_manifest_missing`
- malformed JSON returns `host_reboot_pre_manifest_invalid`
- helper generation drift returns `helper_generation_changed` but overall result remains ok when ping/protocol are healthy

Example:

```python
def test_host_reboot_post_reports_generation_changed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    evidence.chmod(0o700)
    (evidence / "host-reboot-pre.json").write_text(
        json.dumps({
            "phase": "pre",
            "helper_mode": "direct",
            "helper_details": {"helper_instance_id": "before"},
        }),
        encoding="utf-8",
    )

    results = helperctl.run_host_reboot_post(
        evidence_dir=evidence,
        bundle_path=tmp_path / "bundle",
        helper_mode="direct",
        socket_path=tmp_path / "helper.sock",
        log_dir=tmp_path / "logs",
        ping_checker=lambda path: helperctl.PingState(
            result=helperctl.CheckResult(True),
            details={"helper_instance_id": "after"},
        ),
    )

    by_name = dict(results)
    CASE.assertTrue(by_name["helper_status"].ok)
    CASE.assertEqual(by_name["helper_generation"].reason, "helper_generation_changed")
```

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py \
  -k "host_reboot_post" -q
```

Expected: fail because post helpers do not exist.

- [x] **Step 3: Implement post phase**

Add `run_host_reboot_post(...) -> list[tuple[str, CheckResult]]` that:

- validates evidence directory
- reads pre manifest
- pings helper using the selected socket
- compares helper generation details from pre and post
- writes `host-reboot-post.json`
- returns named results suitable for human and JSON output

Return generation drift as a named ok result:

```python
("helper_generation", CheckResult(ok=True, reason="helper_generation_changed"))
```

- [x] **Step 4: Run focused post tests**

Expected: pass.

- [x] **Step 5: Commit**

```bash
git add tools/macos-vz-helper/scripts/vz-helperctl.py tools/macos-vz-helper/Tests/test_vz_helperctl.py
git commit -m "feat(vz): validate host reboot postflight evidence"
```

## Task 4: CLI Wiring And Restored-Helper Smoke

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Test: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`

- [x] **Step 1: Add failing CLI tests**

Add tests that assert:

- `host-reboot-drill pre --json` emits parseable JSON
- `host-reboot-drill post --json` emits parseable JSON
- `post --run-smoke` calls `run_vz_linux_host_smoke` with the restored helper socket, not `smoke_helper`
- launchd mode requires explicit label/plist or returns a clear reason

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py \
  -k "host_reboot_drill_cli or host_reboot_post_runs_smoke" -q
```

Expected: fail because CLI wiring does not exist.

- [x] **Step 3: Implement CLI parser**

Add subparser:

```text
host-reboot-drill {pre,post}
```

Arguments:

- `--evidence-dir` required
- `--bundle`
- `--helper-mode {direct,launchd}` default `direct`
- `--helper`
- `--socket`
- `--log-dir`
- `--serial-log-dir`
- `--label`
- `--plist-output`
- `--create-evidence-dir`
- `--allow-volatile-evidence-dir`
- `--run-smoke`
- `--python`
- `--dry-run`
- `--json`

For `post --run-smoke`, call `run_vz_linux_host_smoke(bundle_path=..., socket_path=...)`.
Do not call `smoke_helper`, because `smoke_helper` delegates to the helper-owning
script path and can start a separate helper.

- [x] **Step 4: Run focused CLI tests**

Expected: pass.

- [x] **Step 5: Commit**

```bash
git add tools/macos-vz-helper/scripts/vz-helperctl.py tools/macos-vz-helper/Tests/test_vz_helperctl.py
git commit -m "feat(vz): add host reboot drill CLI"
```

## Task 5: Operator Docs And Task Closeout

**Files:**
- Modify: `tools/macos-vz-helper/README.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
- Modify: `backlog/tasks/task-438 - Design-VZ-helper-host-reboot-validation-procedure.md`

- [x] **Step 1: Add docs before final verification**

Document:

- durable evidence directory requirement
- pre/reboot/post sequence
- direct vs launchd helper mode
- post-reboot smoke must target restored helper socket
- diagnostics and dry-run repair remain operator-reviewed
- scheduled CI must not reboot hosts
- expected skip/blocking behavior

- [x] **Step 2: Run focused helperctl tests**

Run:

```bash
python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q
```

Expected: all helperctl tests pass.

- [x] **Step 3: Run syntax and whitespace checks**

Run:

```bash
python -m py_compile tools/macos-vz-helper/scripts/vz-helperctl.py
git diff --check
```

Expected: both pass.

- [x] **Step 4: Run Bandit on touched Python**

Run:

```bash
python -m bandit -r tools/macos-vz-helper/scripts/vz-helperctl.py \
  -f json -o /tmp/bandit_vz_host_reboot_drill.json
```

Expected: `results=[]`, `errors=[]`.

- [x] **Step 5: Record optional prepared-host validation status**

Only on a prepared Apple silicon host:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py host-reboot-drill pre \
  --evidence-dir "$HOME/Library/Logs/tldw/vz-host-reboot-drill/manual-$(date +%Y%m%d-%H%M%S)" \
  --bundle /path/to/canonical/bundle \
  --create-evidence-dir

# manually reboot

tools/macos-vz-helper/scripts/vz-helperctl.py host-reboot-drill post \
  --evidence-dir "$HOME/Library/Logs/tldw/vz-host-reboot-drill/<same-run-id>" \
  --bundle /path/to/canonical/bundle \
  --run-smoke
```

Expected: helper status/protocol pass, generation drift is reported as expected
when applicable, and real smoke passes against the restored helper socket.

Status: not run for Task 5 closeout because it requires a disruptive manual
operator reboot. The docs record this as a manual or explicitly
operator-triggered prepared-host validation only.

Post-review hardening added a host boot marker to the pre/post manifests,
non-mutating lifecycle readiness results for direct and launchd modes, bundle
dry-run validation in the pre phase, and a blocking
`host_reboot_not_detected` result when the post phase sees the same boot marker
recorded before reboot. The host reboot drill now fails closed when it cannot
prove the host reboot boundary it is meant to validate.

- [x] **Step 6: Update Backlog and commit**

Record verification and known host-gated skips in TASK-438 and TASK-443, then
commit the docs/task closeout only:

```bash
git add \
  tools/macos-vz-helper/README.md \
  Docs/Sandbox/macos-runtime-operator-notes.md \
  Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md \
  Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md \
  "backlog/tasks/task-438 - Design-VZ-helper-host-reboot-validation-procedure.md" \
  "backlog/tasks/task-443 - Document-VZ-host-reboot-validation-drill-workflow.md"
git commit -m "docs(vz): document host reboot validation drill"
```

## Final Verification For PR

Run before opening a PR:

```bash
source .venv/bin/activate
python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q
python -m py_compile tools/macos-vz-helper/scripts/vz-helperctl.py
python -m bandit -r tools/macos-vz-helper/scripts/vz-helperctl.py -f json -o /tmp/bandit_vz_host_reboot_drill.json
git diff --check
```

If the prepared host is available, also run the manual pre/post flow and record
the evidence directory path in the PR notes. If it is not available, document
that real reboot validation remains host-gated and was not run locally.
