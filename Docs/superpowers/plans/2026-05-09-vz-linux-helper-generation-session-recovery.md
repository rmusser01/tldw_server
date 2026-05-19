# VZ Linux Helper Generation Session Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `vz_linux` session reuse detect helper-generation drift and preserve persisted session-control state when helper truth is unavailable or protocol-incompatible.

**Architecture:** The Swift helper owns live generation truth and exposes per-process generation details in existing protocol `details` dictionaries. Python persists that generation with VZ session-control rows and reuses a VM only when live helper status, ownership metadata, session metadata, and generation all agree.

**Tech Stack:** Swift 5.9 helper package, Python sandbox runner/store, SQLite/Postgres store migrations, pytest, Swift Testing, Bandit.

---

## File Map

- Modify `tools/macos-vz-helper/Sources/Server/HelperService.swift`: add helper generation fields and attach them to ping/create/status/list details.
- Modify `tools/macos-vz-helper/Tests/PingTests.swift`: assert helper generation appears and is stable for one helper instance.
- Modify `tools/macos-vz-helper/Tests/HelperServiceVMTests.swift`: assert create/status/list VM details include helper generation.
- Modify `tools/macos-vz-helper/PROTOCOL.md`: document `helper_instance_id` and `helper_started_at` details.
- Modify `tldw_Server_API/app/core/Sandbox/store.py`: extend VZ session-control abstract/in-memory/SQLite/Postgres stores and migrations.
- Modify `tldw_Server_API/app/core/Sandbox/orchestrator.py`: pass through optional helper-generation fields.
- Modify `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`: add reuse eligibility helpers and persist generation after create.
- Modify `tldw_Server_API/tests/sandbox/test_vz_linux_session_control_store.py`: verify persisted generation fields.
- Modify `tldw_Server_API/tests/sandbox/test_store_sqlite_migrations.py`: verify migration adds VZ generation columns.
- Modify `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`: add and update reuse/fail-closed tests.
- Modify `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`: verify fake/helper detail parsing if needed.
- Modify `tldw_Server_API/app/core/Sandbox/README.md`: add minimal operator note for generation-aware reuse.
- Modify `backlog/tasks/task-160 - Harden-vz_linux-helper-generation-session-recovery.md`: keep status, notes, verification, and final summary current.

---

### Task 1: Add Helper-Owned Generation Details

**Files:**
- Modify: `tools/macos-vz-helper/Sources/Server/HelperService.swift`
- Modify: `tools/macos-vz-helper/Tests/PingTests.swift`
- Modify: `tools/macos-vz-helper/Tests/HelperServiceVMTests.swift`
- Modify: `tools/macos-vz-helper/PROTOCOL.md`

- [ ] **Step 1: Write failing Swift tests for helper generation**

Add tests that instantiate `HelperService(helperInstanceID: "helper-test-1", helperStartedAt: "2026-05-09T00:00:00Z")` and assert:

```swift
#expect(response.details["helper_instance_id"] == "helper-test-1")
#expect(response.details["helper_started_at"] == "2026-05-09T00:00:00Z")
```

Cover `ping()`, `createVM(...)`, `getVMStatus(...)`, and `listVMs()`.

- [ ] **Step 2: Run Swift tests and verify they fail**

Run:

```bash
cd tools/macos-vz-helper
swift test --filter 'PingTests|HelperServiceVMTests'
```

Expected: FAIL because `HelperService` has no injectable helper generation and details omit the new keys.

- [ ] **Step 3: Implement helper generation details**

Add stored properties to `HelperService`:

```swift
private let helperInstanceID: String
private let helperStartedAt: String
```

Extend `init(...)` with defaulted parameters:

```swift
helperInstanceID: String = UUID().uuidString,
helperStartedAt: String = ISO8601DateFormatter().string(from: Date())
```

Add helper methods:

```swift
private func helperGenerationDetails() -> [String: String] {
    [
        "helper_instance_id": helperInstanceID,
        "helper_started_at": helperStartedAt,
    ]
}

private func withHelperGeneration(_ details: [String: String]) -> [String: String] {
    var merged = details
    for (key, value) in helperGenerationDetails() {
        merged[key] = value
    }
    return merged
}
```

Use `withHelperGeneration(...)` in `ping()` and `vmDetails(for:)`.

- [ ] **Step 4: Update helper protocol docs**

In `tools/macos-vz-helper/PROTOCOL.md`, document that helper response `details` may include:

```text
helper_instance_id: per-helper-process UUID, changes after helper restart
helper_started_at: helper process start timestamp in ISO-8601 format
```

- [ ] **Step 5: Run Swift tests and verify they pass**

Run:

```bash
cd tools/macos-vz-helper
swift test --filter 'PingTests|HelperServiceVMTests'
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/macos-vz-helper/Sources/Server/HelperService.swift tools/macos-vz-helper/Tests/PingTests.swift tools/macos-vz-helper/Tests/HelperServiceVMTests.swift tools/macos-vz-helper/PROTOCOL.md
git commit -m "feat: expose vz helper generation details"
```

---

### Task 2: Persist Helper Generation In VZ Session Control

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/store.py`
- Modify: `tldw_Server_API/app/core/Sandbox/orchestrator.py`
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_session_control_store.py`
- Modify: `tldw_Server_API/tests/sandbox/test_store_sqlite_migrations.py`

- [ ] **Step 1: Write failing store tests**

Extend `test_store_persists_vz_linux_session_control_metadata()` to call:

```python
store_a.put_vz_session_control(
    session_id="sess-1",
    runtime="vz_linux",
    vm_id="vm-session-1",
    template_id="vz_linux:ubuntu-24.04",
    workspace_mount="/tmp/ws",
    agent_ready=True,
    helper_instance_id="helper-a",
    helper_started_at="2026-05-09T00:00:00Z",
)
```

Assert the returned row includes both values.

Extend `test_sqlite_store_migrations_add_new_columns()` to assert
`sandbox_vz_sessions` includes `helper_instance_id` and `helper_started_at`.

- [ ] **Step 2: Run the focused failing tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_session_control_store.py tldw_Server_API/tests/sandbox/test_store_sqlite_migrations.py -q
```

Expected: FAIL because store signatures/schema do not include the generation fields.

- [ ] **Step 3: Update store interfaces and in-memory implementation**

In `SandboxStore.put_vz_session_control(...)`, `InMemoryStore.put_vz_session_control(...)`, and returned in-memory rows, add optional keyword-only parameters:

```python
helper_instance_id: str | None = None,
helper_started_at: str | None = None,
```

Persist normalized strings or `None`:

```python
"helper_instance_id": (str(helper_instance_id or "").strip() or None),
"helper_started_at": (str(helper_started_at or "").strip() or None),
```

- [ ] **Step 4: Update SQLite store schema and migrations**

Add `helper_instance_id TEXT` and `helper_started_at TEXT` to `CREATE TABLE IF NOT EXISTS sandbox_vz_sessions`.

Add migrations:

```python
_ensure_sqlite_column("sandbox_vz_sessions", "helper_instance_id", "TEXT")
_ensure_sqlite_column("sandbox_vz_sessions", "helper_started_at", "TEXT")
```

Include the columns in `INSERT`, `ON CONFLICT`, `SELECT`, and `list_vz_session_controls()`.

- [ ] **Step 5: Update Postgres store schema and migrations**

Add `helper_instance_id TEXT` and `helper_started_at TEXT` to the Postgres table definition, `_ensure_column(...)` calls, `INSERT`, `ON CONFLICT`, `SELECT`, and `list_vz_session_controls()`.

- [ ] **Step 6: Update orchestrator facade**

Add the same optional parameters to `SandboxOrchestrator.put_vz_session_control(...)` and pass them to the store.

- [ ] **Step 7: Run store tests and verify they pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_session_control_store.py tldw_Server_API/tests/sandbox/test_store_sqlite_migrations.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/store.py tldw_Server_API/app/core/Sandbox/orchestrator.py tldw_Server_API/tests/sandbox/test_vz_linux_session_control_store.py tldw_Server_API/tests/sandbox/test_store_sqlite_migrations.py
git commit -m "feat: persist vz helper generation metadata"
```

---

### Task 3: Harden Runner Reuse Eligibility

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`
- Optionally modify: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`

- [ ] **Step 1: Write failing runner tests**

Add or update tests in `test_vz_linux_runner.py`:

- healthy same-generation reuse includes stored `helper_instance_id`, status `details["helper_instance_id"]`, matching status metadata owner/runtime/session, and does not call create.
- generation mismatch deletes stale row, creates a fresh VM, stores the replacement generation, and completes.
- helper unavailable during reuse returns failed status and does not delete or put session-control state.
- protocol mismatch during reuse returns failed status and does not delete or put session-control state.
- missing generation with matching live `tldw/vz_linux` session metadata follows the explicit legacy behavior from the spec.

Use `HelperVMMetadata` on fake statuses:

```python
metadata=HelperVMMetadata(
    owner="tldw",
    runtime="vz_linux",
    session_id="sess-1",
    session_mode=True,
)
```

- [ ] **Step 2: Run the focused failing runner tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py -q
```

Expected: FAIL because current reuse only checks `healthy` and deletes rows on helper unavailable/protocol mismatch.

- [ ] **Step 3: Add normalization helpers**

In `VZLinuxRunner`, add small private helpers:

```python
@staticmethod
def _clean_generation_value(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
```

Add helpers to extract generation from session-control rows and helper status/details:

```python
def _session_generation(self, row: dict[str, Any]) -> tuple[str | None, str | None]
def _status_generation(self, status: Any) -> tuple[str | None, str | None]
```

- [ ] **Step 4: Add reuse predicate**

Add `_can_reuse_session_vm(...)` that returns `True` only when:

- status is present and `healthy` is true
- metadata owner/runtime is `tldw`/`vz_linux`
- non-empty metadata session ID matches requested session ID
- stored and live `helper_instance_id` match when both are present
- stored and live `helper_started_at` match when both are present
- missing generation follows the spec's conservative legacy path

- [ ] **Step 5: Change helper failure handling**

In `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`, add `MacOSVirtualizationHelperProtocolError` explicitly to `_VZ_LINUX_RUNNER_NONCRITICAL_EXCEPTIONS` next to `MacOSVirtualizationHelperUnavailable`. This is intentionally explicit even though the class currently subclasses `RuntimeError`, because the fail-closed contract should survive future exception tuple narrowing.

In the reuse block, do not convert `MacOSVirtualizationHelperUnavailable` or `MacOSVirtualizationHelperProtocolError` into `status = None`. Let them propagate to the existing outer failure path so the run fails and the row is preserved.

For reachable `None` or unhealthy status, call `_delete_session_control(...)` and provision a fresh VM.

- [ ] **Step 6: Persist generation after create**

Update `_store_session_control(...)` to accept optional `helper_instance_id` and `helper_started_at`. Pass values extracted from `vm.details` after `helper.create_vm(...)`.

- [ ] **Step 7: Run runner tests and verify they pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py
git commit -m "fix: harden vz linux session reuse generation checks"
```

---

### Task 4: Document Operator Contract

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`
- Modify: `backlog/tasks/task-160 - Harden-vz_linux-helper-generation-session-recovery.md`

- [ ] **Step 1: Update sandbox README**

Add a concise `vz_linux` session reuse note:

```markdown
Session VM reuse is generation-aware. The runner reuses a persisted VM only
when helper status is healthy, live ownership metadata matches the requested
session, and helper generation metadata matches the stored row. If the helper is
unavailable or protocol-incompatible, the run fails closed and preserves the row
for explicit recovery/repair instead of deleting it.
```

- [ ] **Step 2: Update TASK-160 notes**

Record the implemented files and focused verification commands in the Backlog task.

- [ ] **Step 3: Run docs diff checks**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/README.md 'backlog/tasks/task-160 - Harden-vz_linux-helper-generation-session-recovery.md'
git commit -m "docs: describe vz helper generation reuse"
```

---

### Task 5: Final Verification And PR Preparation

**Files:**
- Modify: `backlog/tasks/task-160 - Harden-vz_linux-helper-generation-session-recovery.md`

- [ ] **Step 1: Run focused Python verification**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_vz_linux_session_control_store.py tldw_Server_API/tests/sandbox/test_store_sqlite_migrations.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py -q
```

Expected: PASS.

- [ ] **Step 2: Run focused Swift verification**

Run:

```bash
cd tools/macos-vz-helper
swift test --filter 'PingTests|HelperServiceVMTests'
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched Python code**

Run:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py tldw_Server_API/app/core/Sandbox/store.py tldw_Server_API/app/core/Sandbox/orchestrator.py -f json -o /tmp/bandit_vz_helper_generation_recovery.json
```

Expected: zero new findings.

- [ ] **Step 4: Run diff checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended task/spec/plan/code/doc changes before final commit.

- [ ] **Step 5: Update TASK-160 final summary**

Record verification results, known skips, and any host-gated real-VZ smoke that was not run.

- [ ] **Step 6: Final commit**

```bash
git add 'backlog/tasks/task-160 - Harden-vz_linux-helper-generation-session-recovery.md'
git commit -m "chore: close vz helper generation recovery task"
```

- [ ] **Step 7: Prepare PR**

After all commits are ready:

```bash
git status --short
git log --oneline origin/dev..HEAD
```

Expected: clean worktree with a short, reviewable commit stack.
