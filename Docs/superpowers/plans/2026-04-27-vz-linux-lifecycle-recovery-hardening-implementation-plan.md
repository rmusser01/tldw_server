# vz_linux Lifecycle Recovery Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `vz_linux` lifecycle recovery deterministic by separating read-only reconciliation from explicit admin repair, classifying helper compatibility failures, and refreshing stale operator docs.

**Architecture:** Add a focused Python reconciliation module that consumes helper-owned VM truth and Python-owned session-control metadata. Wire diagnostics to the read-only report, add a separate admin repair operation for safe stale-row deletion, keep orphan VM termination deferred, and tighten runner/helper error classification without expanding helper ownership.

**Tech Stack:** Python 3, FastAPI, Pydantic, pytest, Loguru, existing macOS Virtualization helper protocol client.

---

## Source References

- Spec: `Docs/superpowers/specs/2026-04-27-vz-linux-lifecycle-recovery-hardening-design.md`
- Doctrine: `Docs/Sandbox/sandbox-architecture-doctrine.md`
- Current diagnostics: `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`
- Current helper client: `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
- Current runner: `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
- Current admin endpoint: `tldw_Server_API/app/api/v1/endpoints/sandbox.py`
- Current API schemas: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`

## File Structure

- Create `tldw_Server_API/app/core/Sandbox/vz_reconciliation.py`
  - One responsibility: build read-only reconciliation reports and repair plans from persisted VZ session-control rows plus helper VM state.
- Create `tldw_Server_API/tests/sandbox/test_vz_reconciliation.py`
  - Unit coverage for reconciliation categories and helper/protocol failure classification.
- Modify `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`
  - Delegate reconciliation to the new module and classify helper protocol mismatch consistently.
- Modify `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
  - Add additive reconciliation detail fields and repair request/response models.
- Modify `tldw_Server_API/app/core/Sandbox/service.py`
  - Add explicit admin repair service method; keep diagnostics read-only.
- Modify `tldw_Server_API/app/api/v1/endpoints/sandbox.py`
  - Add admin-only repair endpoint and error mapping.
- Modify `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
  - Make helper unavailable/protocol mismatch paths explicit during session reuse.
- Modify tests:
  - `tldw_Server_API/tests/sandbox/test_macos_diagnostics.py`
  - `tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py`
  - `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`
  - Create `tldw_Server_API/tests/sandbox/test_admin_macos_reconciliation_repair.py`
- Modify docs:
  - `tldw_Server_API/app/core/Sandbox/README.md`
  - `Docs/Sandbox/macos-runtime-operator-notes.md`
  - Optionally `tools/macos-vz-helper/README.md`

## Task 1: Add Read-Only VZ Reconciliation Module

**Files:**
- Create: `tldw_Server_API/app/core/Sandbox/vz_reconciliation.py`
- Create: `tldw_Server_API/tests/sandbox/test_vz_reconciliation.py`

- [x] **Step 1: Write failing tests for reconciliation categories**

Create `tldw_Server_API/tests/sandbox/test_vz_reconciliation.py` with fake orchestrator/helper classes.

Test cases:

```python
def test_reconciliation_reports_healthy_stale_unhealthy_and_orphaned_vms():
    # persisted:
    # sess-live -> vm-live healthy
    # sess-stale -> vm-missing
    # sess-unhealthy -> vm-unhealthy live but healthy=False
    # live helper also has vm-orphan
    report = collect_vz_reconciliation(
        orchestrator=_FakeOrchestrator(...),
        helper_client=_FakeHelper(...),
    )

    assert report["computed"] is True
    assert report["persisted_sessions"] == 3
    assert report["live_vms"] == 3
    assert report["healthy_session_ids"] == ["sess-live"]
    assert report["stale_session_ids"] == ["sess-stale"]
    assert report["unhealthy_session_ids"] == ["sess-unhealthy"]
    assert report["orphaned_vm_ids"] == ["vm-orphan"]
    assert {item["status"] for item in report["items"]} >= {
        "healthy",
        "stale_session",
        "unhealthy_vm",
        "orphaned_vm",
    }
```

Also add tests:

```python
def test_reconciliation_classifies_helper_unavailable():
    ...
    assert report["computed"] is False
    assert "macos_virtualization_helper_unavailable" in report["reasons"]


def test_reconciliation_classifies_protocol_mismatch():
    ...
    assert report["computed"] is False
    assert "macos_virtualization_helper_protocol_mismatch" in report["reasons"]


def test_reconciliation_marks_active_stale_sessions_as_skipped():
    report = collect_vz_reconciliation(..., active_session_checker=lambda sid: sid == "sess-active")
    assert report["skipped_active_session_ids"] == ["sess-active"]
    assert any(item["status"] == "skipped_active_session" for item in report["items"])
```

- [x] **Step 2: Run tests and verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_reconciliation.py -v
```

Expected: fail with import error for `vz_reconciliation` or missing `collect_vz_reconciliation`.

- [x] **Step 3: Implement `vz_reconciliation.py`**

Implement a side-effect-free module:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger

from .macos_virtualization.helper_client import (
    MacOSVirtualizationHelperClient,
    MacOSVirtualizationHelperProtocolError,
    MacOSVirtualizationHelperUnavailable,
)

REASON_HELPER_UNAVAILABLE = "macos_virtualization_helper_unavailable"
REASON_PROTOCOL_MISMATCH = "macos_virtualization_helper_protocol_mismatch"
REASON_RECONCILIATION_UNAVAILABLE = "vz_reconciliation_unavailable"


def collect_vz_reconciliation(
    orchestrator: Any | None,
    *,
    helper_client: Any | None = None,
    active_session_checker: Callable[[str], bool] | None = None,
) -> dict[str, object]:
    ...
```

Output shape must preserve existing fields:

```python
{
    "computed": False,
    "persisted_sessions": 0,
    "live_vms": 0,
    "healthy_session_ids": [],
    "stale_session_ids": [],
    "unhealthy_session_ids": [],
    "skipped_active_session_ids": [],
    "orphaned_vm_ids": [],
    "items": [],
    "reasons": [],
}
```

Implementation rules:

- Return `computed=False` if `orchestrator` is missing or lacks `list_vz_session_controls`.
- Catch `MacOSVirtualizationHelperUnavailable` and return reason `macos_virtualization_helper_unavailable`.
- Catch `MacOSVirtualizationHelperProtocolError` and return reason `macos_virtualization_helper_protocol_mismatch`.
- Do not call delete/terminate APIs.
- Sort all ID lists for deterministic tests.
- Include per-item dictionaries with `status`, `session_id`, `vm_id`, `state`, `healthy`, and `reason` where available.

- [x] **Step 4: Run Task 1 tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_reconciliation.py -v
```

Expected: all Task 1 tests pass.

- [x] **Step 5: Commit Task 1**

```bash
git add tldw_Server_API/app/core/Sandbox/vz_reconciliation.py tldw_Server_API/tests/sandbox/test_vz_reconciliation.py
git commit -m "feat(sandbox): add vz reconciliation report"
```

## Task 2: Wire Diagnostics To Reconciliation Truth

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
- Modify: `tldw_Server_API/tests/sandbox/test_macos_diagnostics.py`
- Modify: `tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py`

- [x] **Step 1: Write failing diagnostics/schema tests**

Update `test_collect_macos_diagnostics_reports_reconciliation_mismatches` to assert additive fields:

```python
assert data["reconciliation"]["healthy_session_ids"] == ["sess-live"]
assert data["reconciliation"]["stale_session_ids"] == ["sess-stale"]
assert data["reconciliation"]["items"]
```

Add a protocol mismatch test:

```python
def test_collect_macos_diagnostics_classifies_helper_protocol_mismatch(monkeypatch):
    class _FakeHelper:
        def ping(self):
            raise MacOSVirtualizationHelperProtocolError("macos_virtualization_helper_protocol_mismatch")

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _FakeHelper)

    data = diagnostics_module.collect_macos_diagnostics()

    assert "macos_virtualization_helper_protocol_mismatch" in data["helper"]["reasons"]
```

Update schema tests so `SandboxAdminMacOSDiagnosticsResponse` accepts:

- `healthy_session_ids`
- `unhealthy_session_ids`
- `skipped_active_session_ids`
- `items`

- [x] **Step 2: Run tests and verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/sandbox/test_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py \
  -v
```

Expected: fail on missing fields or uncaught protocol mismatch.

- [x] **Step 3: Update Pydantic schemas**

In `sandbox_schemas.py`, add:

```python
class SandboxAdminMacOSReconciliationItem(BaseModel):
    status: str
    session_id: str | None = None
    vm_id: str | None = None
    state: str | None = None
    healthy: bool | None = None
    reason: str | None = None
```

Extend `SandboxAdminMacOSReconciliationDiagnostics` additively:

```python
healthy_session_ids: list[str] = Field(default_factory=list)
unhealthy_session_ids: list[str] = Field(default_factory=list)
skipped_active_session_ids: list[str] = Field(default_factory=list)
items: list[SandboxAdminMacOSReconciliationItem] = Field(default_factory=list)
```

- [x] **Step 4: Update diagnostics implementation**

In `macos_diagnostics.py`:

- Import `MacOSVirtualizationHelperProtocolError`.
- Import `collect_vz_reconciliation`.
- Catch protocol mismatch in `probe_helper()` and append `macos_virtualization_helper_protocol_mismatch`.
- Catch protocol mismatch in `_vz_linux_template_status()` and append the same reason.
- Replace `probe_reconciliation()` internals with:

```python
return collect_vz_reconciliation(orchestrator)
```

Keep diagnostics read-only.

- [x] **Step 5: Run diagnostics tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/sandbox/test_vz_reconciliation.py \
  tldw_Server_API/tests/sandbox/test_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py \
  -v
```

Expected: all pass.

- [x] **Step 6: Commit Task 2**

```bash
git add \
  tldw_Server_API/app/core/Sandbox/macos_diagnostics.py \
  tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py \
  tldw_Server_API/tests/sandbox/test_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py
git commit -m "feat(sandbox): surface vz reconciliation diagnostics"
```

## Task 3: Add Explicit Admin Repair Operation

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sandbox.py`
- Create: `tldw_Server_API/tests/sandbox/test_admin_macos_reconciliation_repair.py`

- [x] **Step 1: Write failing service/API tests**

Create `test_admin_macos_reconciliation_repair.py`.

Endpoint tests should follow `test_admin_macos_diagnostics.py` app override style.

Required tests:

```python
def test_admin_reconciliation_repair_defaults_to_dry_run(monkeypatch):
    fake_service = SimpleNamespace(
        repair_macos_reconciliation=lambda **kwargs: {
            "dry_run": kwargs["dry_run"],
            "helper": {"ready": True, "protocol_version": "1", "helper_version": "0.1.0"},
            "summary": {...},
            "actions": [{"type": "delete_session_control", "status": "planned"}],
            "reasons": [],
        }
    )
    ...
    resp = client.post("/api/v1/sandbox/admin/macos-reconciliation/repair", json={})
    assert resp.status_code == 200
    assert resp.json()["dry_run"] is True
```

Also test:

- `dry_run=false` passes through and can return `status="deleted"`.
- `terminate_orphaned_vms=true` returns HTTP 400 with `orphan_termination_not_supported`.
- non-admin cannot access endpoint.

Service-level tests should instantiate `SandboxService`, monkeypatch `collect_vz_reconciliation` imported in service, and verify:

- stale row dry-run plans deletion but does not call `delete_vz_session_control`
- stale row with `dry_run=False` calls `delete_vz_session_control`
- active session item is skipped
- helper unavailable/protocol mismatch with `dry_run=False` raises a service error that endpoint maps to 503

- [x] **Step 2: Run tests and verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_admin_macos_reconciliation_repair.py -v
```

Expected: fail because schemas/service/endpoint do not exist.

- [x] **Step 3: Add schemas**

In `sandbox_schemas.py`, add request and response models near admin diagnostics:

```python
class SandboxAdminMacOSReconciliationRepairRequest(BaseModel):
    delete_stale_session_controls: bool = True
    delete_unhealthy_session_controls: bool = True
    terminate_orphaned_vms: bool = False
    dry_run: bool = True


class SandboxAdminMacOSReconciliationRepairAction(BaseModel):
    type: str
    session_id: str | None = None
    vm_id: str | None = None
    status: str
    reason: str | None = None


class SandboxAdminMacOSReconciliationRepairSummary(BaseModel):
    stale_session_controls: int = 0
    unhealthy_session_controls: int = 0
    deleted_session_controls: int = 0
    skipped_active_sessions: int = 0
    orphaned_vms: int = 0
    terminated_orphaned_vms: int = 0


class SandboxAdminMacOSReconciliationRepairResponse(BaseModel):
    dry_run: bool
    helper: dict[str, object] = Field(default_factory=dict)
    summary: SandboxAdminMacOSReconciliationRepairSummary
    actions: list[SandboxAdminMacOSReconciliationRepairAction] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
```

- [x] **Step 4: Add service repair method**

In `service.py`:

- Import `collect_vz_reconciliation`.
- Add small exception class:

```python
class SandboxReconciliationRepairError(RuntimeError):
    def __init__(self, reason: str, status_code: int = 503) -> None:
        self.reason = reason
        self.status_code = int(status_code)
        super().__init__(reason)
```

- Add method:

```python
def repair_macos_reconciliation(
    self,
    *,
    delete_stale_session_controls: bool = True,
    delete_unhealthy_session_controls: bool = True,
    terminate_orphaned_vms: bool = False,
    dry_run: bool = True,
) -> dict[str, object]:
    ...
```

Rules:

- If `terminate_orphaned_vms` is true, raise `SandboxReconciliationRepairError("orphan_termination_not_supported", 400)`.
- Build report using `collect_vz_reconciliation(self._orch, active_session_checker=lambda sid: self._active_session_run_count(sid) > 0)`.
- If report has `macos_virtualization_helper_unavailable` or `macos_virtualization_helper_protocol_mismatch` and `dry_run=False`, raise 503.
- Plan delete actions only for `stale_session` and `unhealthy_vm` items, based on request flags.
- Skip any item with active session status.
- If not dry-run, call `self._orch.delete_vz_session_control(session_id)`.
- Log each planned/deleted/skipped action with Loguru.

Do not terminate orphaned VMs.

- [x] **Step 5: Add endpoint**

In `endpoints/sandbox.py`:

- Import the new request/response schemas.
- Import `SandboxReconciliationRepairError`.
- Add:

```python
@router.post(
    "/admin/macos-reconciliation/repair",
    response_model=SandboxAdminMacOSReconciliationRepairResponse,
    summary="Admin: repair macOS sandbox reconciliation state",
)
async def admin_repair_macos_reconciliation(
    request: SandboxAdminMacOSReconciliationRepairRequest = Body(default_factory=SandboxAdminMacOSReconciliationRepairRequest),
    _principal: AuthPrincipal = Depends(auth_deps.require_roles("admin")),
    _current_user: User = Depends(get_request_user),
) -> SandboxAdminMacOSReconciliationRepairResponse:
    try:
        payload = _service.repair_macos_reconciliation(**request.model_dump())
    except SandboxReconciliationRepairError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.reason) from exc
    return SandboxAdminMacOSReconciliationRepairResponse.model_validate(payload)
```

- [x] **Step 6: Run repair tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/sandbox/test_admin_macos_reconciliation_repair.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py \
  -v
```

Expected: all pass.

- [x] **Step 7: Commit Task 3**

```bash
git add \
  tldw_Server_API/app/core/Sandbox/service.py \
  tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/sandbox.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_reconciliation_repair.py
git commit -m "feat(sandbox): add explicit vz reconciliation repair"
```

## Task 4: Tighten Session Reuse Failure Classification

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`

- [x] **Step 1: Write failing runner tests**

Add tests:

```python
def test_vz_linux_session_reuse_helper_unavailable_does_not_delete_control(monkeypatch, tmp_path):
    deleted = []
    class _Store:
        def get_vz_session_control(self, session_id): ...
        def delete_vz_session_control(self, session_id):
            deleted.append(session_id)

    class _FakeHelper:
        def get_vm_status(self, vm_id):
            raise MacOSVirtualizationHelperUnavailable("macos_virtualization_helper_unavailable")

    status = VZLinuxRunner(session_control_store=_Store()).start_run(...)
    assert status.phase == RunPhase.failed
    assert "macos_virtualization_helper_unavailable" in status.message
    assert deleted == []
```

And:

```python
def test_vz_linux_session_reuse_protocol_mismatch_does_not_delete_control(...):
    class _FakeHelper:
        def get_vm_status(self, vm_id):
            raise MacOSVirtualizationHelperProtocolError("macos_virtualization_helper_protocol_mismatch")
    ...
```

- [x] **Step 2: Run tests and verify current behavior**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py -v
```

Expected: tests may already pass for deletion behavior but may fail on explicit protocol-mismatch message if not imported/caught clearly. Keep them as regression coverage either way.

- [x] **Step 3: Make runner intent explicit**

In `vz_linux_runner.py`:

- Import `MacOSVirtualizationHelperProtocolError`.
- Include it in the runner noncritical exception tuple if needed.
- Around `helper.get_vm_status(candidate_vm_id)`, do not catch helper unavailable/protocol mismatch and do not call `_delete_session_control` in those cases.
- Ensure final failure message contains the underlying helper reason.

The behavior must stay:

- missing/unhealthy status object: delete row and recreate
- helper unavailable exception: fail without deleting
- protocol mismatch exception: fail without deleting

- [x] **Step 4: Run runner tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py -v
```

Expected: all pass.

- [x] **Step 5: Commit Task 4**

```bash
git add tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py tldw_Server_API/tests/sandbox/test_vz_linux_runner.py
git commit -m "fix(sandbox): fail closed on vz session helper mismatch"
```

## Task 5: Refresh macOS Sandbox Runtime Docs

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Optionally modify: `tools/macos-vz-helper/README.md`

- [x] **Step 1: Update stale current-state language**

Remove or replace claims that:

- actual `Virtualization.framework` boot driver is incomplete
- real vsock binding is incomplete

Replace with:

- real `vz_linux` execution uses the Swift helper `VirtualizationLinuxBootDriver`
- helper owns vsock session management
- guest `tldw-agent` connects over vsock
- operator smoke validates ephemeral execution and same-session reuse

- [x] **Step 2: Document reconciliation split**

Add documentation that:

- diagnostics are read-only
- `GET /api/v1/sandbox/admin/macos-diagnostics` reports reconciliation outcomes
- `POST /api/v1/sandbox/admin/macos-reconciliation/repair` is the explicit repair path
- dry-run is default
- orphan VM termination is report-only until helper VM metadata is richer

- [x] **Step 3: Run doc grep checks**

Run:

```bash
rg -n "boot path.*incomplete|vsock.*incomplete|real vsock transport binding.*incomplete|Virtualization.framework.*incomplete" \
  tldw_Server_API/app/core/Sandbox/README.md Docs/Sandbox/macos-runtime-operator-notes.md tools/macos-vz-helper/README.md
```

Expected: no stale incomplete claims remain, except if explicitly described as historical context.

- [x] **Step 4: Commit Task 5**

```bash
git add \
  tldw_Server_API/app/core/Sandbox/README.md \
  Docs/Sandbox/macos-runtime-operator-notes.md \
  tools/macos-vz-helper/README.md
git commit -m "docs(sandbox): refresh vz linux lifecycle recovery notes"
```

If `tools/macos-vz-helper/README.md` was not changed, omit it from `git add`.

## Task 6: Full Verification And Security Pass

**Files:**
- No source changes unless verification finds issues.

- [x] **Step 1: Run focused sandbox tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/sandbox/test_vz_reconciliation.py \
  tldw_Server_API/tests/sandbox/test_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_reconciliation_repair.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_runner.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_session_cleanup.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_session_lifecycle.py \
  -v
```

Expected: all pass.

- [x] **Step 2: Run helper-client regression tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py \
  -v
```

Expected: all pass.

- [x] **Step 3: Run Bandit on touched code**

Run:

```bash
source .venv/bin/activate && python -m bandit \
  -r \
  tldw_Server_API/app/core/Sandbox \
  tldw_Server_API/app/api/v1/endpoints/sandbox.py \
  tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py \
  -f json \
  -o /tmp/bandit_vz_lifecycle_recovery.json
```

Expected: no new high/medium findings in touched code. If Bandit is not installed, install/use the project dev environment only after approval if needed.

- [x] **Step 4: Run formatting/whitespace checks**

Run:

```bash
git diff --check
```

Expected: no output.

- [x] **Step 5: Review final diff**

Run:

```bash
git status --short
git diff --stat dev...HEAD
git diff dev...HEAD -- tldw_Server_API/app/core/Sandbox tldw_Server_API/app/api/v1 Docs/Sandbox tldw_Server_API/app/core/Sandbox/README.md tools/macos-vz-helper/README.md
```

Check:

- diagnostics path is read-only
- repair endpoint is admin-only
- orphan VM termination is rejected
- helper unavailable/protocol mismatch does not delete metadata
- docs match current real VZ/vsock state

- [x] **Step 6: Final commit if needed**

If verification fixes were needed:

```bash
git add <changed-files>
git commit -m "test(sandbox): verify vz lifecycle recovery hardening"
```

If no changes were needed, do not create an empty commit.

## Completion Criteria

- Reconciliation report is reusable outside diagnostics.
- Diagnostics remain read-only.
- Explicit admin repair exists and defaults to dry-run.
- Repair deletes only stale/unhealthy session-control rows, only when requested, and skips active sessions.
- Orphan VM termination is rejected/report-only.
- Helper unavailable and protocol mismatch are explicit fail-closed states.
- Runner tests prove helper unavailable/protocol mismatch do not delete session metadata.
- Docs no longer claim real VZ boot/vsock are incomplete.
- Focused tests and Bandit pass or any inability to run them is documented before PR creation.
