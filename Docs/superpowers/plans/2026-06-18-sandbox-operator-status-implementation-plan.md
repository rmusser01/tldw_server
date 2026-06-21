# Sandbox Operator Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Slice 1 of the sandbox operator/admin status consolidation design as a portable, read-only admin status projection.

**Architecture:** Add a focused `operator_status.py` projection module that accepts already-collected runtime diagnostics, macOS diagnostics, and startup warning summary payloads, then returns a stable operator status dictionary. `SandboxService.operator_status()` should orchestrate the existing sources and isolate section failures; the FastAPI endpoint should only validate/authenticate and call that service method.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, pytest, existing sandbox service and diagnostics modules.

**Command note:** This worktree may not contain its own `.venv`. If `source .venv/bin/activate` fails, use the existing repo virtualenv at `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate` before running Python commands.

---

## File Structure

- Create `tldw_Server_API/app/core/Sandbox/operator_status.py`
  - Owns pure projection helpers, status/action enums as string constants, section builders, overall status classification, and exception-to-section fallback.
  - Must not import helper clients, launchd tooling, image-store mutation, repair mutation, or runner execution code.
- Modify `tldw_Server_API/app/core/Sandbox/service.py`
  - Add `operator_status(startup_warning_summary: dict[str, object] | None = None) -> dict[str, object]`.
  - Gather `runtime_diagnostics_summary()` and `macos_diagnostics()` through existing service methods.
  - Catch section failures and pass fallback section state to `operator_status.py`.
- Modify `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
  - Add response models for operator status, sections, action records, and section summaries.
  - Do not add `generated_at`.
- Modify `tldw_Server_API/app/api/v1/endpoints/sandbox.py`
  - Import the new schema.
  - Add `GET /api/v1/sandbox/admin/operator-status`.
  - Keep `RequireRole("admin")` and `get_request_user`.
  - Offload the service call with `asyncio.to_thread`, matching `runtime-diagnostics`.
- Create `tldw_Server_API/tests/sandbox/test_operator_status.py`
  - Unit tests for pure projection and service method behavior.
- Modify `tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py`
  - Endpoint test for the new admin route.
- Modify `tldw_Server_API/tests/sandbox/test_admin_rbac.py`
  - Add `/api/v1/sandbox/admin/operator-status` to admin-only coverage.
- Modify docs:
  - `Docs/API-related/Sandbox_API.md`
  - `tldw_Server_API/app/core/Sandbox/README.md`
  - `Docs/Sandbox/macos-runtime-operator-notes.md`
- Update Backlog task for implementation in the implementation branch.

## Task 1: Pure Projection Contract

**Files:**
- Create: `tldw_Server_API/app/core/Sandbox/operator_status.py`
- Create: `tldw_Server_API/tests/sandbox/test_operator_status.py`

- [ ] **Step 1: Write the failing ready-state projection test**

Add this test to `tldw_Server_API/tests/sandbox/test_operator_status.py`:

```python
from tldw_Server_API.app.core.Sandbox.operator_status import build_operator_status


def _runtime_diagnostics(*, ready: int = 1) -> dict[str, object]:
    return {
        "source": "feature_discovery",
        "summary": {
            "total": 2,
            "ready": ready,
            "unavailable": 1,
            "host_gated": 1,
            "scaffold": 0,
            "host_local_warning_runtimes": [],
            "repair_supported_runtimes": ["vz_linux"],
        },
        "runtimes": [
            {
                "name": "docker",
                "available": ready > 0,
                "implementation_state": "supported",
                "readiness": "ready" if ready > 0 else "unavailable",
                "reasons": [],
                "normalized_reasons": [],
                "normalized_reason_details": [],
                "boundary_class": "container",
                "vm_grade_isolation": False,
                "untrusted_eligible": True,
                "isolation_warnings": [],
                "strict_deny_all_supported": True,
                "strict_allowlist_supported": False,
                "session_reuse_model": "workspace_only",
                "requires_live_health_check": False,
                "repair_supported": False,
                "recommended_action": "none",
            },
            {
                "name": "vz_linux",
                "available": False,
                "implementation_state": "host_gated",
                "readiness": "host_gated",
                "reasons": ["vz_linux_unavailable"],
                "normalized_reasons": ["runtime_unavailable"],
                "normalized_reason_details": [],
                "boundary_class": "vm_grade",
                "vm_grade_isolation": True,
                "untrusted_eligible": True,
                "isolation_warnings": [],
                "strict_deny_all_supported": False,
                "strict_allowlist_supported": False,
                "session_reuse_model": "warm_vm",
                "requires_live_health_check": True,
                "repair_supported": True,
                "recommended_action": "check_runtime_readiness",
            },
        ],
    }


def _macos_diagnostics_unconfigured() -> dict[str, object]:
    return {
        "helper": {"configured": False, "ready": False, "reasons": []},
        "templates": {},
        "reconciliation": None,
        "image_store": {
            "configured": False,
            "registered_templates": 0,
            "run_manifests": 0,
            "gc_candidates": 0,
            "reasons": [],
        },
        "recovery_summary": None,
        "startup_warning_summary": {"present": False, "blocking": False, "codes": []},
    }


def test_operator_status_ready_when_runtime_ready_and_vz_unconfigured() -> None:
    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=_macos_diagnostics_unconfigured(),
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    assert payload["source"] == "sandbox_operator_status"
    assert payload["overall_status"] == "ready"
    assert payload["overall_severity"] == "info"
    assert payload["sections"]["evidence"]["status"] == "not_configured"
    assert "generated_at" not in payload
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_operator_status.py::test_operator_status_ready_when_runtime_ready_and_vz_unconfigured -q
```

Expected: `ModuleNotFoundError` for `operator_status` or missing `build_operator_status`.

- [ ] **Step 3: Implement minimal projection module**

Create `tldw_Server_API/app/core/Sandbox/operator_status.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

OperatorSection = dict[str, object]


def _as_dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _section(status: str, *, severity: str = "info", **extra: object) -> OperatorSection:
    return {"status": status, "severity": severity, **extra}


def build_operator_status(
    *,
    runtime_diagnostics: Mapping[str, Any] | None,
    macos_diagnostics: Mapping[str, Any] | None,
    startup_warning_summary: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    runtime_payload = _as_dict(runtime_diagnostics)
    macos_payload = _as_dict(macos_diagnostics)
    startup_payload = _as_dict(startup_warning_summary)

    runtime_summary = _as_dict(runtime_payload.get("summary"))
    ready_count = int(runtime_summary.get("ready") or 0)
    helper = _as_dict(macos_payload.get("helper"))
    image_store = _as_dict(macos_payload.get("image_store"))
    recovery = _as_dict(macos_payload.get("recovery_summary"))

    sections = {
        "runtime_readiness": _section(
            "ready" if ready_count else "unavailable",
            severity="info" if ready_count else "error",
            ready=ready_count,
            total=int(runtime_summary.get("total") or 0),
            host_local_warning_runtimes=list(runtime_summary.get("host_local_warning_runtimes") or []),
            repair_supported_runtimes=list(runtime_summary.get("repair_supported_runtimes") or []),
        ),
        "macos_vz": _section(
            "ready" if bool(helper.get("ready")) else "not_configured",
            configured=bool(helper.get("configured")),
            helper_ready=bool(helper.get("ready")),
            reasons=list(helper.get("reasons") or []),
        ),
        "image_store": _section(
            "ready" if bool(image_store.get("configured")) else "not_configured",
            configured=bool(image_store.get("configured")),
            gc_candidates=int(image_store.get("gc_candidates") or 0),
            reasons=list(image_store.get("reasons") or []),
        ),
        "reconciliation": _section(
            str(recovery.get("status") or "not_configured"),
            severity=str(recovery.get("severity") or "info").replace("ok", "info"),
            counts=dict(recovery.get("counts") or {}),
            repair_endpoint=recovery.get("repair_endpoint"),
            cleanup_plan_endpoint=recovery.get("cleanup_plan_endpoint"),
        ),
        "evidence": _section("not_configured"),
        "security_boundaries": _section(
            "ready",
            host_local_warning_runtimes=list(runtime_summary.get("host_local_warning_runtimes") or []),
        ),
        "startup_warnings": _section(
            "action_required" if bool(startup_payload.get("blocking")) else (
                "degraded" if bool(startup_payload.get("present")) else "ready"
            ),
            severity="error" if bool(startup_payload.get("blocking")) else (
                "warning" if bool(startup_payload.get("present")) else "info"
            ),
            present=bool(startup_payload.get("present")),
            blocking=bool(startup_payload.get("blocking")),
            codes=list(startup_payload.get("codes") or []),
        ),
    }

    overall_status = "ready" if ready_count else "unavailable"
    overall_severity = "info" if ready_count else "error"
    return {
        "source": "sandbox_operator_status",
        "overall_status": overall_status,
        "overall_severity": overall_severity,
        "summary": {
            "runtime_total": int(runtime_summary.get("total") or 0),
            "runtime_ready": ready_count,
            "actions": 0,
        },
        "sections": sections,
        "recommended_actions": [],
        "notes": [],
    }
```

This is intentionally minimal; later steps add action classification and failure handling.

- [ ] **Step 4: Run the test and verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_operator_status.py::test_operator_status_ready_when_runtime_ready_and_vz_unconfigured -q
```

Expected: `1 passed`.

## Task 2: Action Classification And Partial Section Failure

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/operator_status.py`
- Modify: `tldw_Server_API/tests/sandbox/test_operator_status.py`

- [ ] **Step 1: Add failing tests for repair action, image-store action, and partial failure**

Add tests:

```python
def test_operator_status_points_reconciliation_to_dry_run_repair() -> None:
    macos = _macos_diagnostics_unconfigured()
    macos["recovery_summary"] = {
        "status": "action_recommended",
        "severity": "warning",
        "codes": ["vz_stale_session_controls"],
        "counts": {"stale_session_controls": 1},
        "repair_endpoint": "/api/v1/sandbox/admin/macos-reconciliation/repair",
        "cleanup_plan_endpoint": None,
        "notes": [],
    }

    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=macos,
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    assert payload["overall_status"] == "action_required"
    assert payload["recommended_actions"][0]["code"] == "run_repair_dry_run"
    assert payload["recommended_actions"][0]["dry_run_required"] is True


def test_operator_status_keeps_runtime_section_when_macos_section_unavailable() -> None:
    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics={"_section_error": "macos_diagnostics_failed"},
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    assert payload["sections"]["runtime_readiness"]["status"] == "ready"
    assert payload["sections"]["macos_vz"]["status"] == "unknown"
    assert payload["overall_status"] == "degraded"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/sandbox/test_operator_status.py::test_operator_status_points_reconciliation_to_dry_run_repair \
  tldw_Server_API/tests/sandbox/test_operator_status.py::test_operator_status_keeps_runtime_section_when_macos_section_unavailable \
  -q
```

Expected: assertion failures for missing action and section failure behavior.

- [ ] **Step 3: Implement action records and failure sections**

In `operator_status.py`:

- Add helper `_action(code, severity, section, message, endpoint=None, dry_run_required=False)`.
- Add safe coercion helpers such as `_safe_int`, `_safe_list`, and `_safe_dict`
  so malformed diagnostic values become section-local `unknown`/`unavailable`
  state rather than crashing the endpoint.
- Treat `macos_diagnostics` containing `_section_error` as a section-local failure.
- Derive `run_repair_dry_run` when `recovery_summary.repair_endpoint` is present.
- Derive `inspect_image_store_cleanup_plan` when `recovery_summary.cleanup_plan_endpoint` is present.
- Overall precedence:
  - `unavailable` when no runtime is ready.
  - `action_required` when a blocking/startup/repair action exists.
  - `degraded` when section-local failures, non-blocking startup warnings, image-store cleanup candidates, or host-local warnings exist.
  - `ready` otherwise.

- [ ] **Step 4: Run the focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_operator_status.py -q
```

Expected: all tests in the new file pass.

## Task 3: Service Method Integration

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Modify: `tldw_Server_API/tests/sandbox/test_operator_status.py`

- [ ] **Step 1: Add failing service tests**

Add tests that monkeypatch service methods rather than real helper paths:

```python
from tldw_Server_API.app.core.Sandbox.service import SandboxService


def test_service_operator_status_uses_existing_diagnostics(monkeypatch) -> None:
    svc = SandboxService()
    monkeypatch.setattr(svc, "runtime_diagnostics_summary", lambda: _runtime_diagnostics())
    monkeypatch.setattr(svc, "macos_diagnostics", lambda: _macos_diagnostics_unconfigured())

    payload = svc.operator_status(startup_warning_summary={"present": False, "blocking": False, "codes": []})

    assert payload["source"] == "sandbox_operator_status"
    assert payload["overall_status"] == "ready"


def test_service_operator_status_isolates_macos_diagnostics_failure(monkeypatch) -> None:
    svc = SandboxService()
    monkeypatch.setattr(svc, "runtime_diagnostics_summary", lambda: _runtime_diagnostics())

    def fail_macos() -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(svc, "macos_diagnostics", fail_macos)

    payload = svc.operator_status(startup_warning_summary={"present": False, "blocking": False, "codes": []})

    assert payload["sections"]["runtime_readiness"]["status"] == "ready"
    assert payload["sections"]["macos_vz"]["status"] == "unknown"
```

- [ ] **Step 2: Run the service tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/sandbox/test_operator_status.py::test_service_operator_status_uses_existing_diagnostics \
  tldw_Server_API/tests/sandbox/test_operator_status.py::test_service_operator_status_isolates_macos_diagnostics_failure \
  -q
```

Expected: `AttributeError` for missing `operator_status`.

- [ ] **Step 3: Add service wrapper**

In `tldw_Server_API/app/core/Sandbox/service.py`:

```python
from .operator_status import build_operator_status
```

Add method near `runtime_diagnostics_summary()` and `macos_diagnostics()`:

```python
    def operator_status(
        self,
        *,
        startup_warning_summary: dict[str, object] | None = None,
    ) -> dict[str, object]:
        runtime_diagnostics: dict[str, object] | None
        macos_diagnostics: dict[str, object] | None
        try:
            runtime_diagnostics = self.runtime_diagnostics_summary()
        except (ConnectionError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
            logger.warning(
                "Sandbox operator status runtime diagnostics unavailable: {}",
                type(exc).__name__,
            )
            runtime_diagnostics = {"_section_error": f"runtime_diagnostics_failed: {type(exc).__name__}"}
        try:
            macos_diagnostics = self.macos_diagnostics()
        except (ConnectionError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
            logger.warning(
                "Sandbox operator status macOS diagnostics unavailable: {}",
                type(exc).__name__,
            )
            macos_diagnostics = {"_section_error": f"macos_diagnostics_failed: {type(exc).__name__}"}
        return build_operator_status(
            runtime_diagnostics=runtime_diagnostics,
            macos_diagnostics=macos_diagnostics,
            startup_warning_summary=startup_warning_summary,
        )
```

Do not catch `Exception` or `BaseException` here. This status method should
isolate expected operational diagnostics failures, but programming errors should
remain visible during development. If implementation discovers a concrete
additional operational exception, add it deliberately and cover it with a test.

- [ ] **Step 4: Run service tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_operator_status.py -q
```

Expected: all operator status tests pass.

## Task 4: Pydantic Schema

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
- Modify: `tldw_Server_API/tests/sandbox/test_operator_status.py`

- [ ] **Step 1: Add failing schema validation test**

Add:

```python
from tldw_Server_API.app.api.v1.schemas.sandbox_schemas import (
    SandboxAdminOperatorStatusResponse,
)


def test_operator_status_payload_validates_against_schema() -> None:
    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=_macos_diagnostics_unconfigured(),
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    model = SandboxAdminOperatorStatusResponse.model_validate(payload)

    assert model.source == "sandbox_operator_status"
    assert model.overall_status == "ready"
```

- [ ] **Step 2: Run the schema test and verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_operator_status.py::test_operator_status_payload_validates_against_schema -q
```

Expected: import error for missing schema.

- [ ] **Step 3: Add schema models**

In `sandbox_schemas.py`, near the existing sandbox admin diagnostics models, add:

```python
from pydantic import ConfigDict

OperatorStatusValue = Literal["ready", "degraded", "action_required", "unavailable", "unknown", "not_configured"]
OperatorSeverity = Literal["info", "warning", "error"]


class SandboxAdminOperatorStatusAction(BaseModel):
    code: str
    severity: OperatorSeverity
    section: str
    message: str
    endpoint: str | None = None
    dry_run_required: bool = False


class SandboxAdminOperatorStatusSection(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: OperatorStatusValue
    severity: OperatorSeverity = "info"
    configured: bool | None = None
    reasons: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    repair_endpoint: str | None = None
    cleanup_plan_endpoint: str | None = None
```

Then:

```python
class SandboxAdminOperatorStatusResponse(BaseModel):
    source: Literal["sandbox_operator_status"]
    overall_status: Literal["ready", "degraded", "action_required", "unavailable", "unknown"]
    overall_severity: OperatorSeverity
    summary: dict[str, int | bool | str | list[str]]
    sections: dict[str, SandboxAdminOperatorStatusSection]
    recommended_actions: list[SandboxAdminOperatorStatusAction] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
```

Do not add `generated_at`.

- [ ] **Step 4: Run schema test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_operator_status.py::test_operator_status_payload_validates_against_schema -q
```

Expected: pass.

## Task 5: Admin Endpoint And RBAC

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/sandbox.py`
- Modify: `tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py`
- Modify: `tldw_Server_API/tests/sandbox/test_admin_rbac.py`

- [ ] **Step 1: Add failing endpoint test**

In `test_admin_macos_diagnostics.py`, add a test mirroring `admin_runtime_diagnostics` setup:

```python
def test_admin_operator_status_returns_structured_payload(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sandbox_endpoint

    fake_service = SimpleNamespace(
        operator_status=lambda *, startup_warning_summary=None: {
            "source": "sandbox_operator_status",
            "overall_status": "ready",
            "overall_severity": "info",
            "summary": {"runtime_total": 1, "runtime_ready": 1, "actions": 0},
            "sections": {"runtime_readiness": {"status": "ready", "severity": "info"}},
            "recommended_actions": [],
            "notes": [],
        }
    )
    monkeypatch.setattr(sandbox_endpoint, "_service", fake_service)

    client = TestClient(app)
    resp = client.get("/api/v1/sandbox/admin/operator-status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "sandbox_operator_status"
    assert data["overall_status"] == "ready"
```

Adjust imports to match the file's existing fixtures and auth setup.

- [ ] **Step 2: Run endpoint test and verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py::test_admin_operator_status_returns_structured_payload -q
```

Expected: 404 or missing schema import.

- [ ] **Step 3: Add endpoint**

In `sandbox.py`:

- Import `SandboxAdminOperatorStatusResponse`.
- Add route after `admin_runtime_diagnostics`:

```python
@router.get(
    "/admin/operator-status",
    response_model=SandboxAdminOperatorStatusResponse,
    summary="Admin: sandbox operator status",
)
async def admin_operator_status(
    request: Request,
    _principal: AuthPrincipal = Depends(RequireRole("admin")),
    _current_user: User = Depends(get_request_user),
) -> SandboxAdminOperatorStatusResponse:
    """Return read-only sandbox operator status consolidated from diagnostics."""
    startup_warning_summary = _sandbox_startup_warning_summary(request)
    payload = dict(
        await asyncio.to_thread(
            _service.operator_status,
            startup_warning_summary=startup_warning_summary,
        )
    )
    return SandboxAdminOperatorStatusResponse.model_validate(payload)
```

- [ ] **Step 4: Add RBAC coverage**

Add `/api/v1/sandbox/admin/operator-status` to the admin-only endpoint list in
`tldw_Server_API/tests/sandbox/test_admin_rbac.py`.

- [ ] **Step 5: Run endpoint/RBAC tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py::test_admin_operator_status_returns_structured_payload \
  tldw_Server_API/tests/sandbox/test_admin_rbac.py \
  -q
```

Expected: pass.

## Task 6: Docs And API Contract

**Files:**
- Modify: `Docs/API-related/Sandbox_API.md`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py` only if adding a doc-contract assertion is necessary.

- [ ] **Step 1: Add docs updates**

Document:

- `GET /api/v1/sandbox/admin/operator-status`.
- It is read-only and admin-only.
- It projects existing runtime diagnostics and macOS diagnostics.
- It does not run helper lifecycle commands, launchd, repair, cleanup, evidence ingestion, or real VMs.
- `evidence.status=not_configured` is non-blocking by default.
- Detailed troubleshooting remains in `runtime-diagnostics` and `macos-diagnostics`.

- [ ] **Step 2: Run docs-relevant tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py \
  tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py \
  -q
```

Expected: pass.

## Task 7: Final Verification And Commit

**Files:**
- Modify: `backlog/tasks/<implementation-task>.md`

- [ ] **Step 1: Run focused sandbox verification**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/sandbox/test_operator_status.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_admin_rbac.py \
  tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py \
  tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py \
  -q
```

Expected: pass. If unrelated tests in the selected files fail, stop and root-cause before narrowing.

- [ ] **Step 2: Run Bandit on touched Python files**

Run:

```bash
source .venv/bin/activate
python -m bandit \
  tldw_Server_API/app/core/Sandbox/operator_status.py \
  tldw_Server_API/app/core/Sandbox/service.py \
  tldw_Server_API/app/api/v1/endpoints/sandbox.py \
  tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py \
  -f json -o /tmp/bandit_sandbox_operator_status_impl.json
```

Expected: no new findings in touched code. If Bandit reports existing findings in broader files, confirm they are not introduced by this change and record them.

- [ ] **Step 3: Check diff hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only planned files modified.

- [ ] **Step 4: Update Backlog task**

Record:

- implementation summary
- verification commands and results
- Bandit result file path
- any skipped host-gated checks and why

- [ ] **Step 5: Commit**

Run:

```bash
git add \
  tldw_Server_API/app/core/Sandbox/operator_status.py \
  tldw_Server_API/app/core/Sandbox/service.py \
  tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/sandbox.py \
  tldw_Server_API/tests/sandbox/test_operator_status.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_admin_rbac.py \
  Docs/API-related/Sandbox_API.md \
  tldw_Server_API/app/core/Sandbox/README.md \
  Docs/Sandbox/macos-runtime-operator-notes.md \
  backlog/tasks/<implementation-task>.md
git commit -m "feat: add sandbox operator status endpoint"
```

Expected: commit succeeds without bypassing hooks.

## Review Checklist

- [ ] Endpoint is admin-only and read-only.
- [ ] No helper lifecycle, launchd, repair mutation, cleanup mutation, image-store root creation, host-gated smoke, or real VM execution is added.
- [ ] `generated_at` is absent from Slice 1.
- [ ] Unconfigured VZ/evidence does not degrade otherwise usable installs.
- [ ] Section failures are isolated and visible.
- [ ] Recommended actions use stable codes and include dry-run context where needed.
- [ ] Existing detailed diagnostics remain the source of truth.
