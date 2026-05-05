# Sandbox Recovery Diagnostics Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a read-only operator recovery summary to macOS sandbox diagnostics.

**Architecture:** Implement the summary as a pure projection over existing diagnostics blocks in `macos_diagnostics.py`, then expose it through the existing admin diagnostics schema and endpoint. Do not add helper/image-store calls while building the summary, and do not add new repair behavior.

**Tech Stack:** Python, FastAPI, Pydantic, pytest, existing sandbox diagnostics helpers.

---

### Task 1: Add Schema And Diagnostics Contract Tests

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_macos_diagnostics.py`
- Modify: `tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py`
- Modify later: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
- Modify later: `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`

- [x] **Step 1: Add failing schema coverage**

Add `recovery_summary` to the sample diagnostics payload and assert the schema
accepts:

- `status`
- `severity`
- `codes`
- `counts`
- `recommended_action`
- optional `repair_endpoint`
- optional `cleanup_plan_endpoint`
- `notes`

- [x] **Step 2: Add failing pure diagnostics coverage**

Add tests for:

- healthy computed reconciliation
- helper-unavailable or uncomputed reconciliation
- stale/unhealthy/orphaned reconciliation state
- image-store cleanup candidates

Expected: FAIL because `collect_macos_diagnostics()` does not emit
`recovery_summary` and schema models do not define it.

- [x] **Step 3: Add failing endpoint shape coverage**

Update the admin diagnostics endpoint test to expect `recovery_summary` in the
response keys.

Expected: FAIL because the current response model omits the field.

### Task 2: Implement Pure Recovery Summary Projection

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`

- [x] **Step 1: Add constants for existing admin endpoints**

Use:

- `/api/v1/sandbox/admin/macos-reconciliation/repair`
- `/api/v1/sandbox/admin/macos-image-store/cleanup-plan`

- [x] **Step 2: Add a pure `summarize_recovery()` helper**

Inputs:

- `reconciliation: dict[str, object] | None`
- `image_store: dict[str, object] | None`
- `observability: dict[str, object] | None`

Output:

- dict matching the new schema

Rules:

- return `unavailable/error` when reconciliation is missing, uncomputed, or has
  reasons
- return `action_recommended/warning` when issue counts or GC candidates exist
- return `healthy/ok` when computed and no issue counts exist
- include repair endpoint only for stale, unhealthy, or owned orphan VM cases
- include cleanup-plan endpoint only when image-store GC candidates exist
- include unknown/foreign orphan codes as inspect-only warnings

- [x] **Step 3: Wire summary into `collect_macos_diagnostics()`**

Call the helper after existing diagnostics blocks are collected and include
`recovery_summary` in the returned payload.

### Task 3: Add Pydantic Response Model

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`

- [x] **Step 1: Add `SandboxAdminMacOSRecoverySummary`**

Fields:

- `status: Literal["healthy", "action_recommended", "unavailable"]`
- `severity: Literal["ok", "warning", "error"]`
- `codes: list[str]`
- `counts: dict[str, int]`
- `recommended_action: str | None`
- `repair_endpoint: str | None`
- `cleanup_plan_endpoint: str | None`
- `notes: list[str]`

- [x] **Step 2: Add the optional field to diagnostics response**

Add `recovery_summary: SandboxAdminMacOSRecoverySummary | None = None` to
`SandboxAdminMacOSDiagnosticsResponse`.

### Task 4: Documentation And Verification

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`
- Modify: `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- Modify: `backlog/tasks/task-70 - Add-macOS-sandbox-recovery-diagnostics-summary.md`

- [x] **Step 1: Update operator docs**

Document that macOS diagnostics include a read-only `recovery_summary` derived
from reconciliation/image-store/observability state.

- [x] **Step 2: Run focused tests**

Run:

```bash
python -m pytest tldw_Server_API/tests/sandbox/test_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py -q
```

Expected: PASS.

- [x] **Step 3: Run py_compile**

Run:

```bash
python -m py_compile tldw_Server_API/app/core/Sandbox/macos_diagnostics.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py
```

Expected: PASS.

- [x] **Step 4: Run Bandit on touched production code**

Run:

```bash
python -m bandit -r tldw_Server_API/app/core/Sandbox/macos_diagnostics.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py -f json -o /tmp/bandit_sandbox_recovery_summary.json
```

Expected: zero new findings.

- [x] **Step 5: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.
