# Sandbox Status Reason Details Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add additive structured metadata for existing sandbox `status_reason_code` values on public and admin run status responses.

**Architecture:** Keep the taxonomy module as the single source of truth. Add a static completeness-checked metadata map keyed by existing `RunStatusReasonCode` literals, expose it through Pydantic schemas, and serialize it in endpoints from the same local helper that computes `status_reason_code`.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, pytest, Ruff, Bandit.

---

## Files

- Modify: `tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sandbox.py`
- Modify: `tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py`
- Modify: `tldw_Server_API/tests/sandbox/test_run_status_contract_durability.py`
- Modify: `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- Modify: `Docs/API-related/Sandbox_API.md`
- Modify: `backlog/tasks/task-122 - Add-structured-sandbox-run-status-reason-metadata.md`

## Task 1: Taxonomy Metadata Contract

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py`
- Test: `tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py`

- [x] **Step 1: Write failing completeness and sample metadata tests**

Add tests that import `RUN_STATUS_REASON_METADATA`, `RunStatusReasonCode`, and `run_status_reason_details()` and assert:

```python
def test_run_status_reason_metadata_covers_every_reason_code() -> None:
    assert set(RUN_STATUS_REASON_METADATA) == set(get_args(RunStatusReasonCode))


def test_run_status_reason_details_exposes_stable_metadata() -> None:
    assert run_status_reason_details("runtime_unavailable").category == "runtime"
    assert run_status_reason_details("runtime_unavailable").severity == "error"
    assert run_status_reason_details("runtime_unavailable").retryable is True
    assert run_status_reason_details("policy_failed").operator_action == "review_policy"
    assert run_status_reason_details("limits_applied").severity == "warning"
```

- [x] **Step 2: Run RED test**

Run:

```bash
python -m pytest tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py::test_run_status_reason_metadata_covers_every_reason_code tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py::test_run_status_reason_details_exposes_stable_metadata -q
```

Expected: fails because metadata symbols do not exist.

- [x] **Step 3: Implement minimal taxonomy metadata**

Add literal types, frozen dataclass, static metadata map, import-time completeness validation, and `run_status_reason_details()`.

Keep unknown external input safe:

```python
def run_status_reason_details(code: RunStatusReasonCode | str | None) -> RunStatusReasonDetails:
    code_value = str(code or "unknown").strip()
    metadata = RUN_STATUS_REASON_METADATA.get(code_value)
    if metadata is None:
        return RUN_STATUS_REASON_METADATA["unknown"]
    return metadata
```

- [x] **Step 4: Run GREEN test**

Run the same pytest command. Expected: both tests pass.

## Task 2: Schema Exposure

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
- Test: `tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py`

- [x] **Step 1: Write failing schema tests**

Extend the existing schema test:

```python
assert "status_reason_details" in public_schema["properties"]
assert "status_reason_details" in admin_schema["properties"]
```

Also instantiate `SandboxRunStatus` and `SandboxAdminRunSummary` with a details object and assert `.model_dump()` preserves `code`, `category`, `severity`, `terminal`, `retryable`, `operator_action`, and `user_message_key`.

- [x] **Step 2: Run RED test**

Run:

```bash
python -m pytest tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py::test_public_and_admin_status_schemas_expose_reason_code -q
```

Expected: fails because the schema field does not exist.

- [x] **Step 3: Add Pydantic details model and fields**

Import taxonomy literal types and define `SandboxRunStatusReasonDetails`.
Add nullable `status_reason_details` fields beside `status_reason_code` on `SandboxRunStatus` and `SandboxAdminRunSummary`.

- [x] **Step 4: Run GREEN test**

Run the same pytest command. Expected: passes.

## Task 3: Endpoint Serialization

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/sandbox.py`
- Test: `tldw_Server_API/tests/sandbox/test_run_status_contract_durability.py`
- Test: `tldw_Server_API/tests/sandbox/test_admin_details_resource_usage.py`
- Test: `tldw_Server_API/tests/sandbox/test_admin_list_filters_pagination.py`

- [x] **Step 1: Write failing response tests**

Add or extend focused tests so public run status and admin run responses include:

```python
assert data["status_reason_code"] == "queued"
assert data["status_reason_details"]["code"] == "queued"
assert data["status_reason_details"]["category"] == "lifecycle"
assert data["status_reason_details"]["terminal"] is False
```

For an admin completed run with truncation/resource usage, assert `limits_applied` details are returned.

- [x] **Step 2: Run RED tests**

Run:

```bash
python -m pytest tldw_Server_API/tests/sandbox/test_run_status_contract_durability.py tldw_Server_API/tests/sandbox/test_admin_details_resource_usage.py tldw_Server_API/tests/sandbox/test_admin_list_filters_pagination.py -q
```

Expected: at least one details assertion fails.

- [x] **Step 3: Add endpoint helper**

Replace `_status_reason_code()` with a helper that computes both:

```python
@dataclass(frozen=True)
class _StatusReasonProjection:
    code: str
    details: SandboxRunStatusReasonDetails
```

Use `normalize_run_status_reason()` once, then `run_status_reason_details(code)`.
Populate all four constructors:

- `start_run()` response
- `get_run_status()`
- `admin_list_runs()`
- `admin_get_run_details()`

Implementation note: the final code uses a small `_status_reason_details()`
helper plus a local `status_reason_code` variable instead of adding an internal
projection dataclass. That keeps endpoint serialization explicit without adding
a one-use abstraction.

- [x] **Step 4: Run GREEN tests**

Run the same pytest command. Expected: passes.

## Task 4: Documentation And Inventory

**Files:**
- Modify: `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- Modify: `Docs/API-related/Sandbox_API.md`
- Modify: `backlog/tasks/task-122 - Add-structured-sandbox-run-status-reason-metadata.md`
- Test: `tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py`

- [x] **Step 1: Add failing docs guard if needed**

If the portable capability gate does not already assert the structured metadata contract, add an assertion that the inventory mentions `status_reason_details` and preserves the Phase 3 limitation that runtime discovery reason metadata is still future work.

- [x] **Step 2: Run RED docs guard**

Run:

```bash
python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py::test_portable_runtime_capability_gate_inventory_no_longer_lists_gate_as_missing -q
```

Expected: fails if a new guard was added before docs are updated.

- [x] **Step 3: Update docs**

Document `status_reason_details` as additive metadata for run status responses. Update the inventory's Phase 3 gap to say run status metadata is covered, while runtime discovery `normalized_reasons` still lack equivalent rich details.

- [x] **Step 4: Run GREEN docs guard**

Run the focused capability gate test. Expected: passes.

## Task 5: Verification And Commit

**Files:**
- All touched files

- [x] **Step 1: Run focused pytest**

```bash
python -m pytest tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py tldw_Server_API/tests/sandbox/test_run_status_contract_durability.py tldw_Server_API/tests/sandbox/test_admin_details_resource_usage.py tldw_Server_API/tests/sandbox/test_admin_list_filters_pagination.py tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q
```

Expected: focused sandbox tests pass.

- [x] **Step 2: Run compile check**

```bash
python -m py_compile tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py tldw_Server_API/app/api/v1/endpoints/sandbox.py tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py tldw_Server_API/tests/sandbox/test_run_status_contract_durability.py
```

Expected: exits 0.

- [x] **Step 3: Run Ruff on touched Python files**

```bash
python -m ruff check tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py tldw_Server_API/app/api/v1/endpoints/sandbox.py tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py tldw_Server_API/tests/sandbox/test_run_status_contract_durability.py tldw_Server_API/tests/sandbox/test_admin_details_resource_usage.py tldw_Server_API/tests/sandbox/test_admin_list_filters_pagination.py tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py
```

Expected: no new Ruff failures. If pre-existing file-level warnings appear in broad files, rerun with `--select F,E9` for production files and record the limitation.

- [x] **Step 4: Run Bandit on touched production Python**

```bash
python -m bandit -r tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py tldw_Server_API/app/api/v1/endpoints/sandbox.py -f json -o /tmp/bandit_sandbox_status_reason_details.json
```

Expected: zero new findings on changed production lines.

- [x] **Step 5: Run whitespace check**

```bash
git diff --check
```

Expected: exits 0.

- [ ] **Step 6: Update TASK-122**

Check acceptance criteria and Definition of Done, add verification notes, and final summary.

- [ ] **Step 7: Commit**

```bash
git add Docs/superpowers/specs/2026-05-08-sandbox-status-reason-details-design.md Docs/superpowers/plans/2026-05-08-sandbox-status-reason-details-plan.md Docs/Sandbox/sandbox-runtime-capability-inventory.md Docs/API-related/Sandbox_API.md "backlog/tasks/task-122 - Add-structured-sandbox-run-status-reason-metadata.md" tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py tldw_Server_API/app/api/v1/endpoints/sandbox.py tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py tldw_Server_API/tests/sandbox/test_run_status_contract_durability.py tldw_Server_API/tests/sandbox/test_admin_details_resource_usage.py tldw_Server_API/tests/sandbox/test_admin_list_filters_pagination.py tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py
git commit -m "feat(sandbox): expose status reason details"
```
