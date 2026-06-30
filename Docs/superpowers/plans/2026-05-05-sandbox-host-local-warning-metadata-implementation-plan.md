# Sandbox Host-Local Warning Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add advisory host-local isolation warning metadata to sandbox runtime discovery.

**Architecture:** Derive warning codes from existing static `RuntimeIsolationMetadata`, expose them through `SandboxService.feature_discovery()`, and type the additive response field in `SandboxRuntimeInfo`. This keeps warnings as discovery metadata and avoids new admission or execution behavior.

**Tech Stack:** Python dataclasses and `Literal` types, FastAPI/Pydantic schemas, pytest, Bandit.

---

### Task 1: Add Failing Tests

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py`
- Modify: `tldw_Server_API/tests/sandbox/test_sandbox_api.py`

- [ ] **Step 1: Add service discovery assertions**

Add a focused test that builds `SandboxService().feature_discovery()` and
asserts:

```python
assert discovery["seatbelt"]["isolation_warnings"] == [
    "host_local_boundary",
    "not_vm_grade_isolation",
    "not_untrusted_eligible",
]
assert discovery["worktree"]["isolation_warnings"] == [
    "host_local_boundary",
    "not_vm_grade_isolation",
    "not_untrusted_eligible",
]
assert "host_local_boundary" not in discovery["vz_linux"]["isolation_warnings"]
```

- [ ] **Step 2: Add API shape assertion**

Extend the runtime discovery shape test to assert `isolation_warnings` is
present and is a list.

- [ ] **Step 3: Run tests and confirm RED**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py \
  tldw_Server_API/tests/sandbox/test_sandbox_api.py::test_runtimes_discovery_shape \
  -q --timeout=60
```

Expected: fail because `isolation_warnings` is missing.

### Task 2: Implement Runtime Warning Metadata

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/runtime_capabilities.py`
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`

- [ ] **Step 1: Add warning-code type and helper**

Add `RuntimeIsolationWarningCode` with the initial host-local warning codes,
then add `runtime_isolation_warnings(runtime)` that derives warnings from
`runtime_isolation_metadata(runtime)`.

- [ ] **Step 2: Wire discovery**

Import the helper in `service.py` and include
`"isolation_warnings": runtime_isolation_warnings(runtime)` in
`_preflight_fields()`.

- [ ] **Step 3: Type the response schema**

Import `RuntimeIsolationWarningCode` in `sandbox_schemas.py` and add:

```python
isolation_warnings: list[RuntimeIsolationWarningCode] = Field(default_factory=list, ...)
```

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run the same focused pytest command from Task 1.

### Task 3: Update Docs And Verify

**Files:**
- Modify: `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- Modify: `Docs/Sandbox/sandbox-security-policy-matrix.md`
- Modify: `backlog/tasks/task-51 - Expose-host-local-sandbox-runtime-warning-metadata.md`

- [ ] **Step 1: Document advisory semantics**

Update docs to state `isolation_warnings` is additive advisory metadata, not an
admission decision.

- [ ] **Step 2: Run full focused verification**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py \
  tldw_Server_API/tests/sandbox/test_feature_discovery_flags.py \
  tldw_Server_API/tests/sandbox/test_sandbox_api.py::test_runtimes_discovery_shape \
  -q --timeout=60
python -m py_compile \
  tldw_Server_API/app/core/Sandbox/runtime_capabilities.py \
  tldw_Server_API/app/core/Sandbox/service.py \
  tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py
python -m bandit -r \
  tldw_Server_API/app/core/Sandbox/runtime_capabilities.py \
  tldw_Server_API/app/core/Sandbox/service.py \
  tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py \
  -f json -o /tmp/bandit_sandbox_host_local_warning_metadata.json
git diff --check
```

- [ ] **Step 3: Finalize task and commit**

Update `TASK-51`, stage the touched files, and commit:

```bash
git add Docs/Sandbox/sandbox-runtime-capability-inventory.md \
  Docs/Sandbox/sandbox-security-policy-matrix.md \
  Docs/superpowers/specs/2026-05-05-sandbox-host-local-warning-metadata-design.md \
  Docs/superpowers/plans/2026-05-05-sandbox-host-local-warning-metadata-implementation-plan.md \
  "backlog/tasks/task-51 - Expose-host-local-sandbox-runtime-warning-metadata.md" \
  tldw_Server_API/app/core/Sandbox/runtime_capabilities.py \
  tldw_Server_API/app/core/Sandbox/service.py \
  tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py \
  tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py \
  tldw_Server_API/tests/sandbox/test_sandbox_api.py
git commit -m "feat(sandbox): expose host-local isolation warnings"
```
