# Sandbox Session Semantics Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a structured `session_contract` to sandbox runtime discovery so clients can distinguish workspace-only sessions, warm VM reuse, scaffolded session shapes, and recovery/repair posture.

**Architecture:** Follow the existing static metadata pattern in `runtime_capabilities.py`, expose the metadata through `SandboxService.feature_discovery()`, type it in `sandbox_schemas.py`, and document the inventory contract. No runtime execution behavior changes.

**Tech Stack:** Python 3.11, FastAPI/Pydantic schemas, pytest, existing sandbox runtime capability helpers.

---

### Task 1: Add Failing Discovery Contract Tests

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py`

- [x] **Step 1: Import session metadata helper names**

Add imports for the new helper after implementation names are selected:
`RUNTIME_SESSION_CONTRACT_METADATA` and `runtime_session_contract_metadata`.

- [x] **Step 2: Add discovery assertion test**

Assert every runtime row includes a `session_contract` object with fields:
`support_state`, `reuse_model`, `requires_live_health_check`,
`recovery_state`, and `repair_state`.

- [x] **Step 3: Add classification assertions**

Assert:

```python
discovery["docker"]["session_contract"]["reuse_model"] == "workspace_only"
discovery["vz_linux"]["session_contract"]["reuse_model"] == "warm_vm"
discovery["vz_linux"]["session_contract"]["requires_live_health_check"] is True
discovery["seatbelt"]["session_contract"]["support_state"] == "scaffold"
discovery["worktree"]["session_contract"]["support_state"] == "scaffold"
```

- [x] **Step 4: Verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py -q
```

Expected: failure because `session_contract` does not exist yet.

### Task 2: Implement Session Metadata

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/runtime_capabilities.py`
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`

- [x] **Step 1: Add typed metadata classes**

Add `RuntimeSessionReuseModel` and `RuntimeSessionContractMetadata` in
`runtime_capabilities.py`.

- [x] **Step 2: Add `RUNTIME_SESSION_CONTRACT_METADATA`**

Create a complete map for every `RuntimeType`.

- [x] **Step 3: Add completeness validation and accessor**

Validate the map at import time and add `runtime_session_contract_metadata()`
that rejects unknown runtimes with `ValueError`.

- [x] **Step 4: Add schema object**

Create `SandboxRuntimeSessionContract` in `sandbox_schemas.py` and add
`session_contract` to `SandboxRuntimeInfo`.

- [x] **Step 5: Wire discovery**

In `SandboxService.feature_discovery()` add `session_contract` to `_preflight_fields()`.

- [x] **Step 6: Verify GREEN**

Run the focused runtime inventory test file again.

### Task 3: Update Documentation And Tracking

**Files:**
- Modify: `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- Modify: `backlog/tasks/task-59 - Add-sandbox-runtime-session-semantics-discovery-contract.md`

- [x] **Step 1: Document the new discovery field**

Add `session_contract` to the discovery contract list.

- [x] **Step 2: Add session contract table**

Document support state, reuse model, live health check requirement, recovery
state, and repair state for each runtime.

- [x] **Step 3: Narrow the current gap**

Replace the broad session-semantics gap with remaining cross-runtime behavior
contract tests/recovery work.

### Task 4: Verification And Commit

**Files:**
- Verify touched code/docs.

- [x] **Step 1: Run focused tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py tldw_Server_API/tests/Docs/test_sandbox_public_docs_contract.py -q
```

- [x] **Step 2: Run Bandit on touched Python**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Sandbox/runtime_capabilities.py tldw_Server_API/app/core/Sandbox/service.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py -f json -o /tmp/bandit_sandbox_session_contract.json
```

- [x] **Step 3: Run whitespace check**

```bash
git diff --check
```

- [x] **Step 4: Record broader API TestClient limitation**

The broader `test_feature_discovery_flags.py` and
`test_sandbox_api.py::test_runtimes_discovery_shape` TestClient run timed out
in unrelated full-app lifespan teardown while background workers/schedulers were
active. This slice added direct `SandboxRuntimesResponse` validation around
`SandboxService.feature_discovery()` to cover the endpoint response model
without starting the full app lifespan.

- [x] **Step 5: Commit**

```bash
git add Docs/API-related/Sandbox_API.md Docs/Published/API-related/Sandbox_API.md Docs/Sandbox/sandbox-runtime-capability-inventory.md Docs/superpowers/specs/2026-05-05-sandbox-session-semantics-contract-design.md Docs/superpowers/plans/2026-05-05-sandbox-session-semantics-contract-implementation-plan.md "backlog/tasks/task-59 - Add-sandbox-runtime-session-semantics-discovery-contract.md" tldw_Server_API/app/core/Sandbox/runtime_capabilities.py tldw_Server_API/app/core/Sandbox/service.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py tldw_Server_API/tests/Docs/test_sandbox_public_docs_contract.py tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py
git commit -m "feat(sandbox): expose runtime session contract metadata"
```
