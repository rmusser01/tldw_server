# Sandbox Network Policy Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured sandbox runtime network policy metadata to runtime discovery.

**Architecture:** Reuse the existing runtime-capability metadata pattern. Add a static metadata map in `runtime_capabilities.py`, project it through `SandboxService.feature_discovery()`, type it in `SandboxRuntimeInfo`, and update docs/tests to keep the contract synchronized.

**Tech Stack:** Python 3.11, FastAPI/Pydantic, pytest, Backlog.md.

---

## Files

- Modify: `tldw_Server_API/app/core/Sandbox/runtime_capabilities.py`
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
- Modify: `tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py`
- Modify: `tldw_Server_API/tests/Docs/test_sandbox_public_docs_contract.py`
- Modify: `Docs/API-related/Sandbox_API.md`
- Modify: `Docs/Published/API-related/Sandbox_API.md`
- Modify: `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- Modify: `Docs/Sandbox/sandbox-security-policy-matrix.md`
- Modify: `backlog/tasks/task-44 - Expose-sandbox-runtime-network-policy-metadata.md`

## Task 1: Contract Tests

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py`
- Modify: `tldw_Server_API/tests/Docs/test_sandbox_public_docs_contract.py`

- [ ] **Step 1: Add failing runtime metadata tests**

Add tests that import `RUNTIME_NETWORK_POLICY_METADATA` and
`runtime_network_policy_metadata`, then assert:

- `set(RUNTIME_NETWORK_POLICY_METADATA) == set(RuntimeType)`
- `runtime_network_policy_metadata("future_runtime")` raises `ValueError`
- discovery includes `network_policy_contract` for every runtime
- `seatbelt` and `worktree` report both policies as unsupported
- `vz_linux` reports host-gated strict `deny_all` and unsupported `allowlist`

- [ ] **Step 2: Add failing schema contract test**

Assert `SandboxRuntimeInfo.model_json_schema()` requires
`network_policy_contract` and the field is not nullable.

- [ ] **Step 3: Run focused tests and confirm RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py \
  tldw_Server_API/tests/Docs/test_sandbox_public_docs_contract.py \
  -q --timeout=60
```

Expected: fail because the metadata map and schema field do not exist.

## Task 2: Runtime Metadata And Discovery

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/runtime_capabilities.py`
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`

- [ ] **Step 1: Add metadata types and map**

Add `RuntimeNetworkPolicySupportState`, `RuntimeNetworkPolicyReadinessSource`,
`RuntimeNetworkPolicyModeMetadata`, and `RuntimeNetworkPolicyMetadata`.

- [ ] **Step 2: Add complete metadata map**

Add `RUNTIME_NETWORK_POLICY_METADATA` for every `RuntimeType` using the design
table from `2026-05-04-sandbox-network-policy-metadata-design.md`.

- [ ] **Step 3: Add import-time completeness validation**

Mirror the isolation metadata validation pattern and raise `RuntimeError` with
missing/extra runtime keys.

- [ ] **Step 4: Add safe accessor**

Add `runtime_network_policy_metadata(runtime)` that coerces string inputs and
raises `ValueError` for unknown or missing runtime metadata.

- [ ] **Step 5: Wire discovery**

In `SandboxService.feature_discovery()`, add `network_policy_contract` to
`_preflight_fields()` from the static accessor. Do not remove existing
readiness booleans.

- [ ] **Step 6: Type schema**

Add Pydantic models for policy mode metadata and the contract field. Mark
`network_policy_contract` required on `SandboxRuntimeInfo`.

- [ ] **Step 7: Run focused tests and confirm GREEN**

Run the same focused pytest command. Expected: pass.

## Task 3: Documentation And Verification

**Files:**
- Modify: `Docs/API-related/Sandbox_API.md`
- Modify: `Docs/Published/API-related/Sandbox_API.md`
- Modify: `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- Modify: `Docs/Sandbox/sandbox-security-policy-matrix.md`
- Modify: `backlog/tasks/task-44 - Expose-sandbox-runtime-network-policy-metadata.md`

- [ ] **Step 1: Update public API docs**

Document `network_policy_contract` as static security posture metadata and
state that `enforcement_ready` remains current host readiness.

- [ ] **Step 2: Update inventory and security matrix**

Add maintenance text requiring updates when runtime/network policy contracts
change. Ensure host-local negative claims are explicit.

- [ ] **Step 3: Run focused tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py \
  tldw_Server_API/tests/Docs/test_sandbox_public_docs_contract.py \
  -q --timeout=60
```

- [ ] **Step 4: Run Bandit on touched Python files**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit \
  -r tldw_Server_API/app/core/Sandbox/runtime_capabilities.py \
     tldw_Server_API/app/core/Sandbox/service.py \
     tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py \
  -f json -o /tmp/bandit_sandbox_network_policy_metadata.json
```

- [ ] **Step 5: Run whitespace check**

Run:

```bash
git diff --check
```

- [ ] **Step 6: Finalize Backlog task**

Check acceptance criteria, record verification, and add final summary.

- [ ] **Step 7: Commit**

Commit all changes with:

```bash
git add Docs tldw_Server_API backlog
git commit -m "feat(sandbox): expose network policy metadata"
```
