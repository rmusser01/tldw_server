# Sandbox Runtime Isolation Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured isolation posture metadata to sandbox runtime discovery so API clients do not infer security guarantees from prose.

**Architecture:** Define a small runtime-to-isolation metadata map in `runtime_capabilities.py`, expose it through `SandboxService.feature_discovery()`, and document the additive schema fields in API/runtime contract docs. The implementation must not change runtime admission or execution behavior.

**Tech Stack:** FastAPI/Pydantic schemas, Python sandbox service code, pytest contract tests, Markdown docs.

---

### Task 1: Add Failing Runtime Discovery Contract Tests

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py`

- [x] **Step 1: Add a focused test for isolation metadata**

Add a test that builds `SandboxService().feature_discovery()` with `SANDBOX_STORE_BACKEND=memory` and asserts each runtime has `boundary_class`, `vm_grade_isolation`, and `untrusted_eligible`.

- [x] **Step 2: Add host-local and non-overclaim assertions**

Assert:

```python
assert discovery["seatbelt"]["boundary_class"] == "host_local"
assert discovery["seatbelt"]["vm_grade_isolation"] is False
assert discovery["seatbelt"]["untrusted_eligible"] is False
assert discovery["worktree"]["boundary_class"] == "host_local"
assert discovery["worktree"]["vm_grade_isolation"] is False
assert discovery["worktree"]["untrusted_eligible"] is False
assert discovery["docker"]["boundary_class"] == "container"
assert discovery["docker"]["vm_grade_isolation"] is False
assert discovery["docker"]["untrusted_eligible"] is True
assert discovery["vz_macos"]["boundary_class"] == "vm_grade_scaffold"
assert discovery["vz_macos"]["vm_grade_isolation"] is False
assert discovery["vz_macos"]["untrusted_eligible"] is False
```

- [x] **Step 3: Run the test and verify it fails**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py -q --timeout=60
```

Expected: fail because the new fields are missing.

### Task 2: Implement Runtime Isolation Metadata

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/runtime_capabilities.py`
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`

- [x] **Step 1: Define isolation metadata types and map**

Add a `RuntimeBoundaryClass` literal, an immutable `RuntimeIsolationMetadata` dataclass, and a `runtime_isolation_metadata(runtime)` helper in `runtime_capabilities.py`.

- [x] **Step 2: Expose metadata through discovery**

Import `runtime_isolation_metadata` in `service.py` and merge the metadata into each runtime row through the existing shared helper path.

- [x] **Step 3: Update the Pydantic response schema**

Add `boundary_class`, `vm_grade_isolation`, and `untrusted_eligible` fields to `SandboxRuntimeInfo` with descriptions that distinguish policy eligibility from current availability.

- [x] **Step 4: Run focused tests**

Run the runtime inventory test again. Expected: pass.

### Task 3: Update Runtime Contract Documentation

**Files:**
- Modify: `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- Modify: `Docs/Sandbox/sandbox-security-policy-matrix.md`
- Modify: `Docs/API-related/Sandbox_API.md`
- Modify: `Docs/Published/API-related/Sandbox_API.md`

- [x] **Step 1: Document the new discovery fields**

Describe the new machine-readable fields near the existing `/api/v1/sandbox/runtimes` explanation.

- [x] **Step 2: Add the runtime posture table**

Document the runtime mapping and the fact that `untrusted_eligible` is policy eligibility, not availability.

- [x] **Step 3: Run public docs contract tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Docs/test_sandbox_public_docs_contract.py -q --timeout=60
```

Expected: pass.

### Task 4: Verify, Update Task, Commit, And Open PR

**Files:**
- Modify: `backlog/tasks/task-36 - Add-structured-sandbox-runtime-isolation-metadata.md`

- [x] **Step 1: Run focused sandbox verification**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py tldw_Server_API/tests/sandbox/test_feature_discovery_flags.py tldw_Server_API/tests/Docs/test_sandbox_public_docs_contract.py -q --timeout=60
```

Expected: pass for direct runtime/docs contract tests. During execution,
`test_feature_discovery_flags.py` timed out in existing FastAPI `TestClient`
teardown/background worker shutdown both alone and in the combined suite, so
TASK-36 records that as a known verification note rather than a metadata
regression.

- [x] **Step 2: Run Bandit on touched Python code**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/Sandbox/runtime_capabilities.py tldw_Server_API/app/core/Sandbox/service.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py -f json -o /tmp/bandit_sandbox_runtime_isolation_metadata.json
```

Expected: no new findings in touched code.

- [x] **Step 3: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [x] **Step 4: Update TASK-36**

Check completed acceptance criteria, add verification notes, and add a final summary.

- [ ] **Step 5: Commit and open PR**

Commit the branch and open a pull request against `dev`.
