# Portable Sandbox Runtime Capability Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a portable sandbox runtime capability gate that catches runtime discovery, metadata, taxonomy, and documentation drift.

**Architecture:** Keep the gate test-only unless it exposes a real contract gap. Inject synthetic runtime preflight rows into `SandboxService.feature_discovery()` so the test exercises the API-facing projection without host runtime probes.

**Tech Stack:** Python, pytest, Pydantic schemas, existing sandbox runtime capability helpers.

---

### Task 1: Add Portable Capability Gate

**Files:**
- Create: `tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py`
- Modify: `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- Modify: `backlog/tasks/task-72 - Add-portable-sandbox-runtime-capability-gate.md`

- [ ] **Step 1: Write the failing test**

Create a test that:

- builds one synthetic `RuntimePreflightResult` per `RuntimeType`
- monkeypatches `SandboxService._collect_runtime_preflights`
- calls `SandboxService().feature_discovery()`
- validates the result with `SandboxRuntimesResponse`
- asserts every runtime row has implementation state, normalized reasons, isolation metadata, network policy contract, and session contract
- asserts every runtime appears in `Docs/Sandbox/sandbox-runtime-capability-inventory.md`

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q
```

Expected: fail on the current branch because the new gate is stricter than existing coverage.

- [ ] **Step 3: Implement the smallest fix**

If the failure is only missing docs about the portable gate, update `Docs/Sandbox/sandbox-runtime-capability-inventory.md`.

If the failure exposes an actual metadata or taxonomy gap, patch the smallest source mapping or projection needed.

- [ ] **Step 4: Run focused verification**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py -q
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py
git diff --check
```

Run Bandit on touched production Python if production code changes. If this remains test/docs-only, record the skip reason.

- [ ] **Step 5: Commit**

```bash
git add Docs/Sandbox/sandbox-runtime-capability-inventory.md Docs/superpowers/specs/2026-05-05-portable-sandbox-runtime-capability-gate-design.md Docs/superpowers/plans/2026-05-05-portable-sandbox-runtime-capability-gate-implementation-plan.md "backlog/tasks/task-72 - Add-portable-sandbox-runtime-capability-gate.md" tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py
git commit -m "test(sandbox): add portable runtime capability gate"
```
