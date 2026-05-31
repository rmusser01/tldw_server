# VZ Orphan VM Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow the explicit macOS reconciliation repair endpoint to terminate orphaned helper VMs when the operator requests mutation.

**Architecture:** Keep diagnostics read-only and keep repair dry-run by default. Reuse the existing reconciliation report and `MacOSVirtualizationHelperClient.terminate_vm()` contract, adding only explicit `terminate_orphaned_vms=True` behavior for items marked `orphaned_vm`.

**Tech Stack:** Python, FastAPI/Pydantic schemas, pytest, existing macOS helper client abstraction.

---

### Task 1: Add Service Tests For Orphan VM Termination

**Files:**
- Modify: `tldw_Server_API/tests/Sandbox/test_admin_macos_reconciliation_repair.py`

- [x] **Step 1: Write failing tests**

Add tests that verify:
- dry-run with `terminate_orphaned_vms=True` plans `terminate_orphaned_vm` actions without calling the helper
- mutating repair calls helper `terminate_vm(vm_id)` and increments `terminated_orphaned_vms`
- helper returning `False` records `missing` without incrementing the terminated count
- helper exception maps to `SandboxReconciliationRepairError("vz_orphan_vm_termination_failed", 503)`

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sandbox/test_admin_macos_reconciliation_repair.py -q
```

Expected: new orphan termination tests fail because the service still rejects `terminate_orphaned_vms=True`.

### Task 2: Wire Existing Helper Termination Into Repair

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`

- [x] **Step 1: Implement minimal service behavior**

Remove the unconditional unsupported error. For each `orphaned_vm` item with a VM id:
- append a `terminate_orphaned_vm` action when `terminate_orphaned_vms=True`
- use `planned` during dry-run
- call `MacOSVirtualizationHelperClient().terminate_vm(vm_id)` only when `dry_run=False`
- record `terminated` or `missing`
- increment `terminated_orphaned_vms` only on `terminated`

- [x] **Step 2: Run focused tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sandbox/test_admin_macos_reconciliation_repair.py -q
```

Expected: repair tests pass.

### Task 3: Update API And Operator Docs

**Files:**
- Modify: `tldw_Server_API/tests/Sandbox/test_admin_macos_reconciliation_repair.py`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `tools/macos-vz-helper/README.md`

- [x] **Step 1: Replace endpoint rejection test**

Update the API test that currently expects `orphan_termination_not_supported` so it asserts `terminate_orphaned_vms=True` is passed through to the service.

- [x] **Step 2: Update docs**

Replace claims that orphan VM termination is unsupported/manual with the new explicit, dry-run-first repair behavior.

- [x] **Step 3: Run focused verification**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sandbox/test_admin_macos_reconciliation_repair.py -q
git diff --check
```

Expected: tests pass and diff check reports no whitespace errors.
