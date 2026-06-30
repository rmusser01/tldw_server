# MCP Filesystem Lock Manager Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an injectable filesystem lock-manager seam while preserving the current process-local in-memory default.

**Architecture:** Keep `fs.lock_acquire`, `fs.lock_release`, and mutation lock validation behavior in `FilesystemModule`, but depend on a narrow lock-manager protocol instead of a hard-coded concrete manager. Add a small backend factory that supports only `memory` now and fails closed for unsupported configured backends, leaving persistent/shared backends for a later slice.

**Tech Stack:** Python 3, pytest, MCP Unified filesystem module, Backlog.md task `TASK-2343`.

---

### Task 1: Add Lock-Manager Contract Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`
- Reference: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Reference: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py`

- [x] **Step 1: Add a failing shared-manager injection test**

Add an async test that creates one `InMemoryFilesystemLockManager`, injects it into two `FilesystemModule` instances, acquires a lock through module A, observes a conflict through module B, releases through module B with the same lease token, and then acquires through module B.

- [x] **Step 2: Run the new test and verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py::test_filesystem_lock_manager_injection_shares_leases_between_modules -q`

Expected: FAIL because `FilesystemModule.__init__()` does not accept a lock manager injection argument.

### Task 2: Implement The Seam

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`

- [x] **Step 1: Add `FilesystemLockManager` protocol**

Define a protocol with `acquire`, `release`, and `validate` signatures matching the existing in-memory manager.

- [x] **Step 2: Add a default manager factory**

Add `create_filesystem_lock_manager(settings)` that returns `InMemoryFilesystemLockManager` for unset, `memory`, or `in_memory` backends and raises `ValueError` for unsupported configured backends.

- [x] **Step 3: Wire `FilesystemModule` to the seam**

Add an optional `lock_manager` constructor parameter. Store the injected manager when supplied, otherwise call `create_filesystem_lock_manager(config.settings)`. Do not change existing lock result shapes or mutation behavior.

- [x] **Step 4: Run the shared-manager test and verify it passes**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py::test_filesystem_lock_manager_injection_shares_leases_between_modules -q`

Expected: PASS.

### Task 3: Add Config And Documentation Coverage

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`
- Modify: `mcp_unified/USER_GUIDE.md`

- [x] **Step 1: Add a failing unsupported-backend config test**

Add a test that constructs `FilesystemModule(ModuleConfig(name="filesystem", settings={"lock_manager_backend": "sqlite"}))` and expects `ValueError`.

- [x] **Step 2: Run the config test and verify it fails before implementation or passes after implementation**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py::test_filesystem_lock_manager_rejects_unsupported_backend_config -q`

Expected after Task 2: PASS.

- [x] **Step 3: Update the user guide**

Clarify that the built-in backend is `lock_manager_backend=memory`, the default remains process-local, and durable/shared backends are future implementations behind the same seam.

### Task 4: Validate And Commit

**Files:**
- Validate touched Python and docs.
- Update Backlog task `TASK-2343`.

- [x] **Step 1: Run focused tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q`

Expected: PASS.

- [x] **Step 2: Compile touched Python**

Run: `source .venv/bin/activate && python -m py_compile tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`

Expected: exit 0.

- [x] **Step 3: Run Bandit on touched implementation scope**

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py -f json -o /tmp/bandit_mcp_fs_lock_manager_seam.json`

Expected: exit 0 with no new findings in touched code.

- [x] **Step 4: Check diff whitespace**

Run: `git diff --check`

Expected: exit 0.

- [x] **Step 5: Update Backlog task and commit**

Record verification results in `TASK-2343`, stage the changed files, and commit with a message linking the MCP filesystem lock seam.
