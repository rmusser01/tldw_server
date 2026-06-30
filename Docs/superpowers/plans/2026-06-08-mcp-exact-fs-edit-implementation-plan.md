# MCP Exact fs.edit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Claude-style exact `fs.edit` MCP filesystem primitive for bounded string replacement.

**Architecture:** Extend the existing `FilesystemModule` rather than creating a parallel editor. `fs.edit` will reuse current workspace path resolution, UTF-8/binary guards, read-receipt/hash preimage authorization, atomic writes, eval metadata, and action-aware path-scope metadata.

**Tech Stack:** Python 3.11, pytest, MCP Unified filesystem module, existing read receipts and path-grant enforcement.

---

### Task 1: Tool Contract And Validation

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`

- [x] **Step 1: Write failing metadata/schema tests**

Add assertions that `fs.edit` appears in `get_tools()` with `path_scope_action == "edit"`, `file_policy_action == "edit"`, `write_capable is True`, strict `additionalProperties is False`, and eval metadata matching `filesystem_edit`.

- [x] **Step 2: Run the focused metadata test**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py::test_filesystem_tools_include_path_scope_metadata -q`

Expected: FAIL because `fs.edit` is not registered.

- [x] **Step 3: Add the `fs.edit` tool definition and argument validation**

Schema fields: `path`, `old_string`, `new_string`, `expected_sha256`, `read_receipt`, `replace_all`, and `dry_run`. Require `path`, `old_string`, and `new_string`. Reject empty `old_string`, non-string content fields, non-string preimage fields, non-boolean flags, and unknown arguments.

- [x] **Step 4: Rerun metadata test**

Expected: PASS.

### Task 2: Exact Edit Semantics

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`

- [x] **Step 1: Write failing behavior tests**

Cover:
- exact single replacement with `expected_sha256`
- rejection when no preimage is supplied
- rejection when `old_string` is absent
- rejection when `old_string` appears multiple times and `replace_all` is false
- `replace_all=true` replaces every occurrence
- `dry_run=true` reports planned output metadata without writing

- [x] **Step 2: Run the new behavior tests**

Run the new test names individually first.

Expected: FAIL because `fs.edit` dispatch/helper does not exist.

- [x] **Step 3: Implement `_edit_file()` and dispatch**

Use `_resolve_workspace_path_no_follow`, UTF-8 read guards, `hashlib.sha256`, `_authorize_edit_preimage()`, `_assert_preimage_unchanged()`, and `_atomic_write_text_file()`. Count exact literal occurrences with `str.count()`. Use `str.replace(old, new, 1)` unless `replace_all` is true. Do not return raw file content.

- [x] **Step 4: Rerun behavior tests**

Expected: PASS.

### Task 3: Read Receipts, Errors, And Policy Integration

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`
- Test: `tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py`

- [x] **Step 1: Write failing receipt and policy tests**

Cover:
- successful edit with a current `fs.read` receipt
- receipt mismatch across session/workspace fails with `edit_read_receipt_mismatch`
- binary/non-UTF-8 content is rejected with an `fs.edit`-specific error
- path enforcer treats `fs.edit` as an `edit` action in path grants

- [x] **Step 2: Run the focused tests**

Expected: FAIL until receipt validation and path metadata are wired.

- [x] **Step 3: Implement receipt validation and error metadata**

Mirror patch/write receipt validation with `edit_*` reason codes. Build `eval` metadata using `tool_name="fs.edit"`, prompt id `mcp.fs.edit.v1`, action family `filesystem_edit`, and result kind `structured_filesystem_edit`.

- [x] **Step 4: Rerun focused tests**

Expected: PASS.

### Task 4: Documentation, Verification, And Commit

**Files:**
- Modify: `mcp_unified/USER_GUIDE.md`
- Modify: `backlog/tasks/task-2279 - Add-Claude-style-exact-fs.edit-primitive.md`

- [x] **Step 1: Document `fs.edit` as the exact-string edit primitive**

Mention that `fs.patch` remains preferred for diff-first edits and `fs.edit` is for small exact replacements.

- [x] **Step 2: Run validation**

Run:
- focused filesystem/path-enforcement tests
- `ruff check` on touched Python files
- `py_compile` on touched production Python files
- `bandit` on touched production Python files
- `git diff --check`

- [x] **Step 3: Update Backlog task final summary**

Record what changed, what was verified, and any skipped checks.

- [x] **Step 4: Commit**

Commit message: `feat: add exact mcp fs edit primitive`
