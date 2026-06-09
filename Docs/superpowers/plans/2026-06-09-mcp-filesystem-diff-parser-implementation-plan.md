# MCP Filesystem Diff Parser V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve `fs.patch` unified-diff fidelity for safe, governed agentic file editing.

**Architecture:** Keep the parser in `filesystem_diff.py` and the filesystem enforcement in `filesystem_module.py`. Add parser state for no-final-newline markers and robust header path parsing without bypassing existing path grants, read receipts, hash checks, or atomic writes.

**Tech Stack:** Python 3.11, pytest, existing MCP Unified filesystem module, Bandit.

---

### Task 1: Parser Fidelity Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py`

- [ ] **Step 1: Add failing parser tests**

Add tests for:
- paths with spaces in `---` / `+++` headers;
- applying a patch that preserves missing final newline;
- rejecting orphan `\ No newline at end of file` markers.

- [ ] **Step 2: Run parser tests and verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py \
  -q
```

Expected: new tests fail against current parser behavior.

### Task 2: Parser Implementation

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py`

- [ ] **Step 1: Extend hunk line model**

Add a boolean field to `PatchHunkLine` to track whether that logical line has a trailing newline in the target file.

- [ ] **Step 2: Preserve no-newline markers**

When `_parse_hunk()` sees `\ No newline at end of file`, attach that state to the previous hunk line. Reject the marker if it appears before any hunk line.

- [ ] **Step 3: Apply additions without forcing EOF newline**

Update `apply_patch_to_text()` so added lines honor the parsed newline flag.

- [ ] **Step 4: Improve header path parsing**

Prefer tab-separated header metadata. Otherwise keep the full path text so filenames with spaces are preserved, then run existing path normalization.

- [ ] **Step 5: Run parser tests and verify GREEN**

Run the parser test command from Task 1.

### Task 3: End-to-End `fs.patch` Coverage

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`

- [ ] **Step 1: Add failing module test**

Add an async test that patches an existing file to content without a final newline, using `expected_sha256_by_path`.

- [ ] **Step 2: Run focused module test and verify RED/GREEN**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py::test_filesystem_patch_preserves_no_final_newline \
  -q
```

Expected: fail before implementation, pass after implementation.

### Task 4: Final Verification And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-2332 - Implement-MCP-filesystem-unified-diff-parser-and-governed-patch-primitive.md`

- [ ] **Step 1: Run focused filesystem suite**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py \
  -q
```

- [ ] **Step 2: Run Bandit**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py \
  -f json -o /tmp/bandit_mcp_fs_diff_parser_v2.json
```

- [ ] **Step 3: Run diff hygiene**

```bash
git diff --check
```

- [ ] **Step 4: Update TASK-2332**

Record implementation notes, verification results, known non-goals, and final summary.

- [ ] **Step 5: Commit**

```bash
git add Docs/superpowers/specs/2026-06-09-mcp-filesystem-diff-parser-design.md \
  Docs/superpowers/plans/2026-06-09-mcp-filesystem-diff-parser-implementation-plan.md \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py \
  "backlog/tasks/task-2332 - Implement-MCP-filesystem-unified-diff-parser-and-governed-patch-primitive.md"
git commit -m "fix: improve MCP filesystem patch parser fidelity"
```
