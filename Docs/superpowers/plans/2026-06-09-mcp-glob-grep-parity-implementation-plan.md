# MCP Glob/Grep Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the existing workspace-bounded `fs.glob` and `fs.grep` tools closer to useful Claude Code Glob/Grep behavior without weakening path scope, no-shell execution, or safe literal-search defaults.

**Architecture:** Extend `FilesystemModule` in place. Keep all filesystem traversal inside the existing workspace resolver and `asyncio.to_thread` execution path. Add narrowly scoped schema, validation, search-filter, and response-shaping support while preserving existing response fields for compatibility.

**Tech Stack:** Python, FastAPI-side MCP module code, pytest, Backlog.md, Bandit.

---

### Task 1: Write Failing Parity Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`

- [x] **Step 1: Add `fs.glob` mtime sorting coverage**
  - Create files with controlled mtimes.
  - Assert default `fs.glob` returns newest-first paths with truncation metadata.
  - Assert `sort_by="path"` remains available for deterministic path-ordered results.

- [x] **Step 2: Add `fs.grep` output mode and filter coverage**
  - Assert `output_mode="files_with_matches"` returns only matched file paths and is the default.
  - Assert `output_mode="content"` preserves existing line records.
  - Assert `output_mode="count"` returns per-file counts.
  - Assert `glob` and `type` filters narrow the scan without bypassing workspace scope.
  - Assert directory grep respects `.gitignore` by default and can disable that filtering.
  - Assert safe multiline regex works for file/count outputs.

- [x] **Step 3: Add direct-file base path coverage**
  - Assert `fs.grep` can search a directly named text file via `base_path`.
  - Assert unsupported direct-file inputs still report bounded skip/error behavior.

- [x] **Step 4: Run targeted tests and confirm RED**
  - Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -k "glob or grep" -q`
  - Expected: fail on the new unsupported arguments/ordering/default-mode assertions.

### Task 2: Implement Bounded Search Parity Additions

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`

- [x] **Step 1: Extend tool schemas and validation**
  - Add `fs.glob.sort_by` with values `modified_at` and `path`; default to `modified_at`.
  - Add `fs.grep.output_mode` with values `files_with_matches`, `content`, and `count`; default to `files_with_matches`.
  - Add `fs.grep.glob` as a single-pattern convenience filter and `fs.grep.type` for a bounded extension alias map.

- [x] **Step 2: Implement `fs.glob` sorting**
  - Capture per-match mtime internally.
  - Sort newest-first when `sort_by="modified_at"`.
  - Preserve response compatibility by omitting internal sort metadata from records unless already present.

- [x] **Step 3: Implement `fs.grep` result shaping**
  - Collect matches internally as content records.
  - Shape output by mode after scanning.
  - Keep existing `matches`, `truncated`, `remaining_count`, `remaining_count_known`, `truncation_reasons`, and `skipped` fields.

- [x] **Step 4: Implement filters, gitignore handling, multiline, and direct-file search**
  - Merge `glob` into include filters.
  - Map common `type` aliases such as `py`, `js`, `ts`, `tsx`, `md`, `json`, `yaml`, `rust`, `go`, and `java` to portable include patterns.
  - Add root `.gitignore` handling using a declared `pathspec` dependency.
  - Add safe multiline regex support for `files_with_matches` and `count` output modes.
  - If `base_path` resolves to a file, search only that file while applying the same binary, decode, size, and budget checks.

- [x] **Step 5: Run targeted tests and confirm GREEN**
  - Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -k "glob or grep" -q`
  - Expected: pass.

### Task 3: Documentation, Backlog, And Verification

**Files:**
- Modify: `mcp_unified/USER_GUIDE.md`
- Modify: `backlog/tasks/task-2280 - Bring-fs.glob-and-fs.grep-to-Claude-Code-parity.md`

- [x] **Step 1: Update user guide**
  - Document mtime-sorted glob results and grep output modes/filters.
  - State that regex remains opt-in and no host shell is invoked.

- [x] **Step 2: Update Backlog.md task**
  - Record implementation notes, validation commands, and known follow-ups for gitignore and multiline behavior.

- [x] **Step 3: Run verification**
  - Run targeted pytest.
  - Run Bandit on the touched MCP filesystem module.
  - Run `git diff --check`.

- [x] **Step 4: Commit and open PR**
  - Commit the plan, tests, code, docs, and backlog task.
  - Push `codex/mcp-glob-grep-parity`.
  - Open a draft PR against `dev` with a human-readable change summary placeholder.
