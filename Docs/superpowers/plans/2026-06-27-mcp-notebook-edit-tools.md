# MCP Notebook Edit Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add notebook-safe MCP tools for reading Jupyter notebook structure and editing cells by id without exposing whole-notebook overwrite behavior.

**Architecture:** Keep notebook JSON parsing and cell mutation in a focused helper module, then expose `notebook.read` and `notebook.edit_cell` through the existing `FilesystemModule` so path grants, read receipts, lock leases, and reporting stay on the established MCP filesystem path. Preserve the conservative policy model: read uses file-policy `read`, cell mutation uses file-policy `edit`, and source output is explicit and bounded.

**Tech Stack:** Python 3.11, FastAPI-adjacent MCP module code, stdlib `json`/`hashlib`/`pathlib`, pytest/pytest-asyncio, existing `mcp_unified` package contracts.

---

## Source Documents

- Spec: `Docs/Design/2026-06-27-mcp-notebook-edit-tools-design.md`
- Backlog task: `backlog/tasks/task-2282 - Add-NotebookEdit-style-notebook-file-tools.md`

## Baseline

Before implementation, the existing filesystem module test file produced:

- 103 passed
- 1 failed: `test_filesystem_glob_marks_file_size_unavailable`
- Failure: `OSError("metadata unavailable")` from the existing glob path

Do not touch or fix this unrelated glob failure unless notebook work directly changes it. Use narrower notebook-focused tests for red/green verification, and record the baseline failure in `TASK-2282`.

## File Structure

- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/notebook_files.py`
  - Pure notebook parsing, validation, summary, mutation, and serialization helpers.
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
  - Tool definitions, argument validation, workspace/preimage/lock integration, execution dispatch.
- Modify: `apps/mcp-unified/src/mcp_unified/profiles/presets.py`
  - Add notebook tools to existing file read/edit tool preset collections.
- Modify: `apps/mcp-unified/src/mcp_unified/USER_GUIDE.md`
  - Document notebook read/edit flow, tool allow-list requirements, and path grants.
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_notebook_files.py`
  - Helper-level unit tests.
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_notebook_tools.py`
  - MCP module integration tests for tools, receipts, locks, path metadata, and summaries.
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`
  - Preset inclusion tests.

Use the existing root virtualenv while working in the worktree:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest ...
```

## Task 1: Notebook Helper Read Model

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/notebook_files.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_notebook_files.py`

- [x] **Step 1: Write failing helper tests for valid read summaries**

Add tests that build small notebook JSON payloads as bytes and assert:

```python
state = parse_notebook_payload(payload)
summary = summarize_notebook(state, include_source=False)
assert summary["cell_count"] == 2
assert summary["cells"][0]["id"] == "markdown-1"
assert "source_preview" not in summary["cells"][0]
```

Also test `include_source=True`, `cell_ids=["code-1"]`, `max_source_chars`, and `max_total_source_chars`.

- [x] **Step 2: Run tests to verify red**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_notebook_files.py -q
```

Expected: fail because `notebook_files.py` does not exist.

- [x] **Step 3: Implement minimal parser and summary helpers**

Implement stdlib-only helpers:

```python
@dataclass(frozen=True, slots=True)
class NotebookFormat:
    trailing_newline: bool
    indent: int

@dataclass(frozen=True, slots=True)
class ParsedNotebook:
    document: dict[str, Any]
    payload: bytes
    sha256: str
    size: int
    format: NotebookFormat
```

Functions:

```python
def parse_notebook_payload(payload: bytes, *, max_bytes: int | None = None) -> ParsedNotebook: ...
def summarize_notebook(
    notebook: ParsedNotebook,
    *,
    include_source: bool = False,
    cell_ids: list[str] | None = None,
    max_source_chars: int = 4_000,
    max_total_source_chars: int = 20_000,
) -> dict[str, Any]: ...
```

Validation requirements:

- UTF-8 JSON object.
- Top-level `cells` list.
- Each cell is an object with non-empty string `id`.
- Duplicate ids raise `ValueError("notebook_duplicate_cell_id")`.
- Missing ids raise `ValueError("notebook_cell_id_required")`.
- Source may be string, list of strings, or absent.

- [x] **Step 4: Run tests to verify green**

Run the same pytest command. Expected: all helper read tests pass.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/notebook_files.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_notebook_files.py
git commit -m "feat: add notebook file read helpers"
```

## Task 2: Notebook Helper Cell Mutations

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/notebook_files.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_notebook_files.py`

- [x] **Step 1: Write failing tests for replace, insert, delete, and serialization**

Test behavior:

- `replace` changes only target cell source.
- Replacing a code cell clears `outputs` and sets `execution_count` to `None`.
- `insert` supports `insert_position="before"` and `"after"`.
- Inserted cells receive a unique generated id when `new_cell_id` is omitted.
- Caller-provided `new_cell_id` must be unique and valid.
- `delete` removes only the target cell.
- Serialization preserves trailing newline when input had one.
- Replaced list-style source remains list-style; string-style remains string-style.

Example assertion:

```python
result = apply_cell_edit(
    parsed,
    mode="replace",
    cell_id="code-1",
    source="print('new')\n",
)
assert result.summary["output_count_before"] == 1
assert result.summary["output_count_after"] == 0
assert result.document["cells"][1]["execution_count"] is None
```

- [x] **Step 2: Run tests to verify red**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_notebook_files.py -q
```

Expected: fail because mutation helpers are missing.

- [x] **Step 3: Implement mutation helpers**

Add:

```python
@dataclass(frozen=True, slots=True)
class NotebookEditResult:
    document: dict[str, Any]
    data: bytes
    sha256_after: str
    bytes_after: int
    summary: dict[str, Any]

def apply_cell_edit(
    notebook: ParsedNotebook,
    *,
    mode: str,
    cell_id: str,
    source: str | None = None,
    cell_type: str | None = None,
    insert_position: str | None = None,
    new_cell_id: str | None = None,
) -> NotebookEditResult: ...
```

Use `copy.deepcopy()` before mutation. Raise stable `ValueError` reason codes:

- `notebook_cell_id_not_found`
- `notebook_invalid_mode`
- `notebook_insert_position_required`
- `notebook_source_required`
- `notebook_invalid_cell_type`
- `notebook_duplicate_cell_id`

- [x] **Step 4: Run tests to verify green**

Run helper tests. Expected: pass.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/notebook_files.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_notebook_files.py
git commit -m "feat: add notebook cell edit helpers"
```

## Task 3: MCP Tool Definitions And Validation

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_notebook_tools.py`

- [x] **Step 1: Write failing tool metadata and validation tests**

Create tests that instantiate `FilesystemModule` and assert:

- `notebook.read` and `notebook.edit_cell` appear in `get_tools()`.
- Both tools have `additionalProperties: False`.
- `notebook.read` has `readOnlyHint=True`, `path_scope_action="read"`, and file-policy `read`.
- `notebook.edit_cell` has `write_capable=True`, `path_scope_action="edit"`, and file-policy `edit`.
- Validation rejects unknown args, non-string `path`, invalid modes, missing source for replace/insert, missing `insert_position` for insert, invalid booleans, and missing preimage for edit.

- [x] **Step 2: Run tests to verify red**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_notebook_tools.py -q
```

Expected: fail because tools are not defined.

- [x] **Step 3: Add tool definitions and validators**

In `get_tools()`, add `notebook.read` near `fs.read` and `notebook.edit_cell` near `fs.edit`.

Use metadata:

```python
{
    "category": "retrieval",
    "readOnlyHint": True,
    "capabilities": ["filesystem.read", "notebook.read"],
    "path_scope_action": "read",
    **_file_policy_metadata("read"),
    **shared_path_metadata,
}
```

and:

```python
{
    "category": "management",
    "readOnlyHint": False,
    "write_capable": True,
    "capabilities": ["filesystem.edit", "notebook.edit"],
    "path_scope_action": "edit",
    **_file_policy_metadata("edit"),
    **shared_path_metadata,
}
```

Extend `validate_tool_arguments()` with dedicated branches for both tools.

- [x] **Step 4: Run tests to verify green**

Run the notebook tool test file. Expected: metadata and validation tests pass.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_notebook_tools.py
git commit -m "feat: register notebook mcp tools"
```

## Task 4: MCP Execution Integration

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_notebook_tools.py`

- [x] **Step 1: Write failing execution tests**

Use a fake workspace root resolver and temp `.ipynb` files. Assert:

- `notebook.read` returns structure-only summaries by default.
- `notebook.read` issues a read receipt when configured and full hash is available.
- `notebook.edit_cell` requires `expected_sha256` or `read_receipt`.
- Replace, insert, and delete update the file and return bounded summaries.
- `dry_run=True` returns `edited=False` and does not write.
- Stale `expected_sha256` raises `ValueError("edit_preimage_mismatch")` or notebook-specific equivalent.
- Non-`.ipynb` path raises `ValueError("notebook_path_required")`.
- Invalid JSON raises `ValueError("notebook_invalid_json")`.
- Code cell replacement clears outputs and execution count.

- [x] **Step 2: Run tests to verify red**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_notebook_tools.py -q
```

Expected: execution tests fail because dispatch is missing.

- [x] **Step 3: Implement execution dispatch**

Add branches in `execute_tool()`:

- Resolve paths with `_resolve_workspace_path_no_follow()`.
- Reject non-`.ipynb` suffix.
- Read bytes using existing no-follow regular-file helper.
- Use helper parser/summarizer for `notebook.read`.
- Issue read receipts with the same manager and context metadata as `fs.read`.
- For edits, call `_validate_mutation_lock()`, authorize preimage with existing edit receipt logic, call `apply_cell_edit()`, enforce write-size limit, recheck preimage, and write atomically.

Return eval metadata with:

```python
build_execution_eval_metadata(
    tool_name="notebook.read",
    tool_prompt_id="mcp.notebook.read.v1",
    tool_prompt_version="2026.06.27",
    action_family="notebook_read",
    result_kind="structured_notebook_read",
    path_filter_used=True,
    truncated=...,
)
```

and similarly `notebook.edit_cell` / `notebook_edit`.

- [x] **Step 4: Run tests to verify green**

Run the notebook tool test file. Expected: pass.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_notebook_tools.py
git commit -m "feat: execute notebook mcp tools"
```

## Task 5: Presets And Documentation

**Files:**
- Modify: `apps/mcp-unified/src/mcp_unified/profiles/presets.py`
- Modify: `apps/mcp-unified/src/mcp_unified/USER_GUIDE.md`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`

- [x] **Step 1: Write failing preset tests**

Update tests to assert:

```python
assert "notebook.read" in presets._FILES_READ_TOOLS
assert "notebook.edit_cell" in presets._FILES_EDIT_TOOLS
assert "notebook.edit_cell" in presets._FILES_WRITE_TOOLS
```

Also assert legacy read/write lists are unchanged.

- [x] **Step 2: Run tests to verify red**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -q
```

Expected: fail because presets do not include notebook tools.

- [x] **Step 3: Update presets and user guide**

Add:

```python
_FILES_READ_TOOLS = ["fs.list", "fs.read", "fs.stat", "fs.glob", "fs.grep", "notebook.read"]
_FILES_EDIT_TOOLS = [*_FILES_READ_TOOLS, "fs.patch", "notebook.edit_cell"]
```

Update `USER_GUIDE.md` near the safe file read/patch/write section:

- explain structure-first `notebook.read`;
- explain `notebook.edit_cell` replace/insert/delete by id;
- show path grant example using `read` and `edit`;
- state that path grants do not grant the tool itself;
- state code-cell source replacement clears stale outputs.

- [x] **Step 4: Run tests to verify green**

Run profile preset tests. Expected: pass.

- [x] **Step 5: Commit**

```bash
git add apps/mcp-unified/src/mcp_unified/profiles/presets.py \
  apps/mcp-unified/src/mcp_unified/USER_GUIDE.md \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py
git commit -m "docs: document notebook mcp tools"
```

## Task 6: Focused Regression And Security Verification

**Files:**
- Modify: `backlog/tasks/task-2282 - Add-NotebookEdit-style-notebook-file-tools.md`

- [x] **Step 1: Run focused tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_notebook_files.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_notebook_tools.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  -q
```

Expected: pass.

- [x] **Step 2: Re-run existing filesystem baseline**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q
```

Expected: either the same pre-existing glob failure remains, or the file passes. If a new notebook-related failure appears, fix it before continuing.

- [x] **Step 3: Run Bandit on touched Python files**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/notebook_files.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py \
  apps/mcp-unified/src/mcp_unified/profiles/presets.py \
  -f json -o /tmp/bandit_mcp_notebook_edit_tools.json
```

Expected: no new findings in touched code. If Bandit is unavailable, record the environment failure on `TASK-2282`.

- [x] **Step 4: Update Backlog task**

Record:

- implementation summary;
- touched files;
- test commands and results;
- Bandit result path;
- known baseline filesystem glob failure if still present.

- [x] **Step 5: Final self-review**

Run:

```bash
git diff --check
git status --short
git log --oneline --decorate -5
```

Expected: no whitespace errors, only intended files changed or clean after commit.

- [x] **Step 6: Commit task finalization**

```bash
git add 'backlog/tasks/task-2282 - Add-NotebookEdit-style-notebook-file-tools.md'
git commit -m "chore: finalize task 2282 notebook tools"
```

## Task 7: PR Review Remediation

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/notebook_files.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_notebook_tools.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_notebook_files.py`
- Modify: `backlog/tasks/task-2282 - Add-NotebookEdit-style-notebook-file-tools.md`

- [x] **Step 1: Rebase on latest dev**

Run `git fetch origin dev` and `git rebase origin/dev`. Expected: branch is based on current `origin/dev`.

- [x] **Step 2: Verify review comments against code**

Inspect unresolved PR review threads. Valid review items:

- add docstrings to the new notebook test modules and new symbols;
- map oversized notebook reads/edits to notebook-specific reason codes;
- validate existing notebook cell types during parsing;
- strip code-only metadata when a replace changes a cell to markdown/raw;
- add receipt-authorized notebook edit coverage.

- [x] **Step 3: Write failing tests for behavior changes**

Add tests for oversized notebook reads/edits, invalid existing `cell_type`, non-code replace metadata stripping, and read-receipt-based edits.

- [x] **Step 4: Implement fixes**

Keep changes minimal: normalize notebook-specific oversize errors at notebook call sites, validate canonical existing cell types in parsing, and strip `outputs` / `execution_count` from final non-code cells.

- [x] **Step 5: Verify focused tests and Bandit**

Run the notebook helper/tool tests, preset tests, existing filesystem baseline, and Bandit on touched Python files.

- [x] **Step 6: Update Backlog and push**

Record remediation summary and verification in `TASK-2282`, commit the fixes, force-with-lease push the rebased branch, and resolve or reply to addressed PR threads.
