# MCP Filesystem Helper Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add workspace-bounded, cross-platform `fs.stat`, `fs.glob`, and `fs.grep` MCP tools as the first native filesystem helper slice after profile tooling discovery.

**Architecture:** Extend the existing `FilesystemModule` so all helpers reuse the current workspace-root resolver, path containment checks, argument validation, and `asyncio.to_thread` offloading. Implement portable glob/search logic in Python rather than delegating to OS shell commands, and update profile metadata so read-capable default profiles can discover the new helpers. The existing governed `run(command=...)` command runtime remains the shell-shaped surface; `bash`/`shell` compatibility aliases and CLI aliases for these helpers are a follow-up task, not part of this native-tool slice.

**Tech Stack:** Python 3.11, FastAPI-side MCP module framework, `pathlib`, `os.walk`, `fnmatch`, `re`, pytest, Bandit.

---

## File Map

- Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
  - Add tool definitions, argument validation, path/pattern helpers, and worker-thread implementations for `fs.stat`, `fs.glob`, and `fs.grep`.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`
  - Add TDD coverage for descriptors, validation, cross-platform path behavior, limits, binary/encoding skips, symlink policy, and protocol unknown-argument rejection.
- Modify `mcp_unified/profiles/presets.py`
  - Add the new read-only filesystem helpers to `_FILES_READ_TOOLS`.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`
  - Assert read-capable presets include/discover the new filesystem helpers in tooling metadata.
- Modify `mcp_unified/USER_GUIDE.md`
  - Add a short filesystem helper note under profile/tool discovery docs.
- Modify `backlog/tasks/<task>.md`
  - Track implementation notes, verification, final summary, and Definition of Done for the implementation task created before coding.
- Follow-up, not this slice: `tldw_Server_API/app/core/MCP_unified/modules/implementations/run_command_module.py`,
  `tldw_Server_API/app/core/MCP_unified/command_runtime/registry.py`,
  `tldw_Server_API/app/core/MCP_unified/command_runtime/adapters.py`, and
  command-runtime tests for governed CLI aliases.

## Task 0: Backlog And Baseline

**Files:**
- Modify: implementation Backlog task only.

- [ ] **Step 1: Create or locate the implementation Backlog task**

Use Backlog MCP search first with a query like
`MCP filesystem helper tools fs.stat fs.glob fs.grep`. If MCP is unavailable,
use the CLI fallback:

```bash
backlog search "MCP filesystem helper tools fs.stat fs.glob fs.grep" --plain
```

Expected: no duplicate active implementation task. If none exists, create a task titled `Implement MCP filesystem helper tools`.

- [ ] **Step 2: Record baseline status**

Run:

```bash
git status --short --branch
```

Expected: clean branch before implementation edits.

- [ ] **Step 3: Read existing module and tests**

Read:

```bash
sed -n '1,360p' tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py
sed -n '1,520p' tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
```

Expected: confirm existing `fs.list`, `fs.read_text`, and `fs.write_text` patterns.

## Task 1: Tool Schemas And Validation

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`

- [ ] **Step 1: Write failing descriptor test**

Add a test similar to:

```python
@pytest.mark.asyncio
async def test_filesystem_tools_include_stat_glob_and_grep_metadata() -> None:
    resolver = _FakeWorkspaceRootResolver({"workspace_root": "/workspace/root"})
    mod = FilesystemModule(ModuleConfig(name="filesystem"), workspace_root_resolver=resolver)

    tools = await mod.get_tools()
    by_name = {tool["name"]: tool for tool in tools}

    assert {"fs.stat", "fs.glob", "fs.grep"} <= set(by_name)
    for tool_name in ("fs.stat", "fs.glob", "fs.grep"):
        schema = by_name[tool_name]["inputSchema"]
        metadata = by_name[tool_name]["metadata"]
        assert schema["additionalProperties"] is False
        assert metadata["uses_filesystem"] is True
        assert metadata["path_boundable"] is True
        assert "filesystem.read" in metadata["capabilities"]
        assert metadata["readOnlyHint"] is True
```

- [ ] **Step 2: Run descriptor test and verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py::test_filesystem_tools_include_stat_glob_and_grep_metadata -q
```

Expected: FAIL because tools are not defined.

- [ ] **Step 3: Add minimal tool definitions**

In `get_tools()`, add descriptors for:

- `fs.stat`
- `fs.glob`
- `fs.grep`

Use metadata:

```python
{
    "category": "retrieval",
    "readOnlyHint": True,
    "capabilities": ["filesystem.read"],
    **shared_fs_metadata,
}
```

Set `additionalProperties` to `False` for all three schemas.

- [ ] **Step 4: Add validation tests for unknown and malformed arguments**

Add tests that call `validate_tool_arguments()` directly for:

- unknown keys rejected,
- required `path`/`pattern` rejected when missing or blank,
- booleans must be booleans,
- limits must be positive integers,
- include/exclude must be lists of strings for `fs.grep`.
- regex patterns above the configured maximum length are rejected before
  compilation.

- [ ] **Step 5: Run validation tests and verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q -k "stat_glob_grep_metadata or validates_new_filesystem_helper_arguments"
```

Expected: FAIL until validation branches exist.

- [ ] **Step 6: Implement validation branches**

Update `validate_tool_arguments()` with explicit allowed-key sets:

- `fs.stat`: `path`, `follow_symlinks`
- `fs.glob`: `pattern`, `base_path`, `include_hidden`, `include_files`,
  `include_directories`, `follow_symlinks`, `case_sensitive`, `limit`
- `fs.grep`: `pattern`, `base_path`, `include`, `exclude`, `regex`,
  `case_sensitive`, `include_hidden`, `follow_symlinks`, `limit`,
  `max_file_bytes`

Do not expose walk-limit settings as caller arguments. Read them from module
settings so users cannot raise traversal limits through a tool call.

- [ ] **Step 7: Run Task 1 tests and verify GREEN**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q -k "stat_glob_grep_metadata or validates_new_filesystem_helper_arguments"
```

Expected: PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
git commit -m "feat: add mcp filesystem helper schemas"
```

## Task 2: `fs.stat`

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`

- [ ] **Step 1: Write failing stat behavior tests**

Add tests for:

- file stat returns `path`, `name`, `type == "file"`, `size`, and `is_symlink is False`,
- directory stat returns `type == "directory"`,
- missing path raises `FileNotFoundError`,
- `../escape.txt` raises `PermissionError`,
- symlink stat does not leak target path.

Use exact permission masks only as optional presence checks. Do not assert
platform-specific mode values.

- [ ] **Step 2: Run stat tests and verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q -k "filesystem_stat"
```

Expected: FAIL because `fs.stat` is not implemented.

- [ ] **Step 3: Implement `fs.stat` dispatch**

In `execute_tool()`, add:

```python
if tool_name == "fs.stat":
    target = self._resolve_workspace_path(workspace_root, str(args.get("path")))
    return await asyncio.to_thread(
        self._stat_path,
        workspace_root,
        target,
        bool(args.get("follow_symlinks", False)),
    )
```

- [ ] **Step 4: Implement `_stat_path()`**

Implementation requirements:

- use `target.lstat()` when not following symlinks,
- use `target.stat()` only after verifying the resolved target stays inside
  the workspace,
- return workspace-relative `/` path,
- never return symlink target text,
- include `modified_at` as UTC ISO text when stat succeeds.

- [ ] **Step 5: Run stat tests and verify GREEN**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q -k "filesystem_stat"
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
git commit -m "feat: add workspace bounded fs stat"
```

## Task 3: `fs.glob`

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`

- [ ] **Step 1: Write failing glob tests**

Add tests for:

- `pattern="**/*.py"` returns sorted `/`-separated paths,
- backslash input patterns normalize to `/`,
- `case_sensitive=True` is deterministic across OS,
- `case_sensitive=False` matches mixed-case file names,
- hidden files are excluded by default and included when requested,
- result limit sets `truncated=True` and `remaining_count`,
- traversal cap stops broad walks even when the result limit is not reached,
- absolute, drive-qualified, UNC, and parent-traversal patterns are rejected,
- symlinked directories are not followed by default,
- symlinked directories that resolve outside the workspace are rejected when
  `follow_symlinks=true`,
- hidden means a dot-prefixed workspace-relative segment, not platform hidden
  attributes.

- [ ] **Step 2: Run glob tests and verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q -k "filesystem_glob"
```

Expected: FAIL because `fs.glob` is not implemented.

- [ ] **Step 3: Add portable pattern helpers**

Add helper methods:

- `_normalize_portable_pattern(pattern: str) -> str`
- `_reject_unsafe_pattern(pattern: str) -> None`
- `_portable_pattern_matches(path: str, pattern: str, *, case_sensitive: bool) -> bool`
- `_is_hidden_relative_path(path: str) -> bool`
- `_bounded_positive_int(value: Any, default: int) -> int`
- `_setting_positive_int(name: str, default: int) -> int`

Reject:

- blank patterns,
- absolute POSIX paths,
- Windows drive prefixes like `C:/...`,
- UNC roots like `//server/share`,
- `..` path segments.

Use `_setting_positive_int()` for `glob_result_limit`,
`glob_walk_entry_limit`, `grep_result_limit`, `grep_walk_entry_limit`,
`grep_max_file_bytes`, `grep_max_total_bytes`, `grep_max_files`, and
`grep_max_pattern_length`. Gate regex matching behind `grep_allow_regex`.

- [ ] **Step 4: Implement `fs.glob` dispatch and worker**

Use `os.walk()` from the resolved `base_path`.

For each candidate:

- compute workspace-relative POSIX path,
- skip hidden paths unless requested,
- skip symlink traversal unless `follow_symlinks=true` and target remains in
  scope,
- match with `_portable_pattern_matches()` so `**/*.py` matches both `app.py`
  and `pkg/app.py`,
- lower pattern and candidate only when `case_sensitive=false`,
- increment a visited-entry counter and stop with `truncated=True` once
  `glob_walk_entry_limit` is reached,
- sort by path,
- return records capped by `limit`.

- [ ] **Step 5: Run glob tests and verify GREEN**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q -k "filesystem_glob"
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
git commit -m "feat: add portable workspace glob"
```

## Task 4: `fs.grep`

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`

- [ ] **Step 1: Write failing grep tests**

Add tests for:

- literal search returns path, 1-based line number, line text, and match text,
- regex search works when `regex=true`,
- `case_sensitive=false` matches mixed-case text,
- CRLF/LF mixed newline files report stable line numbers,
- binary files are skipped with `skipped["binary"]`,
- invalid UTF-8 files are skipped with `skipped["decode_error"]`,
- oversized files are skipped without full read,
- include/exclude glob filters apply,
- limit truncates matches deterministically.
- traversal cap stops broad walks deterministically,
- overly long regex patterns raise a validation error,
- invalid regex patterns return an actionable error without reading files,
- symlinked files outside the workspace are rejected or skipped without leaking
  targets,
- symlink loops cannot cause unbounded traversal.

- [ ] **Step 2: Run grep tests and verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q -k "filesystem_grep"
```

Expected: FAIL because `fs.grep` is not implemented.

- [ ] **Step 3: Implement grep file selection**

Reuse the same portable pattern and directory walk helpers from `fs.glob`.

Default:

```python
include = ["*", "**/*"]
exclude = []
```

Only files are searched.

- [ ] **Step 4: Implement grep matching**

Implementation requirements:

- reject overly long regex patterns before compiling,
- report `re.error` messages as actionable tool errors,
- read bytes only after checking file size,
- skip NUL-containing files as binary,
- decode UTF-8 only,
- use `splitlines()` for cross-platform line handling,
- default literal search via `str.find`,
- regex search via compiled `re.Pattern`,
- sort by normalized path and line number,
- increment a visited-entry counter and stop with `truncated=True` once
  `grep_walk_entry_limit` is reached,
- cap returned matches by `limit`.

- [ ] **Step 5: Run grep tests and verify GREEN**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q -k "filesystem_grep"
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
git commit -m "feat: add portable workspace grep"
```

## Follow-Up Task: Governed Shell Facade Aliases

**Status:** Not part of this implementation branch. Create a separate Backlog
task and branch after the native filesystem helpers are merged.

**Goal:** Let models use familiar shell-shaped commands while the backend still
routes every action through profile-granted MCP tools.

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/run_command_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/command_runtime/registry.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/command_runtime/adapters.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/README.md`

- [ ] Add a Backlog task such as `Add governed shell facade aliases`.
- [ ] Keep `run` as the canonical MCP tool.
- [ ] Optionally expose `bash` and/or `shell` as aliases that call the same
  `RunCommandModule` implementation. Their descriptions must say they are
  governed, shell-like facades and not host shell execution.
- [ ] Add registry aliases after backing tools exist:
  - `stat <path>` -> `fs.stat`
  - `glob <pattern> [base]` or `find <pattern> [base]` -> `fs.glob`
  - `rg <pattern> [base]` or `grep-files <pattern> [base]` -> `fs.grep`
- [ ] Leave existing pure `grep` behavior unchanged for pipelines like
  `cat app.log | grep ERROR`. Do not make hybrid stdin/file-backed `grep`
  until command preflight can safely distinguish transform usage from backend
  search usage.
- [ ] Make aliases visible only when their backing MCP tools are executable for
  the active profile.
- [ ] Add tests for policy-filtered visibility, `run --help`, alias help/error
  text, no raw shell delegation, and preservation of the existing presentation
  footer/spill behavior.

## Task 5: Profile Metadata And Docs

**Files:**
- Modify: `mcp_unified/profiles/presets.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`
- Modify: `mcp_unified/USER_GUIDE.md`

- [ ] **Step 1: Write failing preset metadata test**

Add a test:

```python
def test_filesystem_read_presets_include_helper_tools() -> None:
    expected = {"fs.stat", "fs.glob", "fs.grep"}
    for preset in list_builtin_presets():
        tooling = preset.profile.metadata.get("tooling")
        if not isinstance(tooling, dict):
            continue
        enabled_tools = set(tooling.get("enabled_tools") or [])
        if {"fs.list", "fs.read_text"} <= enabled_tools:
            assert expected <= enabled_tools
```

- [ ] **Step 2: Run metadata test and verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py::test_filesystem_read_presets_include_helper_tools -q
```

Expected: FAIL because helper tools are not in `_FILES_READ_TOOLS`.

- [ ] **Step 3: Update `_FILES_READ_TOOLS`**

In `mcp_unified/profiles/presets.py`, change:

```python
_FILES_READ_TOOLS = ["fs.list", "fs.read_text"]
```

to include:

```python
_FILES_READ_TOOLS = ["fs.list", "fs.read_text", "fs.stat", "fs.glob", "fs.grep"]
```

- [ ] **Step 4: Update package user guide**

In `mcp_unified/USER_GUIDE.md`, add a short note that filesystem-capable
profiles can inspect metadata, glob paths, and search UTF-8 text with
workspace-bounded, cross-platform helpers.

- [ ] **Step 5: Run preset/docs-adjacent tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add mcp_unified/profiles/presets.py mcp_unified/USER_GUIDE.md tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py
git commit -m "docs: expose filesystem helper tools to profiles"
```

## Task 6: Protocol, Security, And Final Verification

**Files:**
- Modify: implementation Backlog task only unless verification finds issues.

- [ ] **Step 1: Run focused filesystem tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q
```

Expected: PASS.

- [ ] **Step 2: Run profile/discovery regression tests**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run package boundary test**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q
```

Expected: PASS.

- [ ] **Step 4: Run Bandit on touched code**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py \
  mcp_unified/profiles/presets.py \
  -f json \
  -o /tmp/bandit_mcp_filesystem_helpers.json
```

Expected: exit 0 or only known non-touched baseline findings. Fix new findings.

- [ ] **Step 5: Run whitespace check**

```bash
git diff --check
```

Expected: no output, exit 0.

- [ ] **Step 6: Update Backlog task**

Record:

- RED/GREEN test notes,
- final verification command results,
- Bandit result path,
- platform-specific skips, if any,
- final summary and DOD.

- [ ] **Step 7: Final commit for task tracking if needed**

```bash
git add backlog/tasks/<task-file>.md
git commit -m "docs: finalize mcp filesystem helper task"
```
