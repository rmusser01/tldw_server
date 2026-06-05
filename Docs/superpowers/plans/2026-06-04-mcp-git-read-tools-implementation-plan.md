# MCP Git Read Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add active-workspace, read-only Git inspection tools to MCP Unified with profile grants and shared tool observability/evaluation metadata.

**Architecture:** Add a new optional `GitModule` that resolves the active workspace root through `McpHubWorkspaceRootResolver`, discovers the Git repo root inside that workspace, and executes only allowlisted read-only Git commands through an injected async runner. Add a small shared tool-evaluation metadata helper so Git tools adopt the MCP-wide observability/evaluation contract without retrofitting every existing tool in this PR. Keep command-runtime aliases, Git mutations, multi-repo selection, and the all-tool metadata migration as follow-up work.

**Tech Stack:** Python 3.11, FastAPI-side MCP module framework, `asyncio.create_subprocess_exec`, `pathlib`, pytest, Bandit, existing `mcp_unified.profiles` package metadata.

---

## File Map

- Create `tldw_Server_API/app/core/MCP_unified/tool_observability.py`
  - Shared helpers for tool definition eval metadata and non-sensitive execution eval metadata.
- Create `tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py`
  - `GitModule`, injected Git runner, repo/path resolution, argument validation, Git command construction, output parsing, and bounded responses.
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py`
  - Unit coverage for the shared metadata helpers.
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py`
  - Unit and temp-repo coverage for schemas, validation, repo discovery, read-only execution behavior, parsing, limits, and eval metadata.
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_git_module_registration.py`
  - Server default-module registration coverage for `MCP_ENABLE_GIT_MODULE`.
- Modify `tldw_Server_API/app/core/MCP_unified/server.py`
  - Optional registration for `GitModule` when `MCP_ENABLE_GIT_MODULE=true`.
- Modify `mcp_unified/profiles/presets.py`
  - Align `_GIT_READ_TOOLS` with the full native Git tool set and grant them to relevant presets.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`
  - Assert Git-capable presets include native Git read tools and non-Git presets do not accidentally gain them.
- Modify `tldw_Server_API/app/core/MCP_unified/README.md`
  - Document the optional Git module, enablement flag, safety limits, and eval metadata note.
- Modify `mcp_unified/USER_GUIDE.md`
  - Add packaged-user guidance for enabling Git read tools and where they appear in profiles.
- Modify Backlog task files:
  - `backlog/tasks/task-2257 - Plan-MCP-Git-read-only-inspection-tools-implementation.md`
  - Later implementation task created before code edits.

## Task 0: Baseline, Backlog, And Implementation Task

**Files:**
- Modify: Backlog implementation task only.

- [ ] **Step 1: Create or locate the implementation Backlog task**

Search first:

```bash
backlog search "Implement MCP Git read-only inspection tools" --plain
```

Expected: no duplicate active implementation task. If none exists, create `Implement MCP Git read-only inspection tools`.

- [ ] **Step 2: Record branch baseline**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Expected: clean branch based on latest `origin/dev`.

- [ ] **Step 3: Read required context**

Run:

```bash
sed -n '1,260p' Docs/superpowers/specs/2026-06-04-mcp-git-read-tools-design.md
sed -n '520,650p' tldw_Server_API/app/core/MCP_unified/server.py
sed -n '1,260p' tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py
sed -n '1,260p' tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
sed -n '1,260p' tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py
```

Expected: confirm active workspace root resolver patterns, module registration style, and preset tests.

## Task 1: Shared Tool Observability Helper

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tool_observability.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py`

- [ ] **Step 1: Write failing tests for tool eval metadata**

Create `test_tool_observability.py` with tests like:

```python
from tldw_Server_API.app.core.MCP_unified.tool_observability import (
    build_execution_eval_metadata,
    build_tool_eval_metadata,
)


def test_build_tool_eval_metadata_is_stable_and_non_empty() -> None:
    metadata = build_tool_eval_metadata(
        tool_prompt_id="mcp.git.status.v1",
        tool_prompt_version="2026.06.04",
        task_families=["code_review"],
        expected_result_kind="structured_git_state",
        success_signals=["avoided_mutation"],
    )

    assert metadata == {
        "eval": {
            "tool_prompt_id": "mcp.git.status.v1",
            "tool_prompt_version": "2026.06.04",
            "task_families": ["code_review"],
            "expected_result_kind": "structured_git_state",
            "success_signals": ["avoided_mutation"],
            "prompt_variant": "builtin",
        }
    }
```

Add a second test for `build_execution_eval_metadata()` that asserts unknown raw payload text, absolute paths, and author emails are not accepted as labels.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py -q
```

Expected: FAIL because `tool_observability.py` does not exist.

- [ ] **Step 3: Implement the helper**

Create helper functions:

```python
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _clean_list(values: Iterable[str]) -> list[str]:
    return [str(value).strip() for value in values if str(value).strip()]


def build_tool_eval_metadata(
    *,
    tool_prompt_id: str,
    tool_prompt_version: str,
    task_families: Iterable[str],
    expected_result_kind: str,
    success_signals: Iterable[str],
    prompt_variant: str = "builtin",
) -> dict[str, dict[str, Any]]:
    return {
        "eval": {
            "tool_prompt_id": str(tool_prompt_id).strip(),
            "tool_prompt_version": str(tool_prompt_version).strip(),
            "task_families": _clean_list(task_families),
            "expected_result_kind": str(expected_result_kind).strip(),
            "success_signals": _clean_list(success_signals),
            "prompt_variant": str(prompt_variant).strip() or "builtin",
        }
    }
```

For execution metadata, only include scalar, non-sensitive fields:

```python
def build_execution_eval_metadata(
    *,
    tool_name: str,
    tool_prompt_id: str,
    tool_prompt_version: str,
    action_family: str,
    result_kind: str,
    profile_id: str | None = None,
    path_filter_used: bool | None = None,
    truncated: bool = False,
    reason_code: str | None = None,
    duration_ms: float | None = None,
) -> dict[str, Any]:
    ...
```

- [ ] **Step 4: Run helper tests and verify GREEN**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add tldw_Server_API/app/core/MCP_unified/tool_observability.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py
git commit -m "feat: add mcp tool eval metadata helpers"
```

## Task 2: Git Module Schemas And Validation

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py`
- Create/modify: `tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py`

- [ ] **Step 1: Write failing schema and validation tests**

Add tests for:

- exactly seven tools: `git.status`, `git.diff`, `git.log`, `git.blame`, `git.branches`, `git.conflicts.list`, `git.conflicts.read`;
- `additionalProperties` is `False`;
- metadata includes `category == "git"`, `readOnlyHint is True`, `uses_processes is True`, `uses_filesystem is True`, `path_boundable is True`, and `eval`;
- unknown args are rejected;
- absolute paths are rejected;
- path values that normalize outside the workspace are rejected;
- limits, byte caps, context lines, and line ranges reject booleans, zero, negative, and over-maximum values.

- [ ] **Step 2: Run schema tests and verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "schema or validates"
```

Expected: FAIL because `git_module.py` does not exist.

- [ ] **Step 3: Implement minimal `GitModule` tool definitions**

Create:

```python
class GitModule(BaseModule):
    """Read-only Git inspection tools for the active workspace repository."""

    def __init__(
        self,
        config: ModuleConfig,
        *,
        workspace_root_resolver: McpHubWorkspaceRootResolver | Any | None = None,
        runner: GitCommandRunner | None = None,
    ) -> None:
        super().__init__(config)
        self._workspace_root_resolver = workspace_root_resolver or McpHubWorkspaceRootResolver()
        self._runner = runner or AsyncGitCommandRunner()
```

Use `create_tool_definition()` for every tool and merge `build_tool_eval_metadata(...)` into metadata.

- [ ] **Step 4: Implement strict argument validation**

Add helper validation methods:

- `_reject_unknown(args, allowed)`
- `_positive_int(args, name, default, maximum)`
- `_optional_bool(args, name)` only if a boolean is truly needed;
- `_validate_relative_path(path)`;
- `_validate_line_range(start_line, end_line, limit)`.

Do not expose `include_ignored`. First-slice `git.status` excludes ignored files.

- [ ] **Step 5: Run Task 2 tests and verify GREEN**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "schema or validates"
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py
git commit -m "feat: add mcp git tool schemas"
```

## Task 3: Git Runner And Repository Resolution

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py`

- [ ] **Step 1: Write failing runner and repo-resolution tests**

Add tests that cover:

- runner uses `asyncio.create_subprocess_exec`, not shell;
- command environment disables prompts, pagers, optional locks, and external diff;
- missing Git binary becomes `reason_code == "git_not_available"`;
- non-Git workspace becomes `reason_code == "not_git_repository"`;
- Git root outside workspace becomes `reason_code == "repo_outside_workspace"`;
- command timeout becomes `reason_code == "git_command_timeout"`.

Use fake runners for most tests. Use a temp repo helper only where real Git behavior is necessary:

```python
import shutil
import subprocess


def _init_repo(path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git binary unavailable")
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
```

Skip only real-Git temp-repo tests when `git` is unavailable. Fake-runner validation tests must still run without a local Git binary.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "runner or repository or timeout"
```

Expected: FAIL until runner and repo resolution exist.

- [ ] **Step 3: Implement runner dataclass and async runner**

Add:

```python
@dataclass(frozen=True, slots=True)
class GitCommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
```

Production runner requirements:

- command starts with `git`;
- use `asyncio.create_subprocess_exec(*argv, ...)`; add `# nosec B603` only if Bandit flags the controlled argv subprocess call, with a comment explaining fixed argv/no shell;
- set `GIT_TERMINAL_PROMPT=0`, `GIT_OPTIONAL_LOCKS=0`, `GIT_PAGER=cat`, `GIT_EXTERNAL_DIFF=`;
- wrap with `asyncio.wait_for`.

- [ ] **Step 4: Implement active repo resolution**

Add `_resolve_workspace_root(context)` using the same context metadata approach as `FilesystemModule`. Add `_resolve_git_root(workspace_root)` that runs:

```bash
git -C <workspace_root> rev-parse --show-toplevel
```

Normalize and require Git root containment under workspace root.

- [ ] **Step 5: Run Task 3 tests and verify GREEN**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "runner or repository or timeout"
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py
git commit -m "feat: add bounded mcp git runner"
```

## Task 4: Status, Branches, And Conflict List

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py`

- [ ] **Step 1: Write failing behavior tests**

Add temp-repo and fake-output tests for:

- `git.status` parses `status --porcelain=v2 -z --branch` entries;
- staged, unstaged, untracked, and conflicted counts;
- ignored files do not appear;
- `git.branches` returns current branch and bounded branch list;
- `git.conflicts.list` maps `u` porcelain records or `ls-files -u -z` output to conflicted paths;
- `limit` truncates entries and marks `truncated: true`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "status or branches or conflicts_list"
```

Expected: FAIL until behavior exists.

- [ ] **Step 3: Implement commands and parsers**

Use commands:

```text
git --no-pager -C <repo> status --porcelain=v2 -z --branch --untracked-files=all
git --no-pager -C <repo> branch --format=%(HEAD)%00%(refname:short)%00%(upstream:short)%00%(objectname)
git --no-pager -C <repo> ls-files -u -z
```

Do not include ignored files. If status parsing sees ignored entries anyway, skip them.

- [ ] **Step 4: Add eval execution metadata to responses**

Each response should include an `eval` object from `build_execution_eval_metadata(...)` with no raw paths beyond relative path flags, no author emails, and no diff/file contents.

- [ ] **Step 5: Run Task 4 tests and verify GREEN**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "status or branches or conflicts_list"
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py
git commit -m "feat: add mcp git status tools"
```

## Task 5: Diff, Log, Blame, And Conflict Read

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py`

- [ ] **Step 1: Write failing diff tests**

Cover:

- `scope="unstaged"` uses `diff --no-ext-diff --no-textconv --no-color`;
- `scope="staged"` adds `--cached`;
- `scope="working_tree"` compares staged plus unstaged changes with `HEAD`;
- path filters use `--` before pathspec;
- paths beginning with `-` are treated as paths, not options;
- `max_bytes` truncates text and sets `truncated: true`.

- [ ] **Step 2: Write failing log, blame, and conflict-read tests**

Cover:

- `git.log` bounded commits and path filtering;
- `git.log` and `git.blame` do not return author emails;
- `git.blame` validates line ranges and caps line count;
- `git.conflicts.read` only accepts conflicted files;
- conflict hunk output is bounded by both file bytes and hunk count.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "diff or log or blame or conflicts_read"
```

Expected: FAIL until behavior exists.

- [ ] **Step 4: Implement diff/log/blame/conflict-read**

Use allowlisted argv builders only. All path-bearing commands must append `--` before the relative pathspec.

Suggested commands:

```text
git --no-pager -C <repo> diff --no-ext-diff --no-textconv --no-color --unified=<n> [--cached] [-- <path>]
git --no-pager -C <repo> log --format=%H%x1f%h%x1f%an%x1f%aI%x1f%s%x1e -n <limit> [-- <path>]
git --no-pager -C <repo> blame --line-porcelain -L <start>,<end> -- <path>
```

Use ASCII unit separator (`\x1f`) between `git.log` fields and record separator (`\x1e`) between commits. Do not use one delimiter for both fields and records.

For `working_tree`, prefer a command that captures full working tree change against `HEAD` without external helpers. If separate staged and unstaged diffs are simpler and clearer, return `sections` with `staged` and `unstaged` while preserving the public `scope`.

- [ ] **Step 5: Run Task 5 tests and verify GREEN**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "diff or log or blame or conflicts_read"
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py
git commit -m "feat: add mcp git diff history tools"
```

## Task 6: Server Registration And Profile Grants

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_git_module_registration.py`
- Modify: `mcp_unified/profiles/presets.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`

- [ ] **Step 1: Write failing registration tests**

Create tests modeled after `test_browser_cdp_server_registration.py`:

- `MCP_ENABLE_GIT_MODULE=1` registers `GitModule`;
- disabled or unset flag does not register it;
- filesystem/media/browser env vars are isolated from Git registration.

The capture helper should set `MCP_MODULES_CONFIG` to a missing temp path, clear `MCP_MODULES`, set `MCP_ENABLE_MEDIA_MODULE=0`, set `MCP_ENABLE_FILESYSTEM_MODULE=0`, set `MCP_ENABLE_BROWSER_CDP_MODULE=0`, and clear `MCP_BROWSER_CDP_URL` so Git registration is the only variable under test.

- [ ] **Step 2: Run registration tests and verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module_registration.py -q
```

Expected: FAIL until server registration exists.

- [ ] **Step 3: Implement optional server registration**

In `_register_default_modules()`, add after filesystem and before sandbox:

```python
if self._env_flag_enabled("MCP_ENABLE_GIT_MODULE"):
    modules_to_load.append({
        "id": "git",
        "class": "tldw_Server_API.app.core.MCP_unified.modules.implementations.git_module:GitModule",
        "enabled": True,
        "name": "Git",
        "version": "1.0.0",
        "department": "management",
        "settings": {},
    })
```

- [ ] **Step 4: Write failing profile preset tests**

Add tests:

```python
def test_git_capable_presets_include_native_git_read_tools() -> None:
    git_tools = {
        "git.status",
        "git.diff",
        "git.log",
        "git.blame",
        "git.branches",
        "git.conflicts.list",
        "git.conflicts.read",
    }
    for preset_id in (
        "architect",
        "merge-conflict-resolver",
        "code-reviewer",
        "devops-engineer",
        "backend-engineer",
        "frontend-engineer",
        "qa-engineer",
        "sdet",
    ):
        preset = presets.get_builtin_preset(preset_id)
        assert preset is not None
        tooling = preset.profile.metadata["tooling"]
        assert git_tools <= set(tooling["enabled_tools"])
        assert "git.read" in set(tooling["enabled_capabilities"])
```

Also assert Product Owner and Documentation Writer do not gain Git tools by default.

- [ ] **Step 5: Update preset metadata**

Update `_GIT_READ_TOOLS` to all seven tools. Add `git.read` to relevant `enabled_capabilities` and direct categories where appropriate. Keep Product Owner and Documentation Writer unchanged.

- [ ] **Step 6: Run Task 6 tests and verify GREEN**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module_registration.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 6**

```bash
git add tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/core/MCP_unified/tests/test_git_module_registration.py mcp_unified/profiles/presets.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py
git commit -m "feat: register mcp git read tools"
```

## Task 7: Documentation And Packaged User Guide

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/README.md`
- Modify: `mcp_unified/USER_GUIDE.md`

- [ ] **Step 1: Update MCP README**

Add:

- `MCP_ENABLE_GIT_MODULE=true` enablement;
- list of Git tools;
- read-only guarantees;
- active workspace repo root only;
- no ignored files, no author emails, no external diff/textconv;
- eval metadata note and follow-up `TASK-2256` for all-tool adoption.

- [ ] **Step 2: Update packaged user guide**

Add a concise "Git inspection tools" subsection near profile/tool discovery guidance. Explain which profiles receive Git tools and that the module is optional/configured by operators.

- [ ] **Step 3: Run docs diff check**

Run:

```bash
git diff --check
```

Expected: PASS.

- [ ] **Step 4: Commit Task 7**

```bash
git add tldw_Server_API/app/core/MCP_unified/README.md mcp_unified/USER_GUIDE.md
git commit -m "docs: document mcp git read tools"
```

## Task 8: Final Verification And Backlog Closeout

**Files:**
- Modify: Backlog implementation task.

- [ ] **Step 1: Run focused tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_git_module_registration.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run adjacent MCP regression tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_server_registration.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -q
```

Expected: PASS or document pre-existing skips.

- [ ] **Step 3: Run Bandit on touched source**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified/tool_observability.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py \
  tldw_Server_API/app/core/MCP_unified/server.py \
  mcp_unified/profiles/presets.py \
  -f json -o /tmp/bandit_mcp_git_read_tools.json
```

Expected: 0 findings. If Bandit flags controlled Git subprocess execution, either fix the issue or add the narrowest possible `# nosec B603` with a comment explaining fixed argv, no shell, no user-controlled executable, and allowlisted subcommands.

- [ ] **Step 4: Run final diff checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only expected Backlog task file may remain modified before final commit.

- [ ] **Step 5: Update Backlog task**

Record:

- touched files;
- tests run and results;
- Bandit result path;
- known skips/blockers;
- final summary.

- [ ] **Step 6: Final commit**

```bash
git add backlog/tasks/<implementation-task>.md
git commit -m "docs: record mcp git read tools verification"
```

Expected: branch clean after commit.
