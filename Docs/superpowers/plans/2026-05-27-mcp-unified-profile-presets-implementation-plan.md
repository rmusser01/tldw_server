# MCP Unified Profile Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add package-local MCP profile preset primitives that model the initial front-end modes without changing current `tldw_server` MCP route behavior.

**Architecture:** This slice keeps all new logic inside the standalone `mcp_unified` package. Built-in presets are immutable templates that can be listed, looked up, validated against the spec safety baseline, and duplicated into editable `MCPProfile` instances with preset provenance. No protocol dispatch, host adapter, SQLite store, FastAPI router, gateway entrypoint, or tool execution behavior changes in this slice.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest, Ruff, Mypy, Bandit, Backlog.md.

---

## Source Spec

- Spec: `Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md`
- Previous package-boundary task: `TASK-518`
- Backlog task: `TASK-519`

## Scope Boundary

In scope:

- Add `mcp_unified.profiles.presets` with immutable preset definitions.
- Include the spec's initial role/mode preset ids.
- Add safety validation helpers that fail closed on unsafe default grants.
- Add duplication helper to turn a preset into an editable `MCPProfile`.
- Export preset primitives from `mcp_unified.profiles`.
- Add package-boundary tests for preset import isolation, lookup, duplication, and safety validation.
- Update task bookkeeping and verification evidence.

Out of scope:

- Profile persistence, SQLite stores, migrations, or assignments.
- Policy enforcement inside `MCPProtocol`.
- Host MCP Hub, AuthNZ, path scope, approval, credential, or route behavior changes.
- Gateway entrypoints or stdio transports.
- Tool catalog integration or execution filtering.

## File Structure

Create:

- `mcp_unified/profiles/presets.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`

Modify:

- `mcp_unified/profiles/__init__.py`
- `backlog/tasks/task-519 - Implement-MCP-Unified-Stage-2-profile-preset-primitives.md`

## Task 1: Add Failing Preset Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`

- [x] **Step 1: Write package-boundary preset tests**

Add tests for:

- `mcp_unified.profiles.presets` imports without `tldw_Server_API`.
- `list_builtin_presets()` returns the spec's initial preset ids.
- `get_builtin_preset("architect")` returns a stable preset with version.
- `duplicate_builtin_preset("architect")` returns an `MCPProfile` with `preset_id`, `preset_version`, and provenance.
- `validate_preset_safety()` catches unsafe grants such as process execution without approval provenance.

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -v
```

Expected: FAIL because `mcp_unified.profiles.presets` does not exist.

## Task 2: Implement Preset Primitives

**Files:**
- Create: `mcp_unified/profiles/presets.py`
- Modify: `mcp_unified/profiles/__init__.py`

- [x] **Step 1: Add immutable preset model**

Create `ProfilePreset` as a frozen Pydantic model with:

- `id`
- `version`
- `profile`

The embedded profile should use `MCPProfile` and preserve extension metadata.

- [x] **Step 2: Add bundled preset definitions**

Add presets with these ids:

- `orchestrator`
- `product-owner`
- `architect`
- `merge-conflict-resolver`
- `documentation-writer`
- `project-researcher`
- `deep-researcher`
- `code-reviewer`
- `devops-engineer`
- `backend-engineer`
- `frontend-engineer`
- `qa-engineer`
- `sdet`
- `memory-keeper`

Use conservative defaults:

- no credential grants unless explicitly justified
- no destructive filesystem capability by default
- no process execution capability by default
- external network grants only for `deep-researcher`, with provenance
- write-oriented presets use approval policy and scoped capabilities instead of broad shell/process access

- [x] **Step 3: Add public helpers**

Implement:

- `list_builtin_presets() -> tuple[ProfilePreset, ...]`
- `get_builtin_preset(preset_id: str) -> ProfilePreset | None`
- `duplicate_builtin_preset(preset_id: str, *, profile_id: str | None = None, name: str | None = None) -> MCPProfile`
- `validate_preset_safety(preset: ProfilePreset) -> list[str]`

- [x] **Step 4: Export from `profiles.__init__`**

Export `ProfilePreset`, `duplicate_builtin_preset`, `get_builtin_preset`, `list_builtin_presets`, and `validate_preset_safety`.

- [x] **Step 5: Run preset tests**

Run:

```bash
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -v
```

Expected: PASS.

## Task 3: Verify Package Boundary And Quality Gates

**Files:**
- Modify: `backlog/tasks/task-519 - Implement-MCP-Unified-Stage-2-profile-preset-primitives.md`

- [x] **Step 1: Run focused MCP package tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -v
```

Expected: PASS.

- [x] **Step 2: Run package quality checks**

Run:

```bash
python -m ruff check mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py
python -m mypy mcp_unified --config-file pyproject.toml
python -m bandit -r mcp_unified -f json -o /tmp/bandit_mcp_unified_profile_presets.json
git diff --check
```

Expected:

- Ruff passes.
- Mypy passes for `mcp_unified`.
- Bandit reports 0 findings.
- Diff whitespace check is clean.

- [x] **Step 3: Update Backlog task**

Record implementation summary, touched files, verification commands/results, and known skips/blockers.

- [x] **Step 4: Commit**

Run:

```bash
git add \
  mcp_unified/profiles/__init__.py \
  mcp_unified/profiles/presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  Docs/superpowers/plans/2026-05-27-mcp-unified-profile-presets-implementation-plan.md \
  "backlog/tasks/task-519 - Implement-MCP-Unified-Stage-2-profile-preset-primitives.md"
git commit -m "feat: add mcp unified profile presets"
```

## Final Verification Before PR

Run:

```bash
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -v
python -m ruff check mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py
python -m mypy mcp_unified --config-file pyproject.toml
python -m bandit -r mcp_unified -f json -o /tmp/bandit_mcp_unified_profile_presets.json
git diff --check
```

Expected:

- Preset tests pass.
- Runtime package-boundary tests still pass.
- Profile preset package stays free of `tldw_Server_API` imports.
- Quality checks pass with 0 new security findings.
