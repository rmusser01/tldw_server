# MCP File Policy Action Taxonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Expand MCP file-policy actions beyond the first read/edit/write slice while keeping destructive file operations reserved until dedicated tools land.

**Architecture:** Add a package-owned file-policy action taxonomy and route existing path-grant, path-scope, preview, and filesystem descriptor code through it. This keeps the executable behavior unchanged for `fs.read`, `fs.patch`, and `fs.write`, but lets policy authors and preview tooling express future delete, move, share/export, admin, and lock permissions without collapsing them into generic write authority.

**Tech Stack:** Python 3.11, Pydantic-compatible package models, pytest, existing MCP Unified filesystem/path-enforcement services.

---

### Task 1: Add Shared File Policy Action Metadata

**Files:**
- Create: `mcp_unified/interfaces/file_policy_actions.py`
- Modify: `mcp_unified/interfaces/__init__.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_file_policy_actions.py`

- [x] **Step 1: Write failing action metadata tests**

Add tests that assert:
- `FILE_POLICY_ACTIONS` equals `read`, `edit`, `write`, `delete`, `rename`, `move`, `share`, `export`, `chmod`, `admin`, and `lock`.
- `FILE_POLICY_EXISTING_TOOL_ACTIONS` equals `read`, `edit`, `write`.
- `FILE_POLICY_EXFILTRATION_ACTIONS` includes `share` and `export`.
- `get_file_policy_action_metadata("share")` returns a redacted/operator-safe metadata payload with family `exfiltration` and `implemented` set to false.
- Unknown action lookup fails clearly without falling back to write.

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_file_policy_actions.py -q
```

Expected: fail because the module does not exist yet.

- [x] **Step 2: Implement minimal taxonomy module**

Create a frozen dataclass for action metadata plus constants and helpers:
- `FilePolicyAction`
- `FilePolicyActionMetadata`
- `FILE_POLICY_ACTIONS`
- `FILE_POLICY_EXISTING_TOOL_ACTIONS`
- `FILE_POLICY_DESTRUCTIVE_ACTIONS`
- `FILE_POLICY_EXFILTRATION_ACTIONS`
- `FILE_POLICY_ADMIN_ACTIONS`
- `FILE_POLICY_LOCK_ACTIONS`
- `normalize_file_policy_action()`
- `get_file_policy_action_metadata()`
- `format_file_policy_action_list()`

Do not add executable tools for reserved actions.

- [x] **Step 3: Verify green**

Run the focused metadata test command and confirm it passes.

### Task 2: Wire Taxonomy Into Path Grants And Path Scope Candidates

**Files:**
- Modify: `mcp_unified/profiles/path_grants.py`
- Modify: `mcp_unified/interfaces/path_scope.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_path_grant_authoring.py`

- [x] **Step 1: Write failing validation tests**

Add tests proving:
- Flat and authored path grants accept `delete`, `rename`, `move`, `share`, `export`, `chmod`, `admin`, and `lock`.
- Invalid action diagnostics list the expanded valid action set.
- `normalize_path_scope_candidate()` accepts a reserved action such as `lock`.

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_path_grant_authoring.py -q
```

Expected: fail because reserved actions are currently invalid.

- [x] **Step 2: Replace local action sets with shared constants**

Import the shared constants from `mcp_unified.interfaces.file_policy_actions`. Keep existing effect semantics: only `allow` and `deny`.

- [x] **Step 3: Verify green**

Run the same test file and confirm it passes.

### Task 3: Wire Taxonomy Into Enforcement Preview And Filesystem Metadata

**Files:**
- Modify: `tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Test: `tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`

- [x] **Step 1: Write failing enforcement and descriptor tests**

Add tests proving:
- `preview_effective_path_permission()` can explain an allowed `share` action when a path grant explicitly allows `share`.
- A reserved action without a matching grant is denied as `path_action_not_granted`, not `invalid_path_action`.
- Filesystem tool descriptors include action-family metadata for `fs.read`, `fs.patch`, and `fs.write`.

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py::test_filesystem_tools_include_path_scope_metadata -q
```

Expected: fail because the enforcer still rejects reserved actions and descriptors lack action-family metadata.

- [x] **Step 2: Import taxonomy in the enforcer**

Replace `_PATH_GRANT_ACTIONS` with the shared action set. Keep path-boundable behavior, path normalization, deny precedence, and allowlist fallback unchanged.

- [x] **Step 3: Add non-behavioral filesystem descriptor metadata**

Add metadata keys such as:
- `file_policy_action`: `read`, `edit`, or `write`
- `file_policy_action_family`: `read`, `bounded_edit`, or `whole_write`

Do not add new executable filesystem operation tools in this slice.

- [x] **Step 4: Verify green**

Run the enforcement and descriptor tests and confirm they pass.

### Task 4: Document And Validate

**Files:**
- Modify: `mcp_unified/USER_GUIDE.md`
- Modify: `backlog/tasks/task-2305 - Expand-MCP-file-policy-action-taxonomy-and-operation-tools.md`

- [x] **Step 1: Update user guide**

Add a concise section after the safe file tools path grants explaining:
- Current executable actions: `read`, `edit`, `write`.
- Reserved actions: `delete`, `rename`, `move`, `share`, `export`, `chmod`, `admin`, `lock`.
- `share` and `export` are exfiltration-sensitive and must not be treated as write.
- Reserved actions can be authored and previewed in policy now, but require future dedicated tools before execution.

- [x] **Step 2: Run focused and security validation**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_file_policy_actions.py tldw_Server_API/app/core/MCP_unified/tests/test_path_grant_authoring.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py::test_filesystem_tools_include_path_scope_metadata -q
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check mcp_unified/interfaces/file_policy_actions.py mcp_unified/interfaces/path_scope.py mcp_unified/profiles/path_grants.py tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_file_policy_actions.py tldw_Server_API/app/core/MCP_unified/tests/test_path_grant_authoring.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/interfaces/file_policy_actions.py mcp_unified/interfaces/path_scope.py mcp_unified/profiles/path_grants.py tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py -f json -o /tmp/bandit_mcp_file_policy_action_taxonomy.json
git diff --check
```

- [x] **Step 3: Finalize Backlog task**

Record touched files, verification results, final summary, and mark the Definition of Done complete.
