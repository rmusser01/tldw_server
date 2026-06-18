# Workspace Membership Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tested Workspace membership adapters for prompt, workflow, watchlist, ACP session, and sandbox session resources while explicitly deferring global note and ACP run.

**Architecture:** Extend the existing registry-based membership service. Add optional domain DB handles to `WorkspaceMembershipContext` and endpoint dependencies, keep adapter validation domain-owned, and use Workspace runtime binding descriptors for ACP/Sandbox session membership summaries.

**Tech Stack:** Python, FastAPI dependencies, Pydantic schemas, pytest, Loguru, SQLite/Postgres-safe DB helper methods, Bandit.

---

### Task 1: Resource Registry And Context Contract

**Files:**
- Modify: `tldw_Server_API/app/core/Workspaces/membership_models.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Modify: `tldw_Server_API/app/core/Workspaces/membership_adapters.py`
- Modify: `tldw_Server_API/app/core/Workspaces/membership_service.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py`

- [x] **Step 1: Write failing registry/context tests**

Add tests that expect `prompt`, `workflow`, `watchlist`, `acp_session`, and `sandbox_session` to be supported resource types, while `note` and `acp_run` still fail closed.

- [x] **Step 2: Run test to verify it fails**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q`

Expected: FAIL because new resource types are unsupported.

- [x] **Step 3: Implement minimal registry/context plumbing**

Extend resource-type literals and `WorkspaceMembershipContext` with optional `prompts_db`, `workflows_db`, `watchlists_db`, and request metadata fields used for workflow tenant/admin checks.

- [x] **Step 4: Run test to verify it passes**

Run the same focused pytest command. Expected: PASS for registry/context tests, or progress to adapter-specific missing method failures in later tests.

### Task 2: Prompt, Workflow, And Watchlist Adapters

**Files:**
- Modify: `tldw_Server_API/app/core/Workspaces/membership_adapters.py`
- Modify: `tldw_Server_API/app/core/Workspaces/membership_service.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py`

- [x] **Step 1: Write failing adapter tests**

Add tests for prompt canonical ID resolution, prompt DB unavailable, workflow tenant/owner denial, workflow admin allowance, watchlist active lookup, and deleted summaries.

- [x] **Step 2: Run tests to verify failure**

Run focused adapter tests and confirm they fail because adapter classes/logic are missing.

- [x] **Step 3: Implement minimal adapters**

Add `PromptMembershipAdapter`, `WorkflowMembershipAdapter`, and `WatchlistMembershipAdapter`. Summaries must avoid prompt text, workflow definitions, and watchlist objective/body content.

- [x] **Step 4: Run focused tests**

Run `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q`.

### Task 3: ACP And Sandbox Session Adapters

**Files:**
- Modify: `tldw_Server_API/app/core/Workspaces/membership_adapters.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py`

- [x] **Step 1: Write failing runtime binding adapter tests**

Add tests for active ACP session and Sandbox session runtime binding validation, archived binding summary state, wrong binding kind/domain denial, and missing binding denial.

- [x] **Step 2: Run tests to verify failure**

Expected: FAIL because runtime binding membership adapters are missing.

- [x] **Step 3: Implement runtime binding session adapters**

Use `context.chacha_db.get_workspace_runtime_binding(workspace_id, binding_id, include_deleted=...)` and validate expected kind/domain. Summary metadata must use already-redacted descriptor fields.

- [x] **Step 4: Run focused tests**

Run the Workspace membership adapter tests and confirm green.

### Task 4: API Dependency Wiring

**Files:**
- Modify: `tldw_Server_API/app/api/v1/API_Deps/Prompts_DB_Deps.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/Watchlists_DB_Deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspace_memberships.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py`

- [x] **Step 1: Write failing API tests**

Add tests proving membership create/list paths pass optional prompt/workflow/watchlist handles and workflow request metadata, and that unrelated resource calls still work when optional dependencies are unavailable.

- [x] **Step 2: Run API tests to verify failure**

Run: `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py -q`

- [x] **Step 3: Implement dependency wiring**

Add optional prompt/watchlist dependency helpers and a workflow DB optional helper local to Workspace endpoints if no shared helper exists. Pass handles and workflow metadata into membership service calls.

- [x] **Step 4: Run API tests**

Run membership API tests and adapter tests.

### Task 5: Docs, Backlog, And Verification

**Files:**
- Modify: `tldw_Server_API/app/core/Workspaces/README.md`
- Modify: `backlog/tasks/task-2383 - Add-remaining-Workspace-membership-adapters-for-ACP-adjacent-resource-domains.md`

- [x] **Step 1: Update docs**

Document supported/deferred resource types, runtime binding session semantics, and the invariant that membership is not a trust source.

- [x] **Step 2: Run focused verification**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py tldw_Server_API/tests/Workspaces/test_workspace_context_membership_summary.py -q
python -m bandit -r tldw_Server_API/app/core/Workspaces tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/api/v1/endpoints/workspace_memberships.py tldw_Server_API/app/api/v1/API_Deps/Prompts_DB_Deps.py tldw_Server_API/app/api/v1/API_Deps/Watchlists_DB_Deps.py -f json -o /tmp/bandit_workspace_membership_adapters.json
```

- [x] **Step 3: Finalize Backlog task**

Record touched files, verification results, known deferrals, and final summary.
