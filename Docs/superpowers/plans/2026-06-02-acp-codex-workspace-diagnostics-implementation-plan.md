# Codex ACP Workspace Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Codex ACP sessions visibly and queryably workspace-aware without adding Codex app-server or generic runner-adapter support.

**Architecture:** Extend the existing ACP session record instead of adding a parallel workspace-agent history store. Persist sandbox session/run identifiers when the runner exposes them, derive a bounded `workspace_context` envelope from existing session, MCP, policy, sandbox, and agent-registry metadata, expose `workspace_id` filtering on ACP session list, and let Research Workspace history include direct ACP sessions alongside Agent Tasks runs.

**Tech Stack:** FastAPI, Pydantic, SQLite-backed `ACPSessionsDB`, `ACPSessionStore`, React, TypeScript, Vitest, pytest, Bandit.

**Status:** Complete. Focused backend and frontend tests pass; Bandit reports only existing `ACP_Sessions_DB.py` baseline findings outside the new `list_sessions` query path.

---

## File Structure

- Modify `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py`
  - Add durable sandbox session/run columns and `workspace_id` filtering for session list queries.
- Modify `tldw_Server_API/app/services/admin_acp_sessions_service.py`
  - Carry sandbox fields through `SessionRecord`, `register_session`, info/detail dicts, and workspace-filtered list calls.
- Modify `tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py`
  - Add an additive `ACPSessionWorkspaceContext` schema and expose it on session summary/detail.
- Modify `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
  - Build bounded workspace context from session record plus agent registry metadata, include it in detail/diagnostics, persist sandbox metadata on session creation, and accept `workspace_id` query filtering.
- Modify `apps/packages/ui/src/services/acp/types.ts`
  - Mirror the new session context and sandbox fields.
- Modify `apps/packages/ui/src/services/acp/client.ts`
  - Add `workspace_id` to `listSessions` params.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceACPHistoryModal.tsx`
  - Fetch direct ACP sessions for the current workspace and render them in the same modal with diagnostics/artifacts/audit links.
- Tests:
  - `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py`
  - `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_session_store.py`
  - `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py`
  - `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`
  - Add a focused `WorkspaceACPHistoryModal` test file only if the existing header test becomes too broad.

---

### Task 1: Backend Session Persistence And Workspace Filter

- [x] **Step 1: Write failing DB tests**

Add coverage in `test_acp_sessions_db.py`:

```python
def test_register_session_persists_sandbox_context_and_filters_by_workspace(db):
    db.register_session(
        session_id="s-workspace",
        user_id=1,
        agent_type="codex",
        workspace_id="workspace-1",
        sandbox_session_id="sandbox-session-1",
        sandbox_run_id="sandbox-run-1",
    )
    db.register_session(
        session_id="s-other",
        user_id=1,
        agent_type="codex",
        workspace_id="workspace-2",
    )

    row = db.get_session("s-workspace")
    assert row["sandbox_session_id"] == "sandbox-session-1"
    assert row["sandbox_run_id"] == "sandbox-run-1"

    rows, total = db.list_sessions(user_id=1, workspace_id="workspace-1")
    assert total == 1
    assert rows[0]["session_id"] == "s-workspace"
```

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py::TestSessionCRUD::test_register_session_persists_sandbox_context_and_filters_by_workspace -q
```

Expected: FAIL because `sandbox_session_id`, `sandbox_run_id`, and `workspace_id` filtering are not implemented.

- [x] **Step 2: Implement DB migration and filter**

In `ACP_Sessions_DB.py`:

- Increment `_SCHEMA_VERSION`.
- Add `sandbox_session_id TEXT` and `sandbox_run_id TEXT` to `sessions`.
- Add both columns to `_ALLOWED_MIGRATION_COLUMNS["sessions"]`.
- Accept `sandbox_session_id` and `sandbox_run_id` in `register_session`.
- Add optional `workspace_id` to `list_sessions` query construction.
- Keep existing user/status/agent filters unchanged.

- [x] **Step 3: Write failing service tests**

Extend `test_acp_session_store.py` so `ACPSessionStore.register_session(...)` preserves sandbox IDs and `list_sessions(..., workspace_id="workspace-1")` only returns workspace-bound sessions.

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_session_store.py -q
```

Expected: FAIL until service layer carries the new fields and filter.

- [x] **Step 4: Implement service passthrough**

In `admin_acp_sessions_service.py`:

- Add `sandbox_session_id` and `sandbox_run_id` to `SessionRecord`.
- Include them in `to_info_dict()` and `to_detail_dict()`.
- Accept/pass the fields in `register_session`.
- Accept/pass `workspace_id` in `list_sessions`.

- [x] **Step 5: Run backend persistence tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_session_store.py -q
```

Expected: PASS.

---

### Task 2: API Workspace Context Envelope

- [x] **Step 1: Write failing endpoint tests**

Add coverage in `test_acp_endpoints.py`:

- `POST /api/v1/acp/sessions/new` persists sandbox metadata returned by a runner `get_session_metadata`.
- `GET /api/v1/acp/sessions?workspace_id=workspace-1` only returns matching sessions.
- `GET /api/v1/acp/sessions/{session_id}/diagnostics` includes a bounded `workspace_context` with workspace identifiers, MCP server count/names, policy fingerprint, sandbox IDs, and Codex adapter metadata.
- The diagnostics payload does not include raw MCP env values.

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py -k "workspace_context or workspace_filter or sandbox_metadata" -q
```

Expected: FAIL until endpoint/schema context is implemented.

- [x] **Step 2: Implement schema and context helper**

In `agent_client_protocol.py` schemas:

- Add `ACPSessionWorkspaceContext`.
- Add `workspace_context: ACPSessionWorkspaceContext | None` to `ACPSessionInfo`.

In endpoint module:

- Add `_build_acp_workspace_context(rec)` helper.
- Include only bounded fields:
  - `workspace_id`, `workspace_group_id`, `scope_snapshot_id`
  - `mcp_server_count`, `mcp_server_names`
  - `sandbox_session_id`, `sandbox_run_id`
  - `policy_snapshot_fingerprint`, `policy_snapshot_version`, `policy_refresh_error`
  - `agent_type`, `runtime_backend`, `entrypoint_strategy`
  - `adapter_source`, `adapter_package`, `adapter_version`
  - `support_state`, `verification_level`
- Never include raw MCP env, command args, or full `cwd` in diagnostics context.

- [x] **Step 3: Wire session creation/list/detail/diagnostics**

- Persist `sandbox_session_id` and `sandbox_run_id` from `sandbox_meta`.
- Add `workspace_id` query parameter to `acp_list_sessions`.
- Include context in list/detail responses.
- Include context in diagnostics response.

- [x] **Step 4: Run endpoint tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py -q
```

Expected: PASS.

---

### Task 3: Research Workspace Direct ACP Session History

- [x] **Step 1: Write failing frontend test**

Extend `WorkspaceHeader.test.tsx` or create `WorkspaceACPHistoryModal.test.tsx` to verify:

- Opening ACP run history for `workspace-alpha` fetches `/api/v1/acp/sessions?workspace_id=workspace-alpha`.
- A direct Codex ACP session renders when Agent Tasks history is empty or unavailable.
- Direct session diagnostics/artifacts/audit buttons navigate to `/acp-playground?session=<id>&view=<view>`.

Run:

```bash
cd apps/packages/ui && ./node_modules/.bin/vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx
```

Expected: FAIL until the modal fetches/renders direct ACP sessions.

- [x] **Step 2: Update TS types and client**

- Add `workspace_context`, `sandbox_session_id`, and `sandbox_run_id` to ACP session list/detail types.
- Add `workspace_id` to `ACPRestClient.listSessions()` query params.

- [x] **Step 3: Render direct ACP sessions**

In `WorkspaceACPHistoryModal.tsx`:

- Fetch direct ACP sessions using the same connection/auth transport as existing modal requests.
- Do not block existing Agent Tasks history on direct-session fetch failure; surface a compact warning only if both sources fail.
- Render a direct-session section after Agent Tasks runs.
- Link direct sessions to ACP Playground diagnostics/artifacts/audit views using the same deep-link convention.

- [x] **Step 4: Run focused frontend tests**

```bash
cd apps/packages/ui && ./node_modules/.bin/vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx
```

Expected: PASS.

---

### Task 4: Verification And Closeout

- [x] **Step 1: Run focused backend tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_session_store.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py -q
```

- [x] **Step 2: Run focused frontend tests**

```bash
cd apps/packages/ui && ./node_modules/.bin/vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx
```

- [x] **Step 3: Run Bandit on touched Python**

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py \
  tldw_Server_API/app/services/admin_acp_sessions_service.py \
  tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py \
  tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py \
  -f json -o /tmp/bandit_task_606_acp_workspace_diagnostics.json
```

Expected: no new findings in touched diff. Record any known baseline findings explicitly.

- [x] **Step 4: Run diff hygiene**

```bash
git diff --check
```

- [x] **Step 5: Update Backlog closeout**

Verification:

- `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_session_store.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py -q` passed: 86 tests.
- `./node_modules/.bin/vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx` passed: 40 tests.
- `python -m bandit -r ... -f json -o /tmp/bandit_task_606_acp_workspace_diagnostics.json` reported existing `ACP_Sessions_DB.py` baseline findings only; the new workspace-filter SQL uses static optional filters and no longer appears in the Bandit report.
- `git diff --check` passed.

Record tests, Bandit result, known skips, and the remaining Stage 4/5 follow-ups: Codex app-server, generic runner adapter fallback, and live artifact/reviewer-loop certification beyond direct session diagnostics.
