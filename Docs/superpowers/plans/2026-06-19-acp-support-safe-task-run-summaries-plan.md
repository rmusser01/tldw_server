# ACP Support-Safe Task Run Summaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement GitHub issue #2408 by adding a redacted Agent Tasks run-summary mode that preserves operational metadata while preventing prompt/result preview leakage.

**Architecture:** Keep the existing task detail endpoint and full-fidelity default behavior. Add a `run_summary_mode=full|redacted` query parameter, build the existing enriched run dictionaries, then apply a response-only redaction projection for redacted mode. Update the WebUI Agent Tasks URL-building contract and ACP docs without introducing new storage or broad UI behavior.

**Tech Stack:** FastAPI, Pydantic, pytest, React/TypeScript, Vitest, existing ACP redaction sentinel `[redacted]`.

---

## File Structure

- Modify `tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py`
  - Add run-summary mode constants/types.
  - Pass the selected mode through `get_task()` and `_enrich_task_runs()`.
  - Add helpers for redacted session links, redacted history, redacted diagnostics, and redacted run free-text fields.
- Modify `tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py`
  - Add backend contract coverage for redacted mode and default-mode preservation.
- Modify `apps/packages/ui/src/components/Option/AgentTasks/index.tsx`
  - Centralize task detail URL construction and support `run_summary_mode=redacted` for callers.
- Modify `apps/packages/ui/src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx`
  - Cover default task detail URL behavior and redacted URL-builder behavior.
- Modify ACP docs:
  - `Docs/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md`
  - `Docs/Development/Agent_Client_Protocol.md`
  - `Docs/Development/ACP_Production_Readiness.md`
- Update `backlog/tasks/task-2392 - Implement-ACP-support-safe-Agent-Tasks-run-summaries.md`
  - Record implementation notes, verification, and final summary.

## Stage 1: Backend Contract Tests

**Goal:** Lock the task-detail redaction behavior before implementation.

**Success Criteria:** Tests fail on the current branch because `run_summary_mode` is not implemented.

**Tests:** `pytest` targeted Agent Orchestration API tests.

**Status:** Not Started

### Task 1: Add failing backend redacted-mode test

**Files:**
- Modify: `tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py`

- [ ] **Step 1: Write the failing test**

Add `test_get_task_run_history_redacted_mode_omits_support_unsafe_text`. Build a task/run/session fixture like `test_get_task_run_history_includes_acp_session_drillthrough`, but include distinct secret-like strings in:

- user prompt content;
- assistant result content;
- run `result_summary`;
- run `error` or failure context;
- reviewer feedback;
- diagnostic message and URI.

Call:

```python
detail = await orch_mod.get_task(
    task.id,
    run_summary_mode="redacted",
    user=_TestUser(),
)
payload = detail.model_dump(mode="json")
```

Assert:

```python
serialized = json.dumps(payload)
assert "Task prompt secret" not in serialized
assert "Assistant result secret" not in serialized
assert "Reviewer feedback secret" not in serialized
assert "[redacted]" in serialized
run_payload = payload["runs"][0]
assert run_payload["history"]["support_safe"] is True
assert run_payload["history"]["redacted_fields"]
assert run_payload["history"]["event_count"] == 2
assert run_payload["history"]["audit_event_count"] == 1
assert run_payload["history"]["artifact_count"] == 1
assert run_payload["history"]["diagnostic_count"] == 1
assert run_payload["history"]["stop_reason"] == "end"
assert run_payload["session"]["links"]["detail"].endswith("?redacted=true")
assert run_payload["session"]["links"]["events"].endswith("?redacted=true")
assert run_payload["session"]["links"]["artifacts"].endswith("?redacted=true")
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py::test_get_task_run_history_redacted_mode_omits_support_unsafe_text -q
```

Expected: FAIL because `get_task()` does not accept `run_summary_mode` yet.

## Stage 2: Backend Redaction Projection

**Goal:** Implement `run_summary_mode=full|redacted` without changing stored data or default response behavior.

**Success Criteria:** Redacted-mode test passes, and existing full-mode run-history tests still pass.

**Tests:** Focused Agent Orchestration API tests.

**Status:** Not Started

### Task 2: Implement redacted run-summary mode

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py`

- [ ] **Step 1: Add mode constants and query parameter**

Use `Literal["full", "redacted"]` or an equivalent local alias. Add:

```python
run_summary_mode: Annotated[
    RunSummaryMode,
    Query(description="Controls whether enriched task run summaries include full previews or support-safe redacted fields."),
] = "full"
```

Use `Annotated` so direct unit-test calls receive the string default rather than a `Query` object.

- [ ] **Step 2: Extend session link helper**

Change `_acp_session_links(session_id: str)` to accept `redacted: bool = False`. In redacted mode, append `?redacted=true` to links that support redacted drill-through:

- `detail`;
- `events`;
- `artifacts`.

Do not add `redacted=true` to diagnostics, audit, usage, updates, or event stream unless those endpoints support that parameter.

- [ ] **Step 3: Add redaction helpers**

Add small helpers near `_run_history_summary()`:

```python
_RUN_SUMMARY_REDACTED_MODE = "redacted"
_RUN_HISTORY_REDACTED_FIELDS = (
    "history.prompt.preview",
    "history.result.preview",
    "history.diagnostics.message",
    "history.diagnostics.diagnostic_uri",
    "result_summary",
    "error",
    "failure_context.message",
    "failure_context.diagnostic_uri",
    "review_decision.feedback_preview",
    "reviews.feedback",
)
```

Implement helper behavior:

- preserve `role` and `timestamp` for prompt/result preview objects;
- set `preview` to `[redacted]` only when a preview object exists;
- preserve diagnostic `session_id`, `index`, `timestamp`, `role`, and `reason_code`;
- set diagnostic `message` to `[redacted]` and `diagnostic_uri` to `[redacted]` when present;
- set `history.support_safe = True` and `history.redacted_fields = list(_RUN_HISTORY_REDACTED_FIELDS)`.

- [ ] **Step 4: Pass mode through enrichment**

Add `run_summary_mode` to `_enrich_task_runs()` and use redacted session links/history when mode is `redacted`. Preserve existing no-session zero-count history shape and add support-safe metadata there only in redacted mode.

- [ ] **Step 5: Redact run-level free text**

After full run enrichment is built, redact these run-level fields in redacted mode:

- `result_summary`;
- `error`;
- `failure_context.message`;
- `failure_context.diagnostic_uri`;
- `review_decision.feedback_preview`.
- top-level `reviews[].feedback`.

Keep `failure_context.reason_code`, `source`, run status, IDs, and counts.

- [ ] **Step 6: Run backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py::test_get_task_run_history_includes_acp_session_drillthrough \
  tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py::test_get_task_run_history_includes_failed_session_diagnostics \
  tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py::test_get_task_run_history_redacted_mode_omits_support_unsafe_text \
  -q
```

Expected: PASS.

## Stage 3: WebUI URL Contract

**Goal:** Make Agent Tasks callers able to request redacted task detail summaries without changing the default Inspect workflow.

**Success Criteria:** URL-building coverage proves redacted mode is available and normal Inspect still uses full mode by default.

**Tests:** Agent Tasks Vitest connection tests.

**Status:** Not Started

### Task 3: Add frontend URL builder support

**Files:**
- Modify: `apps/packages/ui/src/components/Option/AgentTasks/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx`

- [ ] **Step 1: Add a focused URL builder**

In `index.tsx`, add and export:

```ts
type RunSummaryMode = "full" | "redacted"

export const buildTaskDetailRequestUrl = (
  apiBase: string,
  taskId: number,
  options?: { runSummaryMode?: RunSummaryMode }
): string => {
  const url = `${apiBase}/tasks/${taskId}`
  if (!options?.runSummaryMode || options.runSummaryMode === "full") {
    return url
  }
  const params = new URLSearchParams({ run_summary_mode: options.runSummaryMode })
  return `${url}?${params.toString()}`
}
```

Use this helper in `handleInspectTask(taskId)` with no options so current UI behavior remains unchanged.

- [ ] **Step 2: Add Vitest coverage**

In `AgentTasksPage.connection.test.tsx`, import `buildTaskDetailRequestUrl` and add a direct test:

```ts
expect(buildTaskDetailRequestUrl(API_BASE, 11)).toBe(`${API_BASE}/tasks/11`)
expect(buildTaskDetailRequestUrl(API_BASE, 11, { runSummaryMode: "redacted" }))
  .toBe(`${API_BASE}/tasks/11?run_summary_mode=redacted`)
```

Keep the existing Inspect test expecting the unredacted task detail URL.

- [ ] **Step 3: Run frontend test**

Run:

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

## Stage 4: Documentation Updates

**Goal:** Replace the old “task previews are not support-safe” guidance with the new split.

**Success Criteria:** Docs explain when to use task-level redacted summaries versus ACP redacted session endpoints.

**Tests:** Documentation diff review plus `git diff --check`.

**Status:** Not Started

### Task 4: Update ACP docs

**Files:**
- Modify: `Docs/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md`
- Modify: `Docs/Development/Agent_Client_Protocol.md`
- Modify: `Docs/Development/ACP_Production_Readiness.md`

- [ ] **Step 1: Update user guide retention/support-safe section**

State that task detail supports `?run_summary_mode=redacted` for support/export overview summaries and ACP session `?redacted=true` remains the detailed support-safe drill-through.

- [ ] **Step 2: Update Agent Client Protocol development doc**

Change the task run transcript preview row from partial/tracked-by-#2408 to compliant or implemented with caveats, explaining the full/redacted split.

- [ ] **Step 3: Update production readiness**

Update the run history row and retention/redaction table to mention the new task-level redacted summary mode and remove stale language that says public evidence must always pivot away from task detail.

- [ ] **Step 4: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

## Stage 5: Verification, Security, and Tracking

**Goal:** Finish with focused verification, Bandit on touched backend code, and updated Backlog state.

**Success Criteria:** Tests pass or known environment blockers are documented; Backlog task has touched files and verification notes.

**Tests:** Focused backend pytest, frontend Vitest, Bandit, diff checks.

**Status:** Not Started

### Task 5: Final verification and tracking

**Files:**
- Modify: `backlog/tasks/task-2392 - Implement-ACP-support-safe-Agent-Tasks-run-summaries.md`

- [ ] **Step 1: Run backend focused suite**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py -q
```

Expected: PASS or document unrelated existing failures.

- [ ] **Step 2: Run frontend focused suite**

Run:

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on backend touched scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py -f json -o /tmp/bandit_acp_task_run_summaries_2408.json
```

Expected: no new high/medium findings in touched code.

- [ ] **Step 4: Run diff checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files changed.

- [ ] **Step 5: Update Backlog task**

Record:

- touched files;
- verification commands and outcomes;
- known skips or blockers;
- final summary.

- [ ] **Step 6: Commit**

Run:

```bash
git add \
  Docs/superpowers/specs/2026-06-19-acp-support-safe-task-run-summaries-design.md \
  Docs/superpowers/plans/2026-06-19-acp-support-safe-task-run-summaries-plan.md \
  "backlog/tasks/task-2392 - Implement-ACP-support-safe-Agent-Tasks-run-summaries.md" \
  tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py \
  tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py \
  apps/packages/ui/src/components/Option/AgentTasks/index.tsx \
  apps/packages/ui/src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx \
  Docs/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md \
  Docs/Development/Agent_Client_Protocol.md \
  Docs/Development/ACP_Production_Readiness.md
git commit -m "feat: add ACP support-safe task run summaries"
```

Expected: commit succeeds.
