# Scheduled Tasks Phase 4C Recurring Question Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `recurring_question` Scheduled Tasks executable through API-first manual runs, scheduled Jobs-backed runs, durable run/result history, Home surfacing, and WebUI/extension reference-client controls.

**Architecture:** Extend the Phase 4B Scheduled Tasks control plane with durable definition resolution state, runs, results, review state, and Recurring Question execution services. Jobs owns queueing, leasing, retry, cancellation, worker progress, and admin visibility; APScheduler only claims due schedule slots and calls the same run creation path; Scheduled Tasks owns user-facing run/result state and Home routing.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, SQLite per-user ScheduledTasks DB, core Jobs `WorkerSDK`, APScheduler, unified RAG request/response schemas, pytest, Bandit, TypeScript, React, TanStack Query, Ant Design, Vitest, and Playwright UI smoke checks.

---

## Context

Spec:
`Docs/superpowers/specs/2026-06-30-scheduled-tasks-phase4c-recurring-question-execution-design.md`

Planning task:
`TASK-12073`

Design task:
`TASK-12072`

This is an implementation plan only. Before implementation edits, create or reuse a separate implementation Backlog task and link this plan. The current plan task must not become the catch-all execution task unless the user explicitly asks for that.

## Scope

In scope:

- API-first Recurring Question execution contracts under `/api/v1/scheduled-tasks`.
- Durable run records for every execution attempt, including no-match and skipped runs.
- Normalized result artifacts for surfaced findings and attention-worthy failures.
- First-slice review states: `unread`, `read`, `dismissed`.
- Definition `resolution_state` with `mark-solved` and `reopen`.
- Manual `Run now` before scheduled execution.
- Jobs worker for Recurring Question execution.
- APScheduler registration for configured, open Recurring Questions.
- WebUI and extension as reference/main enterprise clients.
- Home Automation Inbox consuming normalized surfaced results when available.
- Watchlists behavior preserved and still separate.

Out of scope:

- Agent Task execution.
- Source-specific monitor UI for GitHub, YouTube, or any provider.
- Watchlists migration, replacement, or UX limitation.
- Corpus-change-triggered automation.
- External notification delivery expansion.
- Cross-product saved/bookmark/export behavior.
- Storing raw retrieved document text, secrets, provider keys, raw RAG debug dumps, or raw agent payloads in run/result records.

## File Map

Backend files to modify:

- `tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py`
  - Add definition extension columns, run/result tables, indexes, dataclasses, and repository methods.
- `tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py`
  - Add run, result, review, resolution, finding policy, retention, and reopen request/response schemas.
- `tldw_Server_API/app/services/scheduled_task_automation_service.py`
  - Keep Phase 4B definition/preview/lifecycle ownership; add capability action names and definition response fields.
- `tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py`
  - Add run/result/review/mark-solved/reopen routes before `/{task_id}`.
- `tldw_Server_API/app/services/scheduled_tasks_control_plane_service.py`
  - Project normalized Recurring Question status, next/last run, and result source refs into the unified task list.
- `tldw_Server_API/app/services/startup_content_jobs_pollers.py`
  - Register the Recurring Question Jobs worker under a feature flag.
- `tldw_Server_API/app/services/startup_recurring_schedulers.py`
  - Start/stop the APScheduler bridge for configured, open Recurring Questions.

Backend files to create:

- `tldw_Server_API/app/core/Scheduled_Tasks/__init__.py`
  - Package marker and public exports for Scheduled Tasks execution helpers.
- `tldw_Server_API/app/core/Scheduled_Tasks/recurring_question_models.py`
  - Internal dataclasses/enums for scope resolution, finding policy, run outcome, and worker result classification.
- `tldw_Server_API/app/core/Scheduled_Tasks/recurring_question_scope.py`
  - Capability-driven scope normalization and empty-scope validation.
- `tldw_Server_API/app/core/Scheduled_Tasks/recurring_question_rag_adapter.py`
  - Convert definition input, scope, and finding policy into a safe `UnifiedRAGRequest`; map RAG responses to evidence summaries.
- `tldw_Server_API/app/core/Scheduled_Tasks/recurring_question_jobs.py`
  - Jobs constants, enqueue helper, idempotency key builder, and worker-facing job payload helpers.
- `tldw_Server_API/app/services/scheduled_task_recurring_question_service.py`
  - Orchestrates run creation, result listing/detail, review mutations, mark solved, reopen, and Jobs enqueue.
- `tldw_Server_API/app/services/scheduled_task_recurring_question_worker.py`
  - Worker loop using `WorkerSDK` and `handle_recurring_question_run_job`.
- `tldw_Server_API/app/services/scheduled_task_recurring_question_scheduler.py`
  - APScheduler bridge that loads configured/open definitions and creates scheduled due runs.

Backend tests to add or extend:

- `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py`
- `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py`
- `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py`
- `tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py`
- `tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_scope.py`
- `tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_rag_adapter.py`
- `tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_jobs_worker.py`
- `tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_scheduler.py`
- `tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py`
- `tldw_Server_API/tests/Services/test_startup_recurring_schedulers.py`

Frontend files to modify:

- `apps/packages/ui/src/services/scheduled-tasks-control-plane.ts`
  - Add typed run/result/review/resolution client methods and response types.
- `apps/packages/ui/src/services/__tests__/scheduled-tasks-control-plane.test.ts`
  - Cover new routes, route encoding, review body, idempotency headers, and normalized result contracts.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskAutomationDefinitionEditor.tsx`
  - Replace raw-first Scope JSON with guided Recurring Question controls and keep advanced JSON as an escape hatch.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
  - Wire capabilities, run now, mark solved, reopen, run/result queries, and refresh behavior.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskDetailDrawer.tsx`
  - Add run history and definition-scoped results.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskResultsPanel.tsx`
  - Prefer normalized API results and keep projected legacy signals labeled.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskResultDetailDrawer.tsx`
  - Show answer/evidence, source refs, finding rationale, run metadata, review controls, and diagnostics.
- `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-results.ts`
  - Add normalized result mapping and keep legacy projection fallback.
- `apps/packages/ui/src/components/Option/CompanionHome/cards/AutomationInboxCard.tsx`
  - Consume normalized surfaced Recurring Question findings/failures when provided by the page/home data model.
- `apps/tldw-frontend/extension/routes/option-scheduled-tasks.tsx`
  - Verify compact list/detail/result views and deep links.

Frontend tests to add or extend:

- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskAutomationDefinitionEditor.test.tsx`
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx`
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx`
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultDetailDrawer.test.tsx`
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts`
- `apps/packages/ui/src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx`

## Guardrails

- Preserve Watchlists as a separate user job and UX. Do not remove, hide, rename, or constrain Watchlists controls.
- Keep all new API surfaces under `/api/v1/scheduled-tasks`.
- Add specific routes such as `/results`, `/runs/{run_id}`, and `/definitions/{definition_id}/runs` before the catch-all `/{task_id}` route.
- Keep action statuses compatible with Phase 4B: `available`, `unavailable`, `planned`, or `disabled`. Do not add action-level `degraded`.
- Use family-level `family_availability="degraded"` plus action reasons for partial readiness.
- `all_searchable_library` means capability-reported RAG sources that are enabled, searchable, and readable by the current owner.
- Empty-scope dry runs are out of scope for 4C. Preview and run admission fail with `scope_empty`.
- `generation_mode` does not add provider/model picker UX. Use existing RAG defaults/profiles or preview-validated safe overrides.
- Manual runs are allowed for paused definitions, but solved definitions must be reopened before running.
- Scheduled runs only register configured, open, non-archived, non-disabled definitions.
- Every execution attempt creates a run record, even skipped/no-match cases.
- Create result records only for surfaced findings and attention-worthy failures according to visibility policy.
- Do not duplicate raw source text in run/result records. Store source IDs, titles, citation refs, short redacted snippets, scores, and retrieval metadata.
- Use `source .venv/bin/activate` before Python, pytest, or Bandit.
- Commit after each coherent passing stage.

---

## Stage 0: Implementation Backlog Setup

**Status:** Complete. Tracked in `TASK-12082`.

**Goal:** Establish execution tracking before code edits.

**Success Criteria:**

- A distinct implementation Backlog task exists.
- The task links this plan, design spec, and `TASK-12073`.
- The task is marked `In Progress`.

**Steps:**

- [ ] Read Backlog workflow guidance through MCP resource `backlog://workflow/overview`.
- [ ] Search for an existing implementation task:

```bash
backlog search "Scheduled Tasks Phase 4C Recurring Question implementation" --plain
```

- [ ] If no task exists, create one with labels `scheduled-tasks`, `phase-4c`, `implementation`, `api-first`.
- [ ] Add references:
  - `TASK-12072`
  - `TASK-12073`
  - `Docs/superpowers/specs/2026-06-30-scheduled-tasks-phase4c-recurring-question-execution-design.md`
  - `Docs/superpowers/plans/2026-07-01-scheduled-tasks-phase4c-recurring-question-execution-implementation-plan.md`
- [ ] Mark the task `In Progress`.

**Verification:**

```bash
backlog task <IMPLEMENTATION_TASK_ID> --plain
```

**Commit:** Include the implementation task file in the first implementation commit if it changes tracked files.

---

## Stage 1: Scheduled Tasks Storage And API Schemas

**Status:** Complete. Storage/schema work was implemented and verified in the Stage 1 commits.

**Goal:** Add durable definition extensions, runs, results, review state, and response schemas without starting execution.

**Success Criteria:**

- Existing per-user ScheduledTasks DBs auto-migrate with additive columns/tables.
- Definitions expose `resolution_state`, `resolved_at`, `resolved_by`, `resolved_result_id`, `finding_policy`, and `retention_policy`.
- Runs and results are owner-scoped and queryable.
- Repository tests prove owner isolation, canonical JSON storage, idempotent schema upgrades, and no raw source text storage.

**Steps:**

- [ ] Add failing repository tests in `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py`:

```python
def test_definition_extensions_default_to_open_and_findings_only(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    definition = _create_definition(repo)

    loaded = repo.get_definition(owner_id=101, definition_id=definition.id)

    assert loaded.resolution_state == "open"
    assert loaded.finding_policy["preset"] == "balanced_findings"
    assert loaded.retention_policy["mode"] == "default"
```

```python
def test_run_and_result_storage_is_owner_scoped_and_redacted(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    definition = _create_definition(repo)
    run = repo.create_run(
        owner_id=101,
        definition_id=definition.id,
        definition_version=definition.version,
        trigger_reason="manual",
        status="queued",
        outcome="none",
        scope_snapshot={"sources": ["media_db"]},
        finding_policy_snapshot={"preset": "balanced_findings"},
        rag_request_snapshot={"query": "What changed?", "sources": ["media_db"]},
        run_summary={"message": "Queued"},
    )
    result = repo.create_result(
        owner_id=101,
        definition_id=definition.id,
        run_id=run.id,
        kind="finding",
        title="Possible answer found",
        summary="One relevant source matched.",
        answer=None,
        answer_mode="evidence_only",
        confidence={"label": "medium"},
        source_refs=[{"source_id": "m1", "title": "Doc", "snippet": "short redacted"}],
        dedupe_key="rq:def:run:m1",
        visibility_destination={"home": True, "results": True},
    )

    assert repo.get_run(owner_id=101, run_id=run.id).id == run.id
    assert repo.get_run(owner_id=202, run_id=run.id) is None
    assert repo.get_result(owner_id=101, result_id=result.id).id == result.id
    assert repo.get_result(owner_id=202, result_id=result.id) is None
    assert b"RAW FULL DOCUMENT" not in _database_bytes(repo)
```

- [ ] Run the failing DB tests:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py -v
```

Expected: fails because fields and repository methods do not exist.

- [ ] Extend `DefinitionRow` and `_definition_from_row()` in `Scheduled_Tasks_DB.py`.
- [ ] Update `ensure_schema()` with additive `ALTER TABLE` helpers for existing DBs:
  - `resolution_state TEXT NOT NULL DEFAULT 'open'`
  - `resolved_at TEXT`
  - `resolved_by TEXT`
  - `resolved_result_id TEXT`
  - `finding_policy_json TEXT NOT NULL DEFAULT '{"preset":"balanced_findings"}'`
  - `retention_policy_json TEXT NOT NULL DEFAULT '{"mode":"default"}'`
- [ ] Add tables:
  - `scheduled_task_runs`
  - `scheduled_task_results`
- [ ] Add indexes:
  - runs by `(owner_id, definition_id, created_at)`
  - runs by `(owner_id, status)`
  - runs by `(owner_id, job_id)`
  - results by `(owner_id, definition_id, created_at)`
  - results by `(owner_id, review_state)`
  - unique index for `(owner_id, dedupe_key)` after ensuring every surfaced result receives a non-empty deterministic dedupe key; on conflict, the service returns the existing result instead of inserting a duplicate.
- [ ] Add dataclasses:
  - `RunRow`
  - `ResultRow`
- [ ] Add repository methods:
  - `create_run`
  - `update_run`
  - `get_run`
  - `list_runs`
  - `create_result`
  - `get_result`
  - `list_results`
  - `update_result_review_state`
  - `mark_definition_solved`
  - `reopen_definition`
- [ ] Add Pydantic schemas in `scheduled_tasks_automation_schemas.py`:
  - `ScheduledTaskRunStatus`
  - `ScheduledTaskRunOutcome`
  - `ScheduledTaskRunResponse`
  - `ScheduledTaskRunListResponse`
  - `ScheduledTaskResultResponse`
  - `ScheduledTaskResultListResponse`
  - `ScheduledTaskResultReviewRequest`
  - `ScheduledTaskReviewState`
  - `ScheduledTaskMarkSolvedRequest`
  - `ScheduledTaskReopenRequest`
- [ ] Keep all JSON dumps canonical with existing `_json_dumps()`.

**Verification:**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py \
  -v
```

**Commit:**

```bash
git add tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py \
  tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py
git commit -m "feat: add scheduled task run and result storage"
```

---

## Stage 2: Preview Validation, Capabilities, And Resolution Lifecycle

**Status:** Complete. Stage 2 added generic Recurring Question scope normalization, capability actions, preview normalization, mark-solved/reopen APIs, transition enforcement, and audit coverage.

**Goal:** Make Recurring Question definitions executable-ready at the control-plane layer without yet enqueueing work.

**Success Criteria:**

- Capabilities expose `create_run_manual`, `execute_scheduled`, `read_runs`, `read_results`, `mutate_results`, `mark_solved`, and `reopen`.
- Action statuses remain Phase 4B-compatible.
- Preview validation normalizes scope, finding policy, visibility, retention, and generation mode.
- Empty scope returns typed `scope_empty` validation/admission errors.
- Mark solved and reopen obey the transition matrix from the spec.

**Steps:**

- [ ] Add failing service tests in `test_scheduled_task_automation_service.py`:

```python
def test_capabilities_include_4c_actions_without_degraded_action_status(tmp_path):
    service, _repo = _service(tmp_path)

    caps = service.get_capabilities()
    recurring = next(item for item in caps.items if item.family == "recurring_question")

    for action in ["create_run_manual", "execute_scheduled", "read_runs", "read_results", "mutate_results", "mark_solved", "reopen"]:
        assert action in recurring.actions
        assert recurring.actions[action].status in {"available", "unavailable", "planned", "disabled"}
```

```python
def test_mark_solved_and_reopen_preserve_lifecycle_rules(tmp_path):
    service, _repo = _service(tmp_path)
    definition = _create_definition(service, initial_lifecycle="configured")

    solved = service.mark_solved(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        resolved_result_id=None,
    )
    reopened = service.reopen_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        target_lifecycle="paused",
    )

    assert solved.resolution_state == "solved"
    assert reopened.resolution_state == "open"
    assert reopened.lifecycle == "paused"
```

- [ ] Add failing API tests in `test_scheduled_task_automation_api.py` for `mark-solved`, `reopen`, and route ordering before `/{task_id}`.
- [ ] Run expected failures:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py \
  -v
```

- [ ] Create `tldw_Server_API/app/core/Scheduled_Tasks/__init__.py`.
- [ ] Create `recurring_question_models.py` with internal enums/constants:
  - finding policy presets
  - review states
  - supported generation modes
  - visibility policies
  - typed failure reasons.
- [ ] Create `recurring_question_scope.py`:
  - normalize `all_searchable_library`, `sources`, `collection_ids`, `tag_ids`, `saved_search_ids`, `source_types`, `date_window`, `workspace_id`, and `advanced_filters`.
  - reject unsupported fields with field errors unless schema version rules explicitly allow them.
  - return `scope_empty` when no capability-reported readable searchable source remains.
- [ ] Extend `_validate_recurring_question_config()` to validate:
  - question/prompt
  - scope object
  - finding policy
  - generation mode
  - retention policy
  - visibility policy default `findings_only`.
- [ ] Extend `ScheduledTaskDefinitionResponse` to include resolution/finding/retention fields.
- [ ] Add service methods in `scheduled_task_automation_service.py` or delegate to `scheduled_task_recurring_question_service.py`:
  - `mark_solved`
  - `reopen_definition`
- [ ] Add endpoint routes before `/{task_id}`:
  - `POST /definitions/{definition_id}/mark-solved`
  - `POST /definitions/{definition_id}/reopen`
- [ ] Extend `_AUTOMATION_ERROR_MAP` with:
  - `scope_empty`
  - `definition_solved`
  - `definition_resolution_transition_invalid`
  - `definition_archived`
  - `definition_disabled`
- [ ] Update audit events for `definition.marked_solved` and `definition.reopened`.

**Verification:**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py \
  -v
```

Actual Stage 2 verification:

- Focused Stage 2 pytest: `31 passed, 11 warnings`.
- Full automation service/API/scope pytest: `82 passed, 14 warnings`.
- `git diff --check`: passed.
- Bandit touched production scope: zero findings in `/tmp/bandit_scheduled_tasks_phase4c_stage2.json`.

**Commit:**

```bash
git add tldw_Server_API/app/core/Scheduled_Tasks \
  tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py \
  tldw_Server_API/app/services/scheduled_task_automation_service.py \
  tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py
git commit -m "feat: add recurring question readiness and resolution lifecycle"
```

---

## Stage 3: Run And Result API With Manual Enqueue

**Status:** Complete. Stage 3 added Jobs-backed manual run creation, run/result read APIs, result review mutation, idempotent manual run replay, and normalized API responses.

**Goal:** Add API-first manual `Run now`, run history, result listing/detail, and review mutation before implementing the RAG worker.

**Success Criteria:**

- `POST /definitions/{definition_id}/runs` creates an owner-scoped run and enqueues a Jobs job.
- Manual runs are allowed for `configured` and `paused` open definitions.
- Solved, archived, disabled, empty-scope, quota-blocked, and overlapping runs are rejected or skipped with typed reasons.
- `GET /definitions/{definition_id}/runs`, `GET /runs/{run_id}`, `GET /results`, `GET /results/{result_id}`, and `POST /results/{result_id}/review` return normalized schemas.
- Repeated manual clicks with the same `Idempotency-Key` return the same run response.

**Steps:**

- [ ] Add failing service/API tests:

```python
def test_manual_run_creates_run_and_jobs_payload(tmp_path, monkeypatch):
    service, repo = _recurring_question_service(tmp_path)
    definition = _create_ready_definition(repo)
    created_jobs = []
    monkeypatch.setattr(service, "_create_jobs_entry", lambda **kwargs: created_jobs.append(kwargs) or {"id": 123})

    run = service.create_manual_run(owner_id=OWNER_ID, actor=ACTOR, definition_id=definition.id, idempotency_key="run-1")

    assert run.status == "queued"
    assert run.trigger_reason == "manual"
    assert run.job_id == "123"
    assert created_jobs[0]["payload"]["run_id"] == run.id
```

```python
def test_result_review_mutation_is_owner_scoped(tmp_path):
    service, repo = _recurring_question_service(tmp_path)
    result = _create_result(repo, owner_id=OWNER_ID)

    updated = service.update_result_review_state(owner_id=OWNER_ID, result_id=result.id, review_state="dismissed")

    assert updated.review_state == "dismissed"
    with pytest.raises(ScheduledTaskAutomationError, match="result_not_found"):
        service.update_result_review_state(owner_id=OTHER_OWNER_ID, result_id=result.id, review_state="read")
```

- [ ] Run expected failures:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py \
  -v
```

- [ ] Create `scheduled_task_recurring_question_service.py`.
- [ ] Create `recurring_question_jobs.py` with:
  - `SCHEDULED_TASKS_DOMAIN = "scheduled_tasks"`
  - `RECURRING_QUESTION_JOB_TYPE = "recurring_question_run"`
  - `RECURRING_QUESTION_QUEUE = "scheduled-tasks"`
  - `build_manual_run_idempotency_payload()`
  - `build_scheduled_run_idempotency_key(definition_id, definition_version, schedule_slot)`
  - `enqueue_recurring_question_run_job(jm, run, owner_user_id, priority=5)`.
- [ ] Implement `create_manual_run()`:
  - load definition
  - check family, lifecycle, resolution, permissions/scope/capability placeholders
  - enforce overlap policy
  - create run status `queued`, outcome `none`
  - enqueue Jobs entry with `run_id`, `definition_id`, `definition_version`, owner, trigger reason, and schedule slot
  - update run with `job_id`
  - create audit event.
- [ ] Implement run/result list and detail methods.
- [ ] Implement review mutation method.
- [ ] Add endpoint routes before `/{task_id}`:
  - `POST /definitions/{definition_id}/runs`
  - `GET /definitions/{definition_id}/runs`
  - `GET /runs/{run_id}`
  - `GET /results`
  - `GET /results/{result_id}`
  - `POST /results/{result_id}/review`
- [ ] Ensure `GET /scheduled-tasks/results` returns normalized Recurring Question records. The frontend keeps merging labeled projected legacy signals for families that do not yet have normalized result APIs.

**Verification:**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py \
  -v
```

Actual Stage 3 verification:

- Focused Stage 3 pytest: `5 passed, 11 warnings`.
- Full scheduled task DB/service/API/scope pytest: `132 passed, 14 warnings`.
- `git diff --check`: passed.
- Bandit touched production scope: zero findings in `/tmp/bandit_scheduled_tasks_phase4c_stage3.json`.

**Commit:**

```bash
git add tldw_Server_API/app/core/Scheduled_Tasks/recurring_question_jobs.py \
  tldw_Server_API/app/services/scheduled_task_recurring_question_service.py \
  tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py \
  tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py
git commit -m "feat: add recurring question run and result APIs"
```

---

## Stage 4: RAG Adapter And Jobs Worker

**Goal:** Execute a queued Recurring Question run through the unified RAG pipeline and persist complete run summaries/results.

**Success Criteria:**

- Worker marks runs `running`, then terminal `completed`, `failed`, `skipped`, or `cancelled`.
- Finding, no-match, partial/degraded, generation-unavailable, RAG-unavailable, quota, permission, source-unavailable, and worker failure cases are typed.
- Evidence-only mode works when generation is unavailable and `generation_mode != required`.
- Results are created only when finding/failure rules and visibility policy route outside task history.
- RAG request snapshot excludes raw source text and secrets.

**Steps:**

- [ ] Add failing unit tests for `recurring_question_rag_adapter.py`:
  - maps scope/finding policy to `UnifiedRAGRequest`
  - rejects empty scope
  - disables generation for `generation_mode=disabled`
  - uses profile/defaults without provider/model picker
  - strips secrets/raw text from snapshots.
- [ ] Add failing worker tests in `test_scheduled_task_recurring_question_jobs_worker.py`:
  - synthesized finding
  - evidence-only finding
  - completed no match
  - generation unavailable fallback
  - generation required failure
  - RAG unavailable
  - quota exceeded
  - permission denied
  - retryable worker failure
  - terminal worker failure.
- [ ] Run expected failures:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_rag_adapter.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_jobs_worker.py \
  -v
```

- [ ] Implement `recurring_question_rag_adapter.py`:
  - `build_rag_request_from_definition(definition, scope_snapshot, finding_policy)`
  - `execute_recurring_question_rag(request_context, rag_executor=...)`
  - `summarize_rag_response(...)`
  - `classify_finding(...)`
  - `safe_rag_request_snapshot(...)`.
- [ ] Keep the adapter dependency-injectable. Tests must use a mocked RAG executor; do not require live embeddings/LLM.
- [ ] Implement `handle_recurring_question_run_job(job, rag_executor=None)` in `scheduled_task_recurring_question_worker.py`:
  - parse payload
  - load owner/definition/run
  - mark running
  - execute adapter
  - write run summary/outcome/failure reason
  - create result if routed
  - handle cancellation from Jobs worker SDK.
- [ ] Implement `run_recurring_question_jobs_worker(stop_event=None)` using `WorkerSDK`.
- [ ] Add startup registration in `startup_content_jobs_pollers.py` behind `SCHEDULED_TASKS_RECURRING_QUESTION_WORKER_ENABLED`.
- [ ] Extend startup tests to prove the worker spec is present and disabled/enabled through feature flags.

**Verification:**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_rag_adapter.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_jobs_worker.py \
  tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py \
  -v
```

**Commit:**

```bash
git add tldw_Server_API/app/core/Scheduled_Tasks/recurring_question_rag_adapter.py \
  tldw_Server_API/app/services/scheduled_task_recurring_question_worker.py \
  tldw_Server_API/app/services/startup_content_jobs_pollers.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_rag_adapter.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_jobs_worker.py \
  tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py
git commit -m "feat: execute recurring question runs through jobs"
```

---

## Stage 5: APScheduler Registration And Reconciliation

**Goal:** Schedule configured/open Recurring Questions through APScheduler and repair divergent Jobs/run state.

**Success Criteria:**

- Configured/open Recurring Questions are registered with APScheduler.
- Paused, solved, archived, and disabled definitions are not registered.
- Scheduled slots use deterministic idempotency keys.
- Missed-run and overlap policies are honored.
- Stale queued/running runs become `failed` or `skipped` with repair reasons.
- Orphaned completed Jobs create `needs_attention` repair events.

**Steps:**

- [ ] Add failing scheduler tests:

```python
def test_scheduler_registers_only_configured_open_definitions(tmp_path, monkeypatch):
    service = _scheduler_service(tmp_path)
    configured_open = _definition(lifecycle="configured", resolution_state="open")
    solved = _definition(lifecycle="configured", resolution_state="solved")
    paused = _definition(lifecycle="paused", resolution_state="open")

    loaded = service.load_due_definitions()

    assert configured_open.id in {item.definition_id for item in loaded}
    assert solved.id not in {item.definition_id for item in loaded}
    assert paused.id not in {item.definition_id for item in loaded}
```

- [ ] Add failing reconciliation tests for stale runs and orphaned jobs.
- [ ] Run expected failures:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_scheduler.py \
  -v
```

- [ ] Implement `scheduled_task_recurring_question_scheduler.py`:
  - `RecurringQuestionSchedulerService`
  - `start()`
  - `stop()`
  - `rescan()`
  - `_add_job(definition)`
  - `_enqueue_due_slot(definition_id, definition_version, schedule_slot)`.
- [ ] Support Phase 4B schedule kinds with explicit 4C behavior:
  - `cron`
  - `daily`
  - `weekly`
  - `interval`
  - `one_time` remains readable/editable for backward compatibility but is not registered for recurring APScheduler execution in 4C.
- [ ] Add reconciliation method to `scheduled_task_recurring_question_service.py`.
- [ ] Wire scheduler startup/shutdown through `startup_recurring_schedulers.py` or the existing recurring scheduler lifecycle group.
- [ ] Make capability probes report scheduler readiness and worker readiness.

**Verification:**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_scheduler.py \
  tldw_Server_API/tests/Services/test_startup_recurring_schedulers.py \
  -v
```

**Commit:**

```bash
git add tldw_Server_API/app/services/scheduled_task_recurring_question_scheduler.py \
  tldw_Server_API/app/services/startup_recurring_schedulers.py \
  tldw_Server_API/app/services/scheduled_task_recurring_question_service.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_scheduler.py \
  tldw_Server_API/tests/Services/test_startup_recurring_schedulers.py
git commit -m "feat: schedule recurring question runs"
```

---

## Stage 6: WebUI And Extension Reference Client

**Goal:** Make the WebUI and extension consume the new API-first contracts without replacing Watchlists.

**Success Criteria:**

- Service client exposes typed run/result/review/resolution methods.
- Recurring Question creation is guided and capability-aware.
- Raw Scope JSON remains only under advanced disclosure.
- Detail drawer shows lifecycle, resolution, next/last run, latest outcome, run history, results, audit, and diagnostics.
- Users can run now, pause/resume, mark solved, reopen, duplicate, archive, inspect failures, and review/dismiss results.
- `/scheduled-tasks/results` uses normalized results when available and labels legacy projected signals.
- Home Automation Inbox shows only surfaced findings/failures, not routine no-match runs.
- Extension compact views preserve list/detail/result readability and deep links.

**Steps:**

- [ ] Add failing client tests in `scheduled-tasks-control-plane.test.ts` for:
  - `createScheduledTaskRun`
  - `listScheduledTaskRuns`
  - `getScheduledTaskRun`
  - `listScheduledTaskResults`
  - `getScheduledTaskResult`
  - `updateScheduledTaskResultReview`
  - `markScheduledTaskDefinitionSolved`
  - `reopenScheduledTaskDefinition`.
- [ ] Add failing component tests for guided creation:
  - common scope controls render from capability fixtures
  - advanced JSON hidden by default
  - preview shows RAG readiness, generation mode, evidence-only fallback, quota/cost, retention, and destinations
  - create offers `Run now` when capability is available.
- [ ] Add failing results/detail tests:
  - normalized result maps to table and drawer
  - no-match runs are in run history but not Home by default
  - review state mutation updates UI
  - legacy projected signals are labeled.
- [ ] Run expected failures:

```bash
bunx vitest run \
  apps/packages/ui/src/services/__tests__/scheduled-tasks-control-plane.test.ts \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskAutomationDefinitionEditor.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultDetailDrawer.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
```

- [ ] Extend `scheduled-tasks-control-plane.ts` with the new types and methods.
- [ ] Refactor `ScheduledTaskAutomationDefinitionEditor.tsx`:
  - `Question or prompt`
  - guided scope controls
  - finding behavior presets
  - schedule controls
  - preview section
  - advanced JSON disclosure.
- [ ] Update `ScheduledTasksPage.tsx` queries/mutations:
  - capabilities
  - run now
  - mark solved
  - reopen
  - runs
  - results
  - review state.
- [ ] Update `ScheduledTaskDetailDrawer.tsx` with run history and result sections.
- [ ] Update `ScheduledTaskResultsPanel.tsx` and `ScheduledTaskResultDetailDrawer.tsx` to consume normalized API results.
- [ ] Update `scheduled-task-results.ts` so normalized results are primary and legacy projection remains fallback.
- [ ] Update `AutomationInboxCard.tsx` integration data path to accept normalized surfaced Recurring Question results.
- [ ] Verify `apps/tldw-frontend/extension/routes/option-scheduled-tasks.tsx` still renders compactly and deep links preserve task/run/result IDs.

**Verification:**

```bash
bunx vitest run \
  apps/packages/ui/src/services/__tests__/scheduled-tasks-control-plane.test.ts \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskAutomationDefinitionEditor.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultDetailDrawer.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts \
  apps/packages/ui/src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx
```

**Commit:**

```bash
git add apps/packages/ui/src/services/scheduled-tasks-control-plane.ts \
  apps/packages/ui/src/services/__tests__/scheduled-tasks-control-plane.test.ts \
  apps/packages/ui/src/components/Option/ScheduledTasks \
  apps/packages/ui/src/components/Option/CompanionHome/cards/AutomationInboxCard.tsx \
  apps/packages/ui/src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx \
  apps/tldw-frontend/extension/routes/option-scheduled-tasks.tsx
git commit -m "feat: add recurring question execution UI"
```

---

## Stage 7: Retention, Privacy, Accessibility, And Final Verification

**Goal:** Close product trust and operational hardening before PR.

**Success Criteria:**

- Retention policy prunes old no-match runs before surfaced results.
- Final solved finding is preserved unless dismissed or removed by policy.
- Audit retention follows existing audit policy.
- User-facing records exclude raw document text, secrets, provider keys, raw agent payloads, and raw RAG debug dumps.
- Keyboard and color-independent state coverage exists for new controls.
- Watchlists behavior is unchanged.
- Backend, frontend, Bandit, and targeted e2e smoke checks pass or documented skips are justified.

**Steps:**

- [ ] Add retention tests for no-match pruning, result preservation, and audit preservation.
- [ ] Add privacy tests that scan persisted run/result JSON for seeded raw text and secret sentinels.
- [ ] Add accessibility tests for:
  - keyboard-operable filters/actions/review controls
  - non-color-only status labels
  - result row action labels include task/result context
  - running status live text in every component that renders queued or running runs.
- [ ] Add Watchlists compatibility test proving existing projected Watchlists rows still appear and existing Watchlists UI tests still pass.
- [ ] Run backend targeted suite:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py \
  tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_scope.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_rag_adapter.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_jobs_worker.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_scheduler.py \
  tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py \
  tldw_Server_API/tests/Services/test_startup_recurring_schedulers.py \
  -v
```

- [ ] Run frontend targeted suite:

```bash
bunx vitest run \
  apps/packages/ui/src/services/__tests__/scheduled-tasks-control-plane.test.ts \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskAutomationDefinitionEditor.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultDetailDrawer.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts \
  apps/packages/ui/src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx
```

- [ ] Run Bandit on touched backend scope:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py \
  tldw_Server_API/app/core/Scheduled_Tasks \
  tldw_Server_API/app/services/scheduled_task_automation_service.py \
  tldw_Server_API/app/services/scheduled_task_recurring_question_service.py \
  tldw_Server_API/app/services/scheduled_task_recurring_question_worker.py \
  tldw_Server_API/app/services/scheduled_task_recurring_question_scheduler.py \
  tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py \
  -f json -o /tmp/bandit_scheduled_tasks_phase4c.json
```

- [ ] Run a WebUI smoke or Playwright check if a dev server is already part of the implementation turn. At minimum verify `/scheduled-tasks`, `/scheduled-tasks/results`, and extension sidepanel route render without console errors.
- [ ] Update the implementation Backlog task with verification results and any documented skips.

**Commit:**

```bash
git add <retention/privacy/a11y/test files> <implementation backlog task file>
git commit -m "test: harden recurring question execution coverage"
```

---

## Final Acceptance Checklist

- [ ] API clients can preview, create, inspect, run, monitor, solve, reopen, and review Recurring Question definitions.
- [ ] Manual `Run now` works before scheduled execution.
- [ ] Scheduled runs are APScheduler-triggered and Jobs-executed.
- [ ] Every execution attempt creates a durable run summary.
- [ ] Findings and attention-worthy failures create normalized result records.
- [ ] Home shows surfaced findings/failures, not routine no-match runs.
- [ ] `/scheduled-tasks` shows current state, active/running tasks, run history, failures, results, and actions.
- [ ] `/scheduled-tasks/results` shows normalized Recurring Question findings/failures and labeled legacy projected signals.
- [ ] Mark solved stops upcoming schedules and preserves history.
- [ ] Reopen respects lifecycle, archive, and disabled locks.
- [ ] Watchlists functionality and UX remain unchanged.
- [ ] Tests cover DB, service, API, worker, scheduler, frontend client, UI, retention, privacy, accessibility, and Watchlists compatibility.
- [ ] Bandit has no new findings in touched backend code.

## Execution Handoff

Recommended implementation mode: **Subagent-Driven**.

Use one fresh implementation agent per stage, then review each stage before continuing. Stages 1 through 6 are large enough to deserve separate commits and review gates. Stage 7 is the required final hardening gate.
