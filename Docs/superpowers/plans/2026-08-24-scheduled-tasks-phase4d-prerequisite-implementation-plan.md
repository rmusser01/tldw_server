# Scheduled Tasks Phase 4D.0 Missing-Definition Prerequisite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development while implementing this plan. Use superpowers:verification-before-completion before committing. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix TASK-13113 so an `agent_task_run` Job whose owner-scoped definition is unavailable completes deterministically as a skipped Job without creating an invalid run, notification, or definition-scoped audit record.

**Architecture:** Keep the normalized Scheduled Tasks foreign-key and ownership invariants unchanged. Perform the owner-scoped definition lookup before run creation, return an explicit no-resource outcome when the lookup fails, and leave the existing create-run/dedupe/execute/finalize path untouched for valid definitions.

**Tech Stack:** Python 3.10+, asyncio, Loguru, SQLite-backed `ScheduledTasksDatabase` and `CollectionsDatabase`, Jobs worker contract, pytest, Bandit.

**Spec:** `Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md`

**Backlog tasks:** `TASK-13113` (implementation), `TASK-13116` (planning)

## Global Constraints

- This is the Phase 4D.0 prerequisite only. Do not add Phase 4D revisions, grants, dispatch tokens, secure transcripts, agent execution, migration, API routes, or client changes.
- Preserve `ScheduledTasksDatabase.create_scheduled_task_run()` exactly. A run cannot legally reference a missing definition, and this fix must not weaken that invariant or invent a tombstone definition.
- The missing-definition Job result is exactly `status="skipped"`, the requested `definition_id`, `run_id=None`, and `reason="definition_missing"`.
- The Jobs worker must treat that returned result as successful completion of the Job. It must not route the condition through `fail_job()`.
- No user notification is created because there is no valid definition name or notification policy to authorize one.
- No Scheduled Tasks audit row or metric label is created because both current audit storage and meaningful definition-scoped dimensions require a valid owner-scoped definition. The bounded structured warning and persisted Jobs result are the observable record of the exception.
- The warning may include only `definition_id`, numeric `user_id`, and Jobs `job_id`. It must not include the Job payload, prompt/input, credentials, or a distinction between absent, deleted, and cross-owner definitions.
- Preserve all valid-definition behavior: run-slot dedupe, lifecycle recheck, tool boundary, executor dispatch, timeout, cancellation propagation, health changes, notifications, and definition-scoped audit.
- Preserve Watchlists and standalone Agent Tasks unchanged. This fix only touches the existing Scheduled Tasks Jobs consumer and its focused tests.
- Do not modify `Scheduled_Tasks_DB.py`; the observed `KeyError` is the storage invariant working as designed.

## Delivery Stages

| Stage | Goal | Success Criteria | Tests | Status |
| --- | --- | --- | --- | --- |
| 1 | Freeze the missing-definition outcome | Focused regression fails with the current pre-run `KeyError` | One targeted pytest node | Not Started |
| 2 | Move lookup before run creation | Missing definitions return the exact skipped result and produce no run, notification, audit, or executor call | Focused consumer suite | Not Started |
| 3 | Prove adjacent behavior is unchanged | Automation executor and storage-adjacent tests remain green | Focused regression matrix | Not Started |
| 4 | Close security and review gates | Bandit, diff checks, self-review, Backlog evidence, and one commit are complete | Bandit and git checks | Not Started |

## File Map

**Modify**

- `tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py` - owner-scoped definition preflight and typed no-resource outcome.
- `tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py` - missing and cross-owner definition regressions.
- `backlog/tasks/task-13113 - Fix-Agent-Task-Jobs-consumer-missing-definition-crash.md` - implementation notes, verification, and final status.

**Read-only reference**

- `tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py` - retain the definition-before-run storage invariant.
- `tldw_Server_API/app/services/agent_task_jobs_worker.py` - verify that returned outcomes call `complete_job()`.

### Task 0: Rebase, Attach The Plan, And Record The Baseline

**Files:**
- Modify: `backlog/tasks/task-13113 - Fix-Agent-Task-Jobs-consumer-missing-definition-crash.md`

**Interfaces:**
- Consumes: current `origin/dev`, the approved Phase 4D design, and the existing `handle_agent_task_job()` contract.
- Produces: a clean implementation branch with the task In Progress and this plan linked.

- [ ] **Step 1: Create or reuse an isolated worktree from current dev**

```bash
git fetch origin dev
git worktree add .worktrees/scheduled-tasks-phase4d-prerequisite -b codex/scheduled-tasks-phase4d-prerequisite origin/dev
cd .worktrees/scheduled-tasks-phase4d-prerequisite
git status --short --branch
git log -1 --format='%H %s'
```

Expected: the feature worktree is based on the fetched `origin/dev` tip and has no unrelated changes. If TASK-13113 already has a branch or worktree, use it after proving its merge base and status instead of creating another.

- [ ] **Step 2: Link the implementation task to this plan**

Use the Backlog.md MCP workflow to set `TASK-13113` to In Progress, add this plan under documentation/references, and record the dev base SHA. Do not edit the task file directly while MCP or CLI is available.

- [ ] **Step 3: Run the current focused test to confirm the defect**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py::test_missing_definition_skips
```

Expected before the test change: FAIL with `KeyError: definition not found:` from `ScheduledTasksDatabase.create_scheduled_task_run()`. Record a different failure as a baseline discrepancy before proceeding.

### Task 1: Freeze The No-Resource Result Contract

**Files:**
- Modify: `tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py`

**Interfaces:**
- Consumes: one Job with a valid owner and a definition ID unavailable in that owner's database.
- Produces: the exact result `{status: "skipped", definition_id, run_id: None, reason: "definition_missing"}` plus proof that no dependent resource or execution occurred.

- [ ] **Step 1: Replace the weak missing-definition assertion with a complete regression**

Import `AsyncMock` and `Mock` from `unittest.mock`, and import the `agent_task_jobs` module so its module logger can be patched without changing global Loguru configuration. Ensure the per-user Scheduled Tasks schema exists without creating a definition, register an `AsyncMock` executor, and use a Job with `scheduled_for=SLOT`.

The test must assert all of the following:

```python
assert result == {
    "status": "skipped",
    "definition_id": missing_definition_id,
    "run_id": None,
    "reason": "definition_missing",
}
assert sdb.get_scheduled_task_run_by_slot(
    definition_id=missing_definition_id,
    run_slot_key=SLOT,
) is None
audits, total = sdb.list_audit_events(
    owner_id=user_id,
    definition_id=missing_definition_id,
)
assert audits == [] and total == 0
assert _latest_notification(user_id) is None
executor.assert_not_awaited()
warning.assert_called_once()
```

Inspect the captured warning call and assert its structured fields are limited to `definition_id`, `user_id`, and `job_id`; its message must not contain the serialized Job payload.

- [ ] **Step 2: Add the concealed cross-owner regression**

Create a valid definition for user A, explicitly initialize user B's Scheduled Tasks schema, submit a Job carrying the same definition ID as user B, and assert the identical `definition_missing` result with `run_id=None`. Assert user B has no run, audit, or notification and the executor is not called. Do not expose `cross_owner` as a separate reason.

- [ ] **Step 3: Run RED**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py \
  -k 'missing_definition or cross_owner'
```

Expected: both cases FAIL before executor dispatch because current code calls `create_scheduled_task_run()` first and raises `KeyError`. The failure must not be caused by fixture setup or a missing schema.

### Task 2: Preflight The Definition Before Creating A Run

**Files:**
- Modify: `tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py`

**Interfaces:**
- Consumes: owner-scoped `ScheduledTasksDatabase.get_definition(owner_id, definition_id)`.
- Produces: either a valid `DefinitionRow` used by the unchanged run path or the exact typed skipped result with no normalized resource creation.

- [ ] **Step 1: Move the definition lookup ahead of run creation**

Immediately after resolving `sdb`, perform the existing owner-scoped lookup and normalize `KeyError` to `definition=None`. For the missing branch:

```python
logger.warning(
    "Automation Job skipped because its definition is unavailable",
    definition_id=definition_id,
    user_id=user_id,
    job_id=job.get("id"),
)
return {
    "status": "skipped",
    "definition_id": definition_id,
    "run_id": None,
    "reason": "definition_missing",
}
```

Do not call `_finish()` in this branch. Construct `CollectionsDatabase` only after a valid definition is available, then execute the existing run creation and dedupe logic without changing its arguments or terminal-state behavior.

- [ ] **Step 2: Update the consumer docstring**

Document that `run_id` is nullable only for a pre-run missing-definition skip and that `reason="definition_missing"` identifies that exception. Do not describe a notification or audit event that cannot legally exist.

- [ ] **Step 3: Run GREEN for the complete consumer suite**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py
```

Expected: PASS. Existing valid-definition tests continue to create run rows and preserve lifecycle, dedupe, timeout, health, notification, and audit behavior.

### Task 3: Verify The Worker And Adjacent Automation Contracts

**Files:**
- Test only: `tldw_Server_API/tests/Notifications/test_automation_executors.py`
- Test only: `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py`
- Test only: `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py`

**Interfaces:**
- Consumes: the unchanged valid-definition consumer behavior and the Jobs worker rule that returned results complete Jobs.
- Produces: regression evidence that the narrow preflight did not alter other Scheduled Tasks automation behavior.

- [ ] **Step 1: Run the adjacent regression matrix**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notifications/test_automation_executors.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py
```

Expected: PASS with no changed expectations.

- [ ] **Step 2: Review the worker completion path without changing it**

Confirm `run_agent_task_jobs_worker()` still awaits `handle_agent_task_job()` and passes every returned dict to `JobManager.complete_job()`. Confirm `_AGENT_WORKER_NONCRITICAL_EXCEPTIONS` is no longer reached for this condition. No worker edit is expected.

- [ ] **Step 3: Check the focused diff**

```bash
git diff --check
git diff -- \
  tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py \
  tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py
```

Expected: only the preflight, result-contract documentation, and focused tests changed; no storage, Watchlists, Agent Tasks, API, or frontend file appears.

### Task 4: Security, Backlog, Review, And Commit

**Files:**
- Modify: `backlog/tasks/task-13113 - Fix-Agent-Task-Jobs-consumer-missing-definition-crash.md`

**Interfaces:**
- Consumes: green focused tests and the final two-file code diff.
- Produces: one reviewable prerequisite commit and a complete implementation record.

- [ ] **Step 1: Run Bandit on the touched production module**

```bash
source "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv/bin/activate"
python -m bandit -r \
  tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py \
  -f json -o /tmp/bandit_task_13113.json
```

Expected: exit 0 with no new findings in the changed code.

- [ ] **Step 2: Self-review against TASK-13113**

Verify: no run exists for the missing ID; no notification/audit is fabricated; the Job result and bounded warning are observable; cross-owner existence is concealed; valid dedupe/lifecycle/timeout/error paths are unchanged; no payload or secret appears in logging.

- [ ] **Step 3: Finalize Backlog evidence**

Use Backlog.md MCP to check acceptance criteria 1-4 and definition-of-done items 1-6, record exact test counts and Bandit result, and add a final summary explaining why the no-resource result is the only storage-consistent outcome. Keep the task In Progress until all checks pass, then mark it Done.

- [ ] **Step 4: Commit the prerequisite**

```bash
git status --short
git add \
  tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py \
  tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py \
  'backlog/tasks/task-13113 - Fix-Agent-Task-Jobs-consumer-missing-definition-crash.md'
git diff --cached --check
git commit -m "fix(scheduled-tasks): skip jobs with missing definitions"
```

Expected: one focused commit containing only TASK-13113 implementation, tests, and its Backlog record.

## Completion Review

- TASK-13113 is complete when the exact no-resource contract is implemented and the focused/adjacent suites pass.
- Do not begin Phase 4D.0F in the same commit. The feasibility gate has a separate plan, risk profile, evidence artifacts, and review boundary.
- Any request to create a synthetic definition, weaken foreign keys, hard-delete audit history, or send a guessed notification is out of scope and must be rejected as inconsistent with the approved design.
