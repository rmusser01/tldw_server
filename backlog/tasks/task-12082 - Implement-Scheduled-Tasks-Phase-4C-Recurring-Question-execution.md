---
id: TASK-12082
title: Implement Scheduled Tasks Phase 4C Recurring Question execution
status: Done
assignee: []
created_date: ''
updated_date: 2026-08-24 06:01
labels:
- scheduled-tasks
- phase-4c
- implementation
- api-first
dependencies: []
references:
- Docs/superpowers/specs/2026-06-30-scheduled-tasks-phase4c-recurring-question-execution-design.md
- https://github.com/rmusser01/tldw_server/pull/2566
documentation:
- Docs/superpowers/plans/2026-07-01-scheduled-tasks-phase4c-recurring-question-execution-implementation-plan.md
priority: high
modified_files:
- apps/packages/ui/src/services/scheduled-tasks-control-plane.ts
- apps/packages/ui/src/services/__tests__/scheduled-tasks-control-plane.test.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskAutomationDefinitionEditor.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskDetailDrawer.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskResultDetailDrawer.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskResultsPanel.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-results.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskAutomationDefinitionEditor.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultDetailDrawer.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts
- apps/packages/ui/src/components/Option/CompanionHome/hooks.ts
- apps/packages/ui/src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx
- apps/packages/ui/src/components/Option/CompanionHome/__tests__/CompanionHomePage.test.tsx
- tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py
- tldw_Server_API/app/services/scheduled_task_recurring_question_service.py
- tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py
- tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py
- Docs/superpowers/plans/2026-07-01-scheduled-tasks-phase4c-recurring-question-execution-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved API-first Scheduled Tasks Phase 4C Recurring Question execution plan. Scope includes storage, schemas, preview/admission, manual runs, run/result APIs, Jobs worker, APScheduler bridge, Home surfacing, WebUI/extension reference-client behavior, retention/privacy hardening, tests, Bandit, and review checkpoints. Preserve Watchlists as a separate UX/job and do not introduce source-specific GitHub/YouTube monitor assumptions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Recurring Question definitions support preview, create/update, inspect, run now, scheduled execution, mark solved, reopen, review state, retention metadata, and durable run/result history through API-first contracts.
- [x] #2 Every execution attempt creates a run record; surfaced findings and attention-worthy failures create normalized result records; routine no-match runs remain discoverable in run history without polluting Home.
- [x] #3 Execution uses Jobs for queueing/worker lifecycle and APScheduler only for due schedule claims, with reconciliation for divergent Jobs/run state.
- [x] #4 WebUI and extension behave as reference/main enterprise API clients, including Home surfacing and `/scheduled-tasks` monitoring, without defining product boundaries in UI-only logic.
- [x] #5 Watchlists functionality and UX remain unchanged as a separate persona/job; GitHub and YouTube are treated only as examples, not privileged source assumptions.
- [x] #6 Storage, API, service, worker, scheduler, frontend client, UI, retention/privacy, accessibility, and Watchlists compatibility tests cover the implemented behavior.
- [x] #7 Bandit and targeted backend/frontend verification are run before completion, with any skips documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-01-scheduled-tasks-phase4c-recurring-question-execution-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation started from plan `Docs/superpowers/plans/2026-07-01-scheduled-tasks-phase4c-recurring-question-execution-implementation-plan.md` after rebasing the planning branch onto `origin/dev` at merge `85ab688b34`.

Stage 1 storage/schema slice added durable recurring-question definition resolution defaults, owner-scoped run/result persistence, result review state, owner+dedupe uniqueness, canonical JSON snapshots, and redacted source-ref guards. Verification: focused scheduled task DB pytest passed (31 passed, 3 warnings); Bandit on touched production files reported zero findings.

Stage 1 spec-review fix mapped persisted definition resolution/policy fields into service responses and added recursive private-key validation before run/result JSON persistence. Verification: focused scheduled task DB+service pytest passed (68 passed, 4 warnings); git diff --check passed; Bandit on touched production files reported zero findings.

Stage 1 local code-quality review tightened private payload key detection for common variants such as `rawText`, `openai_api_key`, and `access_token`. Verification after hardening: focused scheduled task DB+service pytest passed (71 passed, 4 warnings); git diff --check passed; Bandit on touched production files reported zero findings.

Stage 2 preview/readiness slice added source-agnostic Recurring Question scope normalization, finding/retention/generation preview normalization, 4C capability actions without action-level degraded status, `mark-solved` and `reopen` service/API routes, transition enforcement for archived/disabled/unsolved definitions, and audit coverage. Verification: focused Stage 2 pytest passed (31 passed, 11 warnings); full scheduled task automation service/API/scope pytest passed (82 passed, 14 warnings); git diff --check passed; Bandit on touched production files reported zero findings.

Stage 3 run/result API slice added idempotent manual Recurring Question run creation, Jobs enqueue payload helpers, normalized run/result list/detail routes, result review mutation with audit, API error aliases, and a legacy-missing-scope compatibility fallback to `all_searchable_library`. Verification: focused Stage 3 pytest passed (5 passed, 11 warnings); full scheduled task DB/service/API/scope pytest passed (132 passed, 14 warnings); git diff --check passed; Bandit on touched production files reported zero findings.

Stage 4 RAG worker slice added a dependency-injected Recurring Question RAG adapter, storage-safe RAG snapshots, worker run state transitions, finding/no-match/failure persistence, retryable failure propagation for Jobs retries, and declarative startup worker discovery behind `SCHEDULED_TASKS_RECURRING_QUESTION_WORKER_ENABLED`. Verification: focused Stage 4 pytest passed (41 passed, 3 warnings); broader scheduled task plus startup pytest passed (173 passed, 14 warnings); git diff --check passed; Bandit on touched production files reported zero findings.

Stage 5 scheduler slice added the APScheduler bridge for configured/open Recurring Questions, deterministic scheduled-run idempotency, overlap prevention, invalid-schedule isolation, stale queued/running run repair, orphaned completed Job/run `needs_attention` repair, lifecycle startup specs behind `SCHEDULED_TASKS_RECURRING_QUESTION_SCHEDULER_ENABLED`, and scheduler/worker readiness capability signals. Verification: focused scheduler pytest passed (6 passed, 3 warnings); adjacent automation/API/startup pytest passed (104 passed, 14 warnings); worker/startup dependency pytest passed (43 passed, 3 warnings); git diff --check passed; Bandit on touched production files reported zero findings in `/tmp/bandit_scheduled_tasks_phase4c_stage5.json`.

Stage 6 frontend/API-client slice added typed WebUI control-plane methods for runs, results, result review, mark-solved, and reopen; guided Recurring Question creation controls with advanced scope JSON disclosure; normalized result mapping and details with answer/evidence; definition detail run history/result sections and execution actions; `/scheduled-tasks` normalized results wiring with projected legacy fallback; and Home Automation Inbox normalized result surfacing that respects Home visibility and dismissed review state while preserving Watchlists and notification-derived signals. Verification: focused Companion Home normalized result test passed after the expected red failure; Companion Home suite passed (12 passed); Automation Inbox card and results panel suites passed (9 passed); full Stage 6 UI bundle passed (9 test files, 125 tests); git diff --check passed. Bandit skipped for Stage 6 because touched production code is TypeScript/UI only and no Python production files changed.

Stage 7 retention/privacy/accessibility hardening added repository pruning for old no-match runs and old dismissible results, service-level retention policy application, solved-result preservation until dismissal, and running-state live-region coverage in `/scheduled-tasks` results surfaces. Existing privacy guard tests continue to cover raw document text, raw RAG debug payloads, provider/API-key sentinels, and raw agent-message storage; existing Watchlists preservation tests continue to cover projected Watchlists rows, Home surfacing, and Watchlists links. Verification: backend targeted suite passed (202 passed, 14 warnings); frontend targeted suite passed (9 files, 126 tests, with existing jsdom CSS/i18next warnings); Bandit on touched backend production scope reported zero findings in `/tmp/bandit_scheduled_tasks_phase4c.json`; `git diff --check` passed. WebUI smoke was skipped because no dev server was part of this final hardening turn; route/component tests covered `/scheduled-tasks`, `/scheduled-tasks/results`, result drawers, Home surfacing, and Watchlists preservation.
2026-08-23 review remediation pass: fetched unresolved PR #2566 review threads and confirmed actionable items across backend reliability/validation/security, frontend result/home/editor behavior, and policy compliance. Next step is rebasing onto latest origin/dev, then applying test-first fixes for each still-valid comment.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Scheduled Tasks Phase 4C Recurring Question execution and merged it into dev in PR #2566 (merge commit 4958cfed65). The API-first slice includes definition preview/lifecycle, manual and scheduled Jobs execution, APScheduler due-run claims, durable run/result history, result review, mark solved/reopen, retention/privacy controls, /scheduled-tasks monitoring, and Home Automation Inbox surfacing while preserving Watchlists as a separate workflow. Final review remediation resolved all 28 PR threads. Verification passed: 115 focused backend tests, 94 focused frontend tests, git diff --check, and Bandit with zero findings. WebUI smoke remained the documented skip; route and component coverage exercised the relevant surfaces.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Run History
<!-- SECTION:RUN_HISTORY:BEGIN -->
<!-- RUN_HISTORY_ENTRY:BEGIN -->
```yaml
event_id: run-61afc7cd420a49db804b93b0fc30509b
type: record_run
actor: codex
timestamp: '2026-08-24T05:21:24.769008Z'
result: succeeded
task_id: TASK-12082
files:
- apps/packages/ui/src/components/Option/CompanionHome/__tests__/CompanionHomePage.test.tsx
- apps/packages/ui/src/components/Option/CompanionHome/hooks.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskAutomationDefinitionEditor.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskAutomationDefinitionEditor.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-results.ts
- tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py
- tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py
- tldw_Server_API/app/core/Scheduled_Tasks/recurring_question_jobs.py
- tldw_Server_API/app/core/Scheduled_Tasks/recurring_question_rag_adapter.py
- tldw_Server_API/app/core/Scheduled_Tasks/recurring_question_scope.py
- tldw_Server_API/app/core/exceptions.py
- tldw_Server_API/app/services/scheduled_task_automation_service.py
- tldw_Server_API/app/services/scheduled_task_recurring_question_scheduler.py
- tldw_Server_API/app/services/scheduled_task_recurring_question_service.py
- tldw_Server_API/app/services/scheduled_task_recurring_question_worker.py
- tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py
- tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py
- tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_jobs_worker.py
- tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_rag_adapter.py
- tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_scheduler.py
- tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_scope.py
verification:
- git diff --check
- python -m pytest -q tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_rag_adapter.py
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_scope.py
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_scheduler.py
  tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_jobs_worker.py
  --maxfail=20 (115 passed, 3 warnings)
- bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskAutomationDefinitionEditor.test.tsx
  src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts src/components/Option/CompanionHome/__tests__/CompanionHomePage.test.tsx
  src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx --maxWorkers=1
  --no-file-parallelism (4 files, 94 tests passed)
- python -m bandit -r touched backend production scope -f json -o /tmp/bandit_scheduled_tasks_phase4c_review.json
  (0 findings)
```
Rebased PR #2566 onto latest origin/dev and addressed review feedback: RAG snapshot key redaction, run pruning, scheduler resilience/schedule parsing/job preservation, worker cancellation/retry state, mark-solved family/error handling, manual-run idempotency, frontend scope precedence/result redaction/Home partial errors/latest-result selection, and Qodo compliance items.
<!-- RUN_HISTORY_ENTRY:END -->
<!-- SECTION:RUN_HISTORY:END -->
