---
id: TASK-213
title: >-
  Implement ACP schedules triggers and background run productionization for
  issue 1474
status: Done
assignee: []
created_date: '2026-05-10 02:28'
updated_date: '2026-05-10 02:43'
labels:
  - ACP
  - schedules
  - triggers
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1474'
  - 'https://github.com/rmusser01/tldw_server/issues/1471'
documentation:
  - Docs/Plans/IMPLEMENTATION_PLAN_acp_schedules_triggers_1474.md
  - Docs/Development/ACP_Production_Readiness.md
  - Docs/Development/Agent_Client_Protocol.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the #1474 ACP schedules, triggers, and background runs workstream in the ACP productionization worktree. Start from the current failing schedule routing tests in test_acp_schedules.py, then harden ownership, concurrency, failure/retry visibility, trigger security, and operator documentation without broadening beyond the issue scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scheduled ACP runs route consistently to acp_run and preserve owner/user state
- [x] #2 Schedule concurrency and skipped-run behavior are explicit and test-covered
- [x] #3 Scheduled ACP failure retry and status outcomes are visible to operators
- [x] #4 Webhook trigger security and stored secret handling are documented and covered by tests
- [x] #5 GitHub issue #1474 is updated with implementation status and verification evidence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Stage 1: Root Cause and Scope
**Goal**: Stabilize the #1474 schedule routing baseline before expanding behavior.
**Success Criteria**: The current failing schedule routing tests have a traced root cause and a narrow compatibility contract.
**Tests**: Targeted failing test_acp_schedules.py cases.
**Status**: Complete

## Stage 2: Schedule Routing Compatibility
**Goal**: Ensure _load_all() and _rescan_once() can discover schedules from both modern DB handles and older/test handles.
**Success Criteria**: ACP schedules route to _add_acp_job, workflow schedules route to _add_job, and owner IDs are preserved.
**Tests**: Added _list_registered_schedules() fallback coverage and reran schedule routing tests.
**Status**: Complete

## Stage 3: Background Run State and Concurrency
**Goal**: Harden scheduled ACP run status, retry/failure visibility, and explicit concurrency semantics.
**Success Criteria**: ACP schedule execution records pending, queued, skipped, and error states in a way operators can inspect.
**Tests**: Added schedule execution tests for submit success, submit failure, disabled schedules, operator response state, and concurrency metadata.
**Status**: Complete

## Stage 4: Trigger Security and Docs
**Goal**: Verify webhook trigger security boundaries and document operational behavior for schedules/triggers.
**Success Criteria**: Trigger secret handling, replay/signature failures, and sanitized webhook errors remain covered; ACP docs/readiness matrix describe ownership, concurrency, failure, and security boundaries.
**Tests**: Trigger endpoint/core tests, schedule tests, Bandit, git diff --check.
**Status**: Complete
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause for the initial #1474 red state: _load_all() and _rescan_once() call _list_registered_schedules(); that helper only calls list_all_schedules(). The failing tests and older/fake DB handles expose list_schedules(), so schedule discovery returns no rows and both ACP and workflow job registration are silently skipped. Targeted reproduction: 3 failed in test_acp_schedules.py schedule routing cases.

Implemented #1474 schedule routing compatibility and operator-visible state. _list_registered_schedules() now falls back from list_all_schedules() to list_schedules() for older/test DB handles; disabled stale schedules record skipped_disabled; ACP schedule responses expose next_run_at, concurrency_mode, misfire_grace_sec, and coalesce; create/update validate and pass concurrency controls. Documentation now records APScheduler -> Scheduler acp_run ownership, schedule states, concurrency behavior, and webhook trigger security boundaries.

Verification: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_schedules.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_triggers_endpoint.py tldw_Server_API/tests/Agent_Client_Protocol/test_webhook_triggers.py -q => 54 passed, 5 warnings. Full ACP suite: python -m pytest tldw_Server_API/tests/Agent_Client_Protocol -q => 818 passed, 18 warnings. Bandit touched backend Python => /tmp/bandit_acp_schedules_triggers_1474.json results=0 errors=0 loc=1607. git diff --check => clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented #1474 ACP schedules/triggers/background-run productionization. Schedule discovery now supports both list_all_schedules() and list_schedules(), scheduled ACP disabled/submission-failure states are operator-visible, ACP schedule API responses expose next_run_at and concurrency controls, create/update validate concurrency mode, and ACP docs/readiness matrix document Scheduler ownership, state model, concurrency, and webhook trigger security. Verification: focused #1474 suite 54 passed; full Agent_Client_Protocol suite 818 passed; Bandit touched backend scope 0 findings; git diff --check clean. GitHub update posted: https://github.com/rmusser01/tldw_server/issues/1474#issuecomment-4414269193. No known blockers for this slice.
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
