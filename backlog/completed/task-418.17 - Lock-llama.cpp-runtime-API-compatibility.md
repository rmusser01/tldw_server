---
id: TASK-418.17
title: Lock llama.cpp runtime API compatibility
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-18 19:43
labels:
- llamacpp
- backend
- api
dependencies: []
documentation:
- Docs/superpowers/plans/2026-05-17-llamacpp-managed-runtime-implementation-plan.md
parent_task_id: TASK-418
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 3 from the llama.cpp managed runtime closeout plan: prove the legacy one-server API targets only the reserved default profile while profile and instance APIs expose all managed runtimes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 POST /api/v1/llamacpp/start_server updates/starts only the reserved default profile.
- [x] #2 POST /api/v1/llamacpp/start-by-model updates/starts only the reserved default profile.
- [x] #3 GET /api/v1/llamacpp/status keeps the legacy response shape and does not enumerate every profile.
- [x] #4 GET /api/v1/llamacpp/instances returns both default and non-default managed profiles.
- [x] #5 GET /api/v1/llamacpp/logs/tail maps default-profile stopped/not-running state to HTTP 409.
- [x] #6 New profile/runtime routes require admin permissions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 3 from Docs/superpowers/plans/2026-05-17-llamacpp-managed-runtime-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Red check: focused API/auth tests failed only for test_v1_log_tail_returns_conflict_when_default_runtime_is_stopped before the endpoint patch. Green check: focused Task 3 suite passed with 79 passed, 5 warnings after patching the legacy default log-tail path.
Renumbered from TASK-418.14 to TASK-418.17 on 2026-10-31 (TASK-13960): TASK-418.14 now names only "Implement WebUI settings and model provider remediation" (backlog/tasks/task-418.14 - Implement-WebUI-settings-and-model-provider-remediation.md). Before 2026-10-31 both tasks used TASK-418.14; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Locked the Task 3 llama.cpp runtime API compatibility contract. Legacy V1 start-by-model/start-server continue to target only the reserved default profile, legacy status keeps its one-server response shape, instances expose default plus non-default runtimes, legacy default log tail now returns HTTP 409 when the default runtime is not running, and profile/runtime routes are covered by 401/403 admin-gate tests. Verification: focused Task 3 pytest suite passed with 79 passed and 5 warnings; git diff --check passed; Bandit on the touched runtime endpoint reported zero findings. Bandit over touched tests reported existing pytest/assert and fixture-literal findings, with no runtime-code findings.
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
