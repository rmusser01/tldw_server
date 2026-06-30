---
id: TASK-431
title: Address PR 1848 llama.cpp log-tail review comments
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-18 20:02
labels:
- llamacpp
- backend
- review-fix
dependencies: []
documentation:
- https://github.com/rmusser01/tldw_server/pull/1848
- Docs/superpowers/plans/2026-05-17-llamacpp-managed-runtime-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address still-valid PR #1848 review findings for the legacy llama.cpp default log-tail endpoint. Verify and fix only actionable comments. Related managed-runtime parent: TASK-418.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default log-tail running-state check is not performed as blocking filesystem-prone endpoint code.
- [x] #2 Default log-tail check and tail operation are atomic with respect to the supervisor profile lock.
- [x] #3 Legacy default stopped/not-running log-tail behavior still maps to HTTP 409.
- [x] #4 Focused runtime API tests pass.
- [x] #5 Bandit is run for touched runtime code.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified Qodo findings as still valid: the endpoint-level get_runtime() could hit the JSON profile store synchronously, and the split state-check/tail sequence was non-atomic. Added LlamaCppSupervisor.tail_logs_if_running() to hold the profile lock, avoid store reads, enforce RUNNING semantics, and run runner log file reads off the event loop.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed both PR #1848 Qodo findings by moving default log-tail running-state enforcement into LlamaCppSupervisor.tail_logs_if_running(). The helper checks runner state and tails logs under the same profile lock, raises a 409-mapped domain conflict when the runtime is not running, avoids profile-store reads from the async endpoint, and offloads the runner log file read with asyncio.to_thread. Added focused supervisor tests for no store lookup, stopped-runner conflict, and stop/tail serialization. Verification: red helper tests failed before implementation; focused helper tests passed; broader focused API/supervisor/auth suite passed with 112 passed and 5 warnings; git diff --check passed; Bandit on touched runtime code reported zero findings.
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
