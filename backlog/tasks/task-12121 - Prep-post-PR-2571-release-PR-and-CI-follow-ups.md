---
id: TASK-12121
title: Prep post-PR 2571 release PR and CI follow-ups
status: In Progress
labels:
- ci
- release
- pr-2571-followup
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2571
- https://github.com/rmusser01/tldw_server/pull/2596
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare the next dev-to-main release PR after PR #2571 by fixing still-valid CI findings from the merged release run, updating release metadata/changelog, validating locally, and opening a new PR against main.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Still-valid post-PR #2571 CI failures are fixed or documented as non-reproducible on current code.
- [x] Release metadata and changelog are prepared for the next dev-to-main release.
- [x] Focused backend, frontend, release-docs, and security validation are recorded.
- [x] Follow-up branch is pushed and a PR is opened against `main`.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Initial CI evidence from merged PR #2571: final check snapshot showed 6 failed checks, 773 pass, 4 pending, 2 cancelled, 5 skipped. Infrastructure-metrics failures reduce to test_notification_service.py::test_notify_generic_redacts_sensitive_payload_before_storage_and_webhook across Ubuntu/macOS/Windows plus one Ubuntu 3.12 distributed lock harness failure. Local focused reproduction on current dev: notification mutation test fails; distributed lock and sandbox concurrency tests pass on macOS/Python 3.11. UX Smoke Gate fails on stage6-interaction-stage1 chat-header-theme-toggle selector; current dev still has the stale selector.

Current branch was rebased onto updated `origin/dev` at 23b4e61af8. The notification payload mutation failure reproduced locally, then passed after changing `notify_generic()` to add synthetic `ts` metadata only to the copied/sanitized recorded payload. The distributed lock and sandbox concurrency failures both passed again locally on current code and were left unchanged as non-reproducible operational/test-harness failures.

The stale UX smoke check exposed a second current issue: Web runtime bootstrap strips the API key from stored `tldwConfig` and keeps it as a runtime single-user override, while `_app` only treated stored config API keys or build-time env auth as authenticated shell state. Added app-layout regression coverage and changed `_app` to treat `getRuntimeApiKey()` as authenticated for single-user shell visibility. Retargeted the theme-toggle smoke assertion to `/documentation`, a low-network shell route; the full stage 6 interaction stage 1 smoke file now passes.

Release prep updated project/docs metadata to 0.1.35 and added a 2026-07-03 changelog rollup for PRs #2582, #2583, #2584, #2586, #2588, #2574, #2593, #2594, #2060, #2061, #2062, plus the post-PR #2571 CI follow-ups.

Validation: `python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/Monitoring/test_notification_service.py::test_notify_generic_redacts_sensitive_payload_before_storage_and_webhook tldw_Server_API/tests/Docs/test_release_docs_contract.py tldw_Server_API/tests/Infrastructure/test_distributed_lock.py::TestFileLockAcquireRelease::test_release_does_not_unlink_same_process_reacquired_lock tldw_Server_API/tests/sandbox/test_execution_concurrency_cap.py::test_background_execution_respects_max_concurrent_runs` passed (16 passed). `bunx vitest run __tests__/app/app-layout.test.tsx` passed (17 passed). `npx playwright test e2e/smoke/stage6-interaction-stage1.spec.ts --reporter=line` passed (2 passed on rerun after one cold `/chat` navigation timeout). `git diff --check` passed. Bandit on `tldw_Server_API/app/core/Monitoring/notification_service.py` wrote `/tmp/bandit_task_12121.json` with 0 results and 0 errors.

Opened PR #2596 against `main`: https://github.com/rmusser01/tldw_server/pull/2596
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared the 0.1.35 release branch follow-up by fixing the valid Guardian notification mutation failure, fixing WebUI runtime-auth shell visibility so the theme-toggle smoke can render the shared header, retargeting the stale smoke assertion to a current shell route, and updating release metadata/changelog. The distributed lock and sandbox concurrency CI failures did not reproduce on current code and were not changed. Opened PR #2596 against `main`.
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
