---
id: TASK-496
title: Verify unified first-time solo onboarding end to end
status: To Do
references:
- TASK-489
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
documentation:
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-10-end-to-end-verification-security-and-release-gate
modified_files:
- apps/tldw-frontend/e2e/workflows/unified-first-run-onboarding.spec.ts
- apps/tldw-frontend/e2e/smoke/page-inventory.ts
- tldw_Server_API/tests/frontend_e2e/test_onboarding_workflow.py
- backlog/tasks/
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 10 release-gate slice from the unified onboarding plan. Add or update E2E coverage, run focused backend/frontend/Playwright checks, run Bandit on touched backend scope, and record final verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 E2E verifies setup shell hides navigation until skip/completion and first-source milestone appears after completion
- [ ] #2 Focused backend and frontend unit/integration checks pass or blockers are documented
- [ ] #3 Bandit and git diff whitespace checks are run before final completion
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
