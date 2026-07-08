---
id: TASK-12919
title: Fix residual apiSend GET burst on chat first-run checks
status: Done
labels:
- frontend
- bug
- uat
priority: High
modified_files:
- apps/packages/ui/src/services/api-send.ts
- apps/packages/ui/src/services/__tests__/api-send.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real-backend UAT after the background-proxy fix still shows four GET /api/v1/persona/profiles requests during /chat load. Trace shows the caller is useFirstRunCheck via apiSend, which bypasses bgRequest GET coalescing. Add the smallest shared fix so safe apiSend GETs single-flight concurrent identical requests, then rerun focused tests and real-backend UAT.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added shared apiSend GET single-flight coalescing and regression coverage. Focused unit suites passed: bunx vitest run src/services/__tests__/api-send.test.ts src/services/__tests__/background-proxy.test.ts (57 tests). Real-backend UAT passed against frontend 127.0.0.1:8080 and backend 127.0.0.1:8000 with zero 429s, zero bad HTTP responses, and no burst targets. Bandit skipped as not applicable to TS-only frontend files.
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
