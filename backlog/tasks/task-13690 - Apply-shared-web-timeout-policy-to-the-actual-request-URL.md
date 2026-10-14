---
id: TASK-13690
title: Apply shared web timeout policy to the actual request URL
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrating current dev PR3238 exposed that ordinary api.get/post pass URL separately while createAbortSignal reads config.url, so shared path floors see an empty path. Pass the actual request path at the one shared call site and verify real fetch AbortSignal behavior with fake timers, preserving explicit overrides, generic30s defaults, cancellation and RetryAfter seconds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Behavioral timer regressions reproduce missing floors on incoming dev
- [ ] #2 All web request callers use the actual path for shared timeout policy without new transport abstraction
- [ ] #3 Existing API behavior and connected source/type/build verification pass; final independent reviews and exact-head CI precede merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR; existing shared transport StageA plan in incoming PR3238 governs intended parity and frozen web error semantics; ADR069 remains Proposed contract-package boundary, not promoted. Minimal fix is data flow into existing deriveRequestTimeout. Related parent TASK13458 and final CI plan13685. Reviewer A confirmed ordinary request config lacks URL; actual request URL is already resolved before creating signal.
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
