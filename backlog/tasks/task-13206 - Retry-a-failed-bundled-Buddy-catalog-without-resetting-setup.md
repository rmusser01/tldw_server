---
id: TASK-13206
title: Retry a failed bundled Buddy catalog without resetting setup
status: Done
created_date: 2026-09-06 02:50
updated_date: 2026-09-06 03:03
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live UAT reproduced a failed bundled catalog load with no local retry action. Users need to recover the catalog without navigating away or losing their current Buddy configuration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failed bundled Buddy catalog offers a labeled retry action and preserves the error until a new attempt.
- [x] #2 Retry reloads the catalog in the current editor and disables repeated requests while loading without copying or activating a pack.
- [x] #3 Focused mounted tests and browser recovery verify the catalog can recover while the active pack is preserved.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Routine recovery affordance using the existing read-only catalog loader; no changed runtime, authentication or ownership boundary.
1. Add a mounted editor regression for failed catalog then manual retry and successful list.
2. Pass the existing loader through the guided builder to a conditional Retry catalog button, disabled while loading.
3. Run focused UI tests and lint/type checks; verify real browser recovery and document unresolved underlying network behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Exposed the existing read-only loader as Refresh catalog and Retry catalog on failure. The button disables during loading and preserves current pack/setup; no automatic network retry or mutating operation was added. The mounted regression failed before implementation, then all 71 catalog/builder tests passed. Scoped production TypeScript has zero diagnostics; lint has zero errors and unchanged warnings (editor 3, editor tests 47). A controlled isolated-backend outage produced the real browser error; after restart, Retry catalog restored choices while Migu Marker Basic stayed active, image loaded and idle. Guide, UAT report and sanitized source-hashed receipt updated. No ADR required for this routine recovery action; Bandit is not applicable to TypeScript-only changes and prior touched Python result has zero findings. The original intermittent cross-origin first-request cause remains unproven and explicitly documented.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Users can retry a failed bundled Buddy catalog in place without changing their active pack. Verified by 71 focused UI tests, scoped TypeScript/lint and real-browser outage/recovery evidence.
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
