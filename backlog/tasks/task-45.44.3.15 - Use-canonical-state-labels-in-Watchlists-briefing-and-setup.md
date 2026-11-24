---
id: TASK-45.44.3.15
title: Use canonical state labels in Watchlists briefing and setup
status: Done
labels:
- watchlists
- design-system
references:
- https://github.com/rmusser01/tldw_server/issues/1660
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the Watchlists Ready and Unavailable label findings from GitHub issue #1660 in Latest Briefing and Pipeline Wizard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The product-state verifier reports no blocked findings for the two Watchlists files.
- [x] #2 Visible and accessible status text remains correct.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Reuse the current design-system state combinations.
The targeted product-state verifier reproduced all six findings before the edit and reported zero afterward. The full verifier still reports 95 blocked findings elsewhere. Comparing finding IDs against the clean c39796aabc base confirmed that only the six Watchlists findings were removed.
Vitest passed all 152 tests across LatestBriefing.test.tsx, PipelineWizard.test.tsx, and watchlists-latest-locale.test.ts. The Impeccable detector reported zero findings on both components. Bandit was out of scope for this TypeScript-only change. Existing user-facing copy and locale resources need no documentation update.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Use the shared design-system Ready and Unavailable labels as localized fallbacks in Latest Briefing and Pipeline Wizard. The sample test now asserts the Ready and Filtered outcomes inside their accessible source-test list.
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
