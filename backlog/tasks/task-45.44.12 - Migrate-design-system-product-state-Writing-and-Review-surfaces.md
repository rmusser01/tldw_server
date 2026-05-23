---
id: TASK-45.44.12
title: 'Migrate design-system product state: Writing and Review surfaces'
status: In Progress
assignee: []
created_date: '2026-05-14 03:20'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1669'
  - >-
    Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mirror the linked GitHub product-area migration issue. Closure requires zero current product-state baseline exceptions for the owned path map area and the verification gates recorded in the GitHub issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The linked GitHub issue owns current count and public status.
- [x] #2 Implementation PR tasks are created under this child when the area is too broad for one PR.
- [x] #3 Backlog notes record PR links and before/after count evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created TASK-45.44.12.2 for the narrow Review slice covering `MediaReviewReadingPane` failed-load Alert migration. Verification on the slice reduced the product-state baseline from 291 to 290 and `Writing and Review surfaces` from 21 to 20. PR: https://github.com/rmusser01/tldw_server/pull/1964
- Created TASK-45.44.12.3 for the narrow Writing slice covering `WritingPlaygroundWordcloudCard` wordcloud error Alert migration. Verification on the slice reduced the product-state baseline from 290 to 289 and `Writing and Review surfaces` from 20 to 19. PR: https://github.com/rmusser01/tldw_server/pull/1965
- Created TASK-45.44.12.4 for the narrow Writing slice covering `WritingPlaygroundResponseInspectorCard` response inspector guidance Alert migration. Verification on the slice reduced the product-state baseline from 289 to 288 and `Writing and Review surfaces` from 19 to 18. PR: https://github.com/rmusser01/tldw_server/pull/1966
- Created TASK-45.44.12.5 for the narrow Writing slice covering `WritingPlaygroundActiveSessionGuard` settings-load Alert migration. Verification on the slice reduced the product-state baseline from 288 to 287 and `Writing and Review surfaces` from 18 to 17. PR: https://github.com/rmusser01/tldw_server/pull/1967
- Created TASK-45.44.12.6 for the narrow Writing slice covering `WritingPlaygroundTokenInspectorCard` unavailable/error Alert migration. Verification on the slice reduced the product-state baseline from 287 to 285 and `Writing and Review surfaces` from 17 to 15. PR: https://github.com/rmusser01/tldw_server/pull/1971
- Created TASK-45.44.12.7 for the narrow Writing slice covering `WritingPlaygroundDiagnosticsPanel` offline/unsupported Alert migration. Verification on the slice reduced the product-state baseline from 285 to 283 and `Writing and Review surfaces` from 15 to 13. PR: https://github.com/rmusser01/tldw_server/pull/1972
- Created TASK-45.44.12.8 for the narrow Writing slice covering `ConnectionWebModal` project-required/no-data Empty and loading Spin migration. Verification on the slice reduced the product-state baseline from 283 to 280 and `Writing and Review surfaces` from 13 to 10. PR: https://github.com/rmusser01/tldw_server/pull/1974
- Created TASK-45.44.12.9 for the narrow Writing slice covering `WritingPlaygroundModalHost` extra_body/template/theme error Alert migration. Verification on the slice reduced the product-state baseline from 275 to 272 and `Writing and Review surfaces` from 10 to 7. PR: https://github.com/rmusser01/tldw_server/pull/1976
- Created TASK-45.44.12.10 for the narrow Writing slice covering `WritingPlayground` shell/session/editor Alert migration. Verification on the slice reduced the product-state baseline from 272 to 268 and `Writing and Review surfaces` from 7 to 3. PR: https://github.com/rmusser01/tldw_server/pull/1979
- Created TASK-45.44.12.11 for the narrow Writing slice covering the remaining `WritingPlayground` advanced-settings Alert migration plus current-dev `WritingActionBar` guard blockers. Verification on the slice reduced the product-state baseline from 268 to 265 and removed the final `Writing and Review surfaces` baseline rows. PR: https://github.com/rmusser01/tldw_server/pull/1998
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
