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
- [ ] #3 Backlog notes record PR links and before/after count evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created TASK-45.44.12.2 for the narrow Review slice covering `MediaReviewReadingPane` failed-load Alert migration. Verification on the slice reduced the product-state baseline from 291 to 290 and `Writing and Review surfaces` from 21 to 20. PR: https://github.com/rmusser01/tldw_server/pull/1964
- Created TASK-45.44.12.3 for the narrow Writing slice covering `WritingPlaygroundWordcloudCard` wordcloud error Alert migration. Verification on the slice reduced the product-state baseline from 290 to 289 and `Writing and Review surfaces` from 20 to 19. PR: https://github.com/rmusser01/tldw_server/pull/1965
- Created TASK-45.44.12.4 for the narrow Writing slice covering `WritingPlaygroundResponseInspectorCard` response inspector guidance Alert migration. Verification on the slice reduced the product-state baseline from 289 to 288 and `Writing and Review surfaces` from 19 to 18. PR: https://github.com/rmusser01/tldw_server/pull/1966
- Created TASK-45.44.12.5 for the narrow Writing slice covering `WritingPlaygroundActiveSessionGuard` settings-load Alert migration. Verification on the slice reduced the product-state baseline from 288 to 287 and `Writing and Review surfaces` from 18 to 17. PR: https://github.com/rmusser01/tldw_server/pull/1967
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
