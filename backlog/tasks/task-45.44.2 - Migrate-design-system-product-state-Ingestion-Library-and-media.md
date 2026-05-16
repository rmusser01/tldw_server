---
id: TASK-45.44.2
title: 'Migrate design-system product state: Ingestion, Library, and media'
status: In Progress
assignee: []
created_date: '2026-05-14 03:19'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1659'
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
- [ ] #2 Implementation PR tasks are created under this child when the area is too broad for one PR.
- [ ] #3 Backlog notes record PR links and before/after count evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- TASK-45.44.2.4 migrates TimelineModal error and empty product-state alerts from AntD Alert to the shared design-system Alert primitive, reducing the baseline from 398 to 396 total exceptions and the Ingestion/Library/media area by 2. PR: https://github.com/rmusser01/tldw_server/pull/1785
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
