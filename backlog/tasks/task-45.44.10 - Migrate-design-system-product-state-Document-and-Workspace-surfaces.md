---
id: TASK-45.44.10
title: 'Migrate design-system product state: Document and Workspace surfaces'
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
  - 'https://github.com/rmusser01/tldw_server/issues/1667'
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
- Created and completed TASK-45.44.10.1 for the first narrow Document/Workspace slice: DocumentPickerModal Alert migration. PR: https://github.com/rmusser01/tldw_server/pull/1950. Before/after product-state verifier evidence in that child task reduced total baseline exceptions from 303 to 300 and Document/Workspace exceptions from 12 to 9.
- Created and completed TASK-45.44.10.2 for the next narrow Document/Workspace slice: DocumentViewer PDF/EPUB Alert migration. PR: https://github.com/rmusser01/tldw_server/pull/1952. Verifier evidence after the slice reduces total baseline exceptions from 300 to 296 and Document/Workspace exceptions from 9 to 5.
- Created and completed TASK-45.44.10.3 for the next narrow Document/Workspace slice: DocumentWorkspacePage loading/health Alert migration. PR: https://github.com/rmusser01/tldw_server/pull/1955. Verifier evidence after the slice reduces total baseline exceptions from 296 to 294 and Document/Workspace exceptions from 5 to 3.
- Created and completed TASK-45.44.10.4 to fix the merged PdfDocument design-system Alert translation binding discovered by the TypeScript check. PR: https://github.com/rmusser01/tldw_server/pull/1955.
- Created and completed TASK-45.44.10.5 for the next narrow Document/Workspace slice: ReferencesTab server-unavailable/error EmptyState migration. PR: https://github.com/rmusser01/tldw_server/pull/1959. Verifier evidence after the slice reduces total baseline exceptions from 294 to 292 and Document/Workspace exceptions from 3 to 1.
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
