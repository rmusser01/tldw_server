---
id: TASK-45.44.7
title: 'Migrate design-system product state: Admin and health expansion'
status: To Do
assignee: []
created_date: '2026-05-14 03:19'
updated_date: '2026-05-24 01:05'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1664'
  - >-
    Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
documentation:
  - >-
    TASK-45.44.3.9 / PR #2019 migrated Admin WatchlistsPage forbidden/not-found
    guard Alerts to the design-system Alert primitive.

    Baseline evidence:
      - total product-state exceptions: 256 -> 254
      - Admin and health expansion exceptions: 41 -> 39
      - WatchlistsPage target rows: 2 -> 0
    Verification recorded in TASK-45.44.3.9.
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

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
