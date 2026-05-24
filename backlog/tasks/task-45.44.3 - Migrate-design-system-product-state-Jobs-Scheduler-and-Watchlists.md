---
id: TASK-45.44.3
title: 'Migrate design-system product state: Jobs, Scheduler, and Watchlists'
status: To Do
assignee: []
created_date: '2026-05-14 03:19'
updated_date: '2026-05-23 22:34'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1660'
  - >-
    Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - 'https://github.com/rmusser01/tldw_server/pull/2012'
  - 'https://github.com/rmusser01/tldw_server/pull/2013'
documentation:
  - |-
    TASK-45.44.3.6 / PR #2012 migrated OutputsTab delivery-issues Alert to the design-system Alert primitive.
    Baseline evidence:
      - total product-state exceptions: 259 -> 258
      - Jobs/Scheduler/Watchlists exceptions: 24 -> 23
      - OutputsTab target rows: 1 -> 0
    Verification recorded in TASK-45.44.3.6.
  - |-
    TASK-45.44.3.7 / PR #2013 migrated RunsTab load-error and reliability-attention banners to the design-system Alert primitive.
    Baseline evidence:
      - total product-state exceptions: 258 -> 257
      - Jobs/Scheduler/Watchlists exceptions: 23 -> 22
      - RunsTab target rows: 1 -> 0
    Verification recorded in TASK-45.44.3.7.
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
