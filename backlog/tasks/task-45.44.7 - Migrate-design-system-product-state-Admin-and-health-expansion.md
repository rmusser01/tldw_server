---
id: TASK-45.44.7
title: 'Migrate design-system product state: Admin and health expansion'
status: To Do
assignee: []
created_date: 2026-05-14 03:19
labels:
- design-system
- webui
- extension
- product-state
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/issues/1664
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
- apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44
priority: medium
documentation:
- "TASK-45.44.3.9 migrated Admin WatchlistsPage forbidden/not-found guard Alerts to\
  \ the design-system Alert primitive.\nBaseline evidence:\n  - total product-state\
  \ exceptions: 256 -> 254\n  - Admin and health expansion exceptions: 41 -> 39\n\
  \  - WatchlistsPage target rows: 2 -> 0\nVerification recorded in TASK-45.44.3.9."
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
