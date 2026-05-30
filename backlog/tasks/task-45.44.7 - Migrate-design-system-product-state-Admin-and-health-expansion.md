---
id: TASK-45.44.7
title: 'Migrate design-system product state: Admin and health expansion'
status: In Progress
assignee: []
created_date: 2026-05-14 03:19
updated_date: 2026-05-30 11:26
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
documentation:
- "TASK-45.44.3.9 / PR #2019 migrated Admin WatchlistsPage forbidden/not-found guard\
  \ Alerts to the design-system Alert primitive.\nBaseline evidence:\n  - total product-state\
  \ exceptions: 256 -> 254\n  - Admin and health expansion exceptions: 41 -> 39\n\
  \  - WatchlistsPage target rows: 2 -> 0\nVerification recorded in TASK-45.44.3.9.\n\
  TASK-45.44.7.1 migrated ServerArgsEditor JSON validation feedback from AntD Alert\
  \ to the design-system Alert primitive.\nBaseline file evidence:\n  - total baseline\
  \ rows: 195 -> 194\n  - Admin path rows: 37 -> 36\n  - ServerArgsEditor target row:\
  \ 1 -> 0\n\nVerifier evidence:\n  - scoped verifier log had no ServerArgsEditor\
  \ findings\n  - full verifier remains blocked by unrelated current-dev drift outside\
  \ this slice\n\nTASK-45.44.7.2 migrated RbacEditorPage admin guard feedback from\
  \ AntD Alert to the design-system Alert primitive.\nBaseline file evidence:\n  -\
  \ total baseline rows: 194 -> 193\n  - Admin path rows: 36 -> 35\n  - RbacEditorPage\
  \ target row: 1 -> 0\n\nVerifier evidence:\n  - scoped verifier log had no RbacEditorPage\
  \ findings\n  - full verifier remains blocked by unrelated current-dev drift outside\
  \ this slice\n\nTASK-45.44.7.3 migrated RuntimeConfigPage forbidden and not-available\
  \ guard feedback from AntD Alert to the design-system Alert primitive in PR #2145.\n\
  Baseline file evidence:\n  - total baseline rows: 193 -> 191\n  - Admin path rows:\
  \ 35 -> 33\n  - RuntimeConfigPage target rows: 2 -> 0\n\nVerifier evidence:\n  -\
  \ scoped verifier log had no RuntimeConfigPage findings\n  - full verifier remains\
  \ blocked by unrelated current-dev drift outside this slice\n\nTASK-45.44.7.4 migrated\
  \ MaintenancePage forbidden and not-available guard feedback from AntD Alert to\
  \ the design-system Alert primitive in PR #2148.\nBaseline file evidence:\n  - total\
  \ baseline rows: 191 -> 189\n  - Admin path rows: 33 -> 31\n  - MaintenancePage\
  \ target rows: 2 -> 0\n\nVerifier evidence:\n  - scoped verifier log had no MaintenancePage\
  \ findings\n  - full verifier remains blocked by unrelated current-dev drift outside\
  \ this slice\n\nTASK-45.44.7.5 / PR #2149 migrated UsageAnalyticsPage forbidden\
  \ and not-available guard feedback from AntD Alert to the design-system Alert primitive.\n\
  Baseline file evidence:\n  - total baseline rows: 189 -> 187\n  - Admin path rows:\
  \ 31 -> 29\n  - UsageAnalyticsPage target rows: 2 -> 0\n\nVerifier evidence:\n \
  \ - scoped verifier log had no UsageAnalyticsPage findings\n  - full verifier remains\
  \ blocked by unrelated current-dev drift outside this slice\n\nTASK-45.44.7.6 /\
  \ PR #2152 migrated BillingDashboardPage forbidden and unsupported-route guard feedback\
  \ from AntD Alert to the design-system Alert primitive.\nBaseline file evidence:\n\
  \  - total baseline rows: 187 -> 185\n  - Admin path rows: 29 -> 27\n  - BillingDashboardPage\
  \ target rows: 2 -> 0\n\nVerifier evidence:\n  - scoped verifier log had no BillingDashboardPage\
  \ findings\n  - full verifier remains blocked by unrelated current-dev drift outside\
  \ this slice\n\nTASK-45.44.7.7 / PR #2153 migrated OrgsTeamsPage forbidden and not-available\
  \ guard feedback from AntD Alert to the design-system Alert primitive.\nBaseline\
  \ file evidence:\n  - total baseline rows: 185 -> 183\n  - Admin path rows: 27 ->\
  \ 25\n  - OrgsTeamsPage target rows: 2 -> 0\n\nVerifier evidence:\n  - scoped verifier\
  \ log had no OrgsTeamsPage findings\n  - full verifier remains blocked by unrelated\
  \ current-dev drift outside this slice\n\nTASK-45.44.7.8 / PR #2154 migrated DataOpsPage\
  \ forbidden and not-available guard feedback from AntD Alert to the design-system\
  \ Alert primitive.\nBaseline file evidence:\n  - total baseline rows: 183 -> 181\n\
  \  - Admin path rows: 25 -> 23\n  - DataOpsPage target rows: 2 -> 0\n\nVerifier\
  \ evidence:\n  - scoped verifier log had no DataOpsPage findings\n  - full verifier\
  \ remains blocked by unrelated current-dev drift outside this slice"
- |-
  TASK-45.44.7.9 / PR #2156 migrated ServerAdminPage users, roles, and media budget inline error feedback from AntD Alert to the design-system Alert primitive.
  Baseline file evidence:
    - total baseline rows: 181 -> 178
    - Admin path rows: 23 -> 20
    - ServerAdminPage target rows: 3 -> 0

  Verifier evidence:
    - scoped verifier log had no ServerAdminPage findings
    - full verifier remains blocked by unrelated current-dev IntegrationPolicyPanel baseline drift/stale rows outside this slice
- |-
  TASK-45.44.7.10 / PR #2158 migrated MlxAdminPage admin guard, temporary-unavailable, active-model, and security-risk product-state UI from AntD Alert/Tag to the design-system Alert/Badge primitives.
  Baseline file evidence:
    - total baseline rows: 178 -> 174
    - Admin path rows: 20 -> 16
    - MlxAdminPage target rows: 4 -> 0

  Verifier evidence:
    - scoped verifier log had no MlxAdminPage findings
    - full verifier remains blocked by unrelated current-dev IntegrationPolicyPanel baseline drift/stale rows outside this slice
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
