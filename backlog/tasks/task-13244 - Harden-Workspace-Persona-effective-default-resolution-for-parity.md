---
id: TASK-13244
title: Harden Workspace Persona effective-default resolution for parity
status: To Do
assignee: []
created_date: '2026-09-13 18:15'
updated_date: '2026-09-13 18:21'
labels:
  - persona
  - workspaces
  - parity
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2950'
documentation:
  - >-
    Docs/superpowers/plans/2026-09-13-persona-workspace-parity-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add privacy-safe degraded responses and logging, preserve malformed-storage diagnostics, and repair the obsolete v48 migration fixture identified during TASK-13243. First backend slice of issue 2950.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Implement the linked stage contract with focused regression coverage and recorded verification; do not claim completion from documentation alone.
- [ ] #2 Permission-denied effective results redact identity; invalid payloads do not leak into logs; corrupt storage retains an invalid-default diagnostic; repaired migration tests retain meaningful upgrade coverage.
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
