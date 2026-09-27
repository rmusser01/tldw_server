---
id: TASK-13248
title: Provision Workspace default Personas with retry-safe ownership
status: To Do
assignee: []
created_date: '2026-09-13 18:20'
labels:
  - persona
  - workspaces
  - parity
dependencies:
  - TASK-13245
  - TASK-13246
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2950'
documentation:
  - >-
    Docs/superpowers/plans/2026-09-13-persona-workspace-parity-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add per-owner idempotent Persona and permission-profile provisioning and reviewable backfill after opt-out storage and tool-profile contracts are in place. Preserve explicit None and existing chats. Issue 2950 Stage 4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Concurrent or retried provisioning creates one owned resource pair; clear, archive, and explicit None win over delayed binding; backfill is reviewable and owner-scoped.
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
