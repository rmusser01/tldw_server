---
id: TASK-13245
title: Persist Workspace Persona opt-out and conversation provenance
status: To Do
assignee: []
created_date: '2026-09-13 18:15'
updated_date: '2026-09-13 18:22'
labels:
  - persona
  - workspaces
  - parity
dependencies:
  - TASK-13244
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2950'
documentation:
  - >-
    Docs/superpowers/plans/2026-09-13-persona-workspace-parity-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Distinguish unset defaults from an explicit None choice, preserve creation-time assistant provenance, and define opt-in server resolution without changing legacy chat-create behavior. Issue 2950 Stage 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Implement the linked stage contract with focused regression coverage and recorded verification; do not claim completion from documentation alone.
- [ ] #2 Explicit None survives restart and provisioning; opt-in inheritance persists bounded provenance atomically; stale versions and accepted retries cannot change identity.
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
