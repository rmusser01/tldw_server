---
id: TASK-13247
title: Adopt Workspace Persona defaults in Research chat and verify parity
status: To Do
assignee: []
created_date: '2026-09-13 18:15'
updated_date: '2026-09-13 18:22'
labels:
  - persona
  - workspaces
  - parity
dependencies:
  - TASK-13248
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2950'
documentation:
  - >-
    Docs/superpowers/plans/2026-09-13-persona-workspace-parity-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After backend gates pass, implement separately reviewed personal Research Workspace startup integration across normal and RAG modes, then verify the cross-project parity matrix. Buddy and design-system backlog work are excluded. Issue 2950 Stage 5.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Implement the linked stage contract with focused regression coverage and recorded verification; do not claim completion from documentation alone.
- [ ] #2 Normal and RAG startup preserve Persona identity and provenance with selected-source restrictions; refreshed cross-project evidence accounts for every parity row.
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
