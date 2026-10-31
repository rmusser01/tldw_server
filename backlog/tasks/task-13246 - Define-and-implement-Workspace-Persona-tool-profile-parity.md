---
id: TASK-13246
title: Define and implement Workspace Persona tool-profile parity
status: To Do
assignee: []
created_date: '2026-09-13 18:15'
updated_date: '2026-09-13 18:22'
labels:
  - persona
  - workspaces
  - parity
dependencies:
  - TASK-13245
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1922'
documentation:
  - >-
    Docs/superpowers/plans/2026-09-13-persona-workspace-parity-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconcile Chatbook string/local permission profiles with MCP Hub-owned server profiles and authorization. Finalize the focused contract under issue 1922 before enabling non-null Workspace references; implement narrowing-only enforcement and revocation tests. Issue 2950 Stage 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Implement the linked stage contract with focused regression coverage and recorded verification; do not claim completion from documentation alone.
- [ ] #2 Approve the server MCP Hub binding contract before activation; validate owner/profile authority and prove deny, ask, cap, and revocation enforcement on affected tool paths.
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
