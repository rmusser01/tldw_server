---
id: TASK-13172
title: Certify Personal Context relay and close TASK-13161
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-04 03:35'
labels:
  - personal-context
  - sync
  - security
  - certification
dependencies:
  - TASK-13166
  - TASK-13167
  - TASK-13168
  - TASK-13169
  - TASK-13170
  - TASK-13171
references:
  - >-
    backlog/tasks/task-13161 -
    Relay-ordered-Personal-Context-authority-publications-through-Sync-V2.md
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
  - Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prove the remediated encrypted Personal Context relay through the production server factory and close TASK-13161 only after its complete acceptance and quality evidence is independently verified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Production server-factory and TestClient flows prove direct canonical mutation, client ingress, authority publication, encrypted pull, and exact-once replay through real persistence boundaries.
- [ ] #2 After-commit relay failure is isolated from the accepted ingress response, durable debt survives restart, recovery runs before a later push or pull, and the authority result is published exactly once.
- [ ] #3 The crash, race, poison, receipt, exact-budget, activation-proof, conflict, and retention matrices from TASK-13166 through TASK-13171 pass together without weakening their assertions.
- [ ] #4 The single authoritative Personal Context dataset invariant is enforced and tested across configuration and runtime lookup.
- [ ] #5 No client-ingress row, hidden pending authority row, plaintext protected value, wrapped key, or content-derived diagnostic can leave its authorized boundary, and version-zero behavior remains compatible.
- [ ] #6 Targeted pytest suites, Ruff, Bandit, artifact scans, and git diff checks pass; the full repository suite is not run unless the user explicitly requests it.
- [ ] #7 TASK-13161 is marked Done only after independent specification and code-quality review accepts every original criterion and Definition-of-Done item; the final report, documentation, and any incident-backed lesson are updated.
- [ ] #8 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md remains the governing decision.
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
