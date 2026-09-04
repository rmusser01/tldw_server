---
id: TASK-13171
title: Enforce active Personal Context exchange without breaking Sync V2
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-04 03:34'
labels:
  - personal-context
  - sync
  - security
  - api
dependencies:
  - TASK-13170
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
Remediate TASK-13161 by applying one precise active-exchange proof gate to Personal Context Sync V2 operations while preserving unrelated legacy and mixed-domain behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One service gate protects Personal Context version-one push, pull, conflict listing, and conflict resolution with the exact persisted activation epoch and token plus a completed link receipt for the requesting device.
- [ ] #2 A Personal Context version-zero ongoing exchange returns activation_required before mutation, delivery, or cursor advancement.
- [ ] #3 Legacy first-link flows and operations containing no selected Personal Context data continue to work.
- [ ] #4 A mixed Personal Context and Notes dataset can list and resolve unrelated Notes conflicts without Personal Context activation proof.
- [ ] #5 Personal Context conflict listing uses the real requesting device identity and completed link receipt, and conflict resolution gates from the selected conflict identities rather than unrelated dataset contents.
- [ ] #6 Only verified stored proof is echoed; exact, stale epoch, stale token, missing proof, incomplete link, tampered proof, and wrong-device cases have distinct verified outcomes.
- [ ] #7 Production TestClient tests cover push, pull, conflict list, and conflict resolve across Personal Context-only, mixed-domain, legacy, missing, stale, incomplete-link, tampered, and exact-proof cases.
- [ ] #8 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs activation proofs and compatibility.
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
