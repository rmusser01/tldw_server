---
id: TASK-13167
title: Harden legacy Personal Context receipt backfill
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-04 03:32'
labels:
  - personal-context
  - sync
  - security
  - relay
dependencies:
  - TASK-13160
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
Remediate TASK-13161 by permitting legacy empty wire-identity backfill only after the stored receipt and decrypted canonical source prove the complete historical publication identity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An empty legacy wire identity is accepted only after exact comparison of every old receipt fact plus decrypted source domain, role, operation, batch size, canonical digest, profile, generation, object, internal version, manifest revision, wire version, batch, and sequence.
- [ ] #2 Validation compares the current authoritative manifest and journal binding before any receipt mutation.
- [ ] #3 Backfill is one transactional compare-and-set from empty only, checks the affected row count, and every subsequent use follows the strict modern identity path.
- [ ] #4 Any mismatch, corrupt ciphertext, changed key, or stale manifest leaves the receipt unchanged and performs no mutation, acknowledgement, or replay.
- [ ] #5 Real SQLite tests cover matching legacy rows and record, scope, manifest, pending-state, terminal-state, corrupt-ciphertext, and changed-key mismatches.
- [ ] #6 Errors, logs, persistence metadata, and test diagnostics remain content-free for protected profile data.
- [ ] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs publication identity and receipt integrity.
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
