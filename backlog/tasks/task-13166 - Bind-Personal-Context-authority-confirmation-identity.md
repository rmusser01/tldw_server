---
id: TASK-13166
title: Bind Personal Context authority confirmation identity
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-04 03:29'
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
Remediate TASK-13161 by requiring deterministic home-authority replay and client-ingress confirmation to prove the complete immutable Sync lineage and originating canonical receipt before any authority row can be finalized.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Deterministic authority-row reuse compares the complete normalized immutable envelope fingerprint including base lineage, object revision, stable identity, client metadata, dependency and mutation-group fields, schema and adapter versions, delete state, encryption and routing metadata, authority metadata, and canonical payload identity.
- [ ] #2 The internal ingress-to-authority confirmation path accepts only a current pending or applied client_ingress envelope bound to the exact canonical receipt identity and rejects every mismatched lineage, device, envelope, digest, generation, object, version, manifest, batch, or sequence fact.
- [ ] #3 Ordinary Personal Context lineage and current-head CAS validation remains unchanged outside the narrow trusted confirmation path.
- [ ] #4 Real Personalization plus Sync SQLite tests prove new and updated record publication, repeated manifest publication, deterministic retry, and rejection of tampered persisted pending rows without source acknowledgement or durable poison.
- [ ] #5 Errors, logs, persistence metadata, and test diagnostics remain content-free for protected profile data.
- [ ] #6 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs authority identity and deterministic relay.
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
