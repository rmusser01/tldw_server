---
id: TASK-13159
title: Version Personal Context ongoing-sync wire contract
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-03 13:39'
labels:
  - personal-context
  - sync
  - contract
  - security
dependencies: []
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
  - Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the server-owned version-1 wire contract for ongoing Personal Context synchronization so Chatbook and future clients can validate capability, activation, authority-role, conflict, cleanup, purge, continuation, and endpoint messages without copying server internals. This task adds models, generated artifacts, and fail-closed route surfaces only; it does not advertise ongoing_sync_version=1 or enable production synchronization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Strict server models cover activation epoch and continuity token on every version-1 push, pull, conflict-list, and conflict-resolution exchange.
- [ ] #2 Authority envelope role, publication identity, relay continuation, conflict candidate IDs, cleanup acknowledgments, and purge generation have bounded versioned schemas.
- [ ] #3 A deterministic checked-in contract artifact carries its schema version, server source commit, and checksum and is suitable for exact vendoring by Chatbook.
- [ ] #4 Valid and invalid fixtures prove unknown-field compatibility, strict required-field validation, bounded values, and rejection of client-spoofed home-authority metadata.
- [ ] #5 The server continues advertising ongoing_sync_version=0 until later readiness work enables the complete implementation.
- [ ] #6 New activation-acknowledgment and signed Sync-purge route surfaces return a bounded capability-unavailable response until their owning implementation tasks land.
- [ ] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs the contract.
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
