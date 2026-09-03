---
id: TASK-13159
title: Version Personal Context ongoing-sync wire contract
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 13:39'
updated_date: '2026-09-03 20:11'
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
- [x] #1 Strict server models cover activation epoch and continuity token on every version-1 push, pull, conflict-list, and conflict-resolution exchange.
- [x] #2 Authority envelope role, publication identity, relay continuation, conflict candidate IDs, cleanup acknowledgments, and purge generation have bounded versioned schemas.
- [x] #3 A deterministic checked-in contract artifact carries its schema version, server source commit, and checksum and is suitable for exact vendoring by Chatbook.
- [x] #4 Valid and invalid fixtures prove unknown-field compatibility, strict required-field validation, bounded values, and rejection of client-spoofed home-authority metadata.
- [x] #5 The server continues advertising ongoing_sync_version=0 until later readiness work enables the complete implementation.
- [x] #6 New activation-acknowledgment and signed Sync-purge route surfaces return a bounded capability-unavailable response until their owning implementation tasks land.
- [x] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs the contract.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. Existing backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs the contract.
1. Add failing strict wire-contract and fail-closed endpoint tests.
2. Define bounded ongoing-sync models and extend the existing Sync V2 schemas without altering unrelated domains.
3. Register activation-acknowledgment and signed-purge routes that remain unavailable while version 0 is advertised.
4. Generate deterministic schema and provenance manifest artifacts from the committed contract source.
5. Run targeted model/endpoint tests, Ruff, Bandit for touched code, artifact reproducibility, and diff hygiene.
6. Complete acceptance criteria and implementation notes, then commit source and generated artifacts at their planned provenance boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Defined strict ongoing-sync contract models, fail-closed versioned routes, and deterministic schema/provenance artifacts. Targeted verification passed: 259 tests, Ruff, and Bandit.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
