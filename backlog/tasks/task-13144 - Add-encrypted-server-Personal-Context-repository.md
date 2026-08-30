---
id: TASK-13144
title: Add encrypted server Personal Context repository
status: Done
assignee:
  - '@codex'
created_date: '2026-08-30 16:23'
updated_date: '2026-08-30 17:35'
labels:
  - personal-context
  - security
  - backend
dependencies: []
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
documentation:
  - Docs/Design/2026-08-30-personal-context-profile-server-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend each user Personalization database with encrypted canonical Personal Context storage and explicit server-side key custody while preserving existing personalization behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A valid explicit 32-byte server master key creates and reopens per-profile encryption and integrity keys without silent replacement.
- [x] #2 Canonical manifest, scope, record, proposal, receipt, and runtime-policy versions are encrypted in the existing per-user Personalization database.
- [x] #3 Repository writes are transactional, optimistic, and profile-isolated with content-free tombstones and terminal receipts.
- [x] #4 The server consumes the exact tldw-profile-core 0.1.0 contract and proves schema, fixture, and canonical-byte parity.
- [x] #5 Plaintext canaries are absent from database, WAL, SHM, temporary storage, and exception text.
- [x] #6 Targeted repository and existing Personalization tests, lint, Bandit, compilation, and independent review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the approved architecture and exact Shared Core contract digest in the server design note and ADR.
2. Add RED key-custody, crypto, schema, repository, and no-plaintext tests.
3. Extend PersonalizationDB with canonical tables and an immediate transaction boundary.
4. Implement explicit master-key custody, envelope encryption, immutable repository operations, CAS heads, and encryption-key rotation.
5. Run targeted regressions, static/security checks, independent review, and commit.

ADR required: yes
ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
Reason: This establishes persistent encrypted storage, key custody, sync authority, conflict policy, and the cross-application contract boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the exact tldw-profile-core 0.1.0 snapshot and digest, encrypted immutable Personal Context persistence in each existing user Personalization database, explicit 32-byte master-key custody, per-profile wrapped keys, optimistic heads, content-free tombstones and receipts, fail-closed integrity checks, runtime policy storage, and encryption-key rotation. ADR: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md. Verification: 59 focused Personal Context plus existing Personalization tests passed; 15 CI workflow contracts passed; 30 Makefile/onboarding contracts passed; 13 final documentation parity tests passed; Ruff lint and format checks passed on the touched scope; compileall passed; Bandit passed with only the existing justified B608 nosec; TOML/YAML parsing, diff hygiene, and wheel metadata/content inspection passed. Independent review completed clean after fixes for orphaned key state, schema-version AAD binding, and Python 3.11 floor consistency. The full repository suite was not run because project instructions require explicit opt-in; no known feature blockers remain.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the encrypted canonical server repository that makes the shared Personal Context record model durable and sync-ready without storing profile plaintext. The server wheel now ships the exact shared schemas and fixtures and consistently requires Python 3.11 across metadata, CI, executable install gates, and active setup documentation.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
