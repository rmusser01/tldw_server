---
id: TASK-13148
title: Bootstrap Personal Context canonical profile
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-30 22:21'
updated_date: '2026-08-30 23:00'
labels:
  - personal-context
  - sync
  - security
dependencies:
  - TASK-13147
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
documentation:
  - Docs/superpowers/plans/2026-08-28-personal-context-04-sync-multidevice.md
  - IMPLEMENTATION_PLAN_personal_context_bootstrap.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose an authenticated, cursor-consistent Personal Context bootstrap that serializes first-link ownership, returns canonical manifest/scopes/heads with wrapped integrity-key material, and prevents pre-reconciliation uploads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The server serializes first-link profile ownership and returns the canonical manifest, scopes, object heads, purge generation, and one consistent bootstrap cursor for the authenticated user.
- [ ] #2 Bootstrap distributes the server-owned integrity key only to an authenticated registered device using the existing wrapped Sync key-record path; plaintext key material never enters logs, diagnostics, or durable bootstrap metadata.
- [ ] #3 Pre-reconciliation Personal Context uploads fail closed, retries are idempotent, and mismatched user, device, schema, quota, or purge generation produce stable content-free outcomes.
- [ ] #4 The bootstrap contract supports Chatbook reviewed reconciliation and full integrity rebaseline without making Sync history the canonical profile authority.
- [ ] #5 Targeted bootstrap, Sync, and Personalization tests plus Ruff, compilation, Bandit, diff hygiene, and independent review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED tests for authenticated cursor-consistent bootstrap, registered-device wrapping, idempotency, and the pre-reconciliation upload fence.
2. Serialize first-link ownership and return canonical manifest, scopes, object heads, purge generation, quotas, and one bootstrap cursor through the profile service boundary.
3. Deliver the server-owned integrity key through the existing Sync key-record enrollment/rewrap path without plaintext persistence.
4. Enforce stable content-free failures for user/device/schema/quota/generation mismatches and keep Sync history non-authoritative.
5. Run targeted tests, Ruff, compilation, Bandit, diff hygiene, independent review, and commit.

ADR required: no (existing)
ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
Reason: ADR-002 already governs canonical server ownership, key custody, whole-object Sync transport, and the service boundary used by bootstrap.
<!-- SECTION:PLAN:END -->

## Progress

- [x] Added the bootstrap contract before production implementation, including
  canonical heads/cursor, registered-device wrapping, compatibility failures,
  pre-link fencing, completion, and plaintext-canary coverage.
- [x] Implemented the server-owned bootstrap and completion boundary with
  canonical Personal Context reads and opaque Sync dataset/key-record state.
- [x] Ran focused SQLite-backed Sync and Personal Context regressions plus
  Python 3.11 compilation and diff hygiene; results are recorded in the slice report.
- [ ] Controller verification remains: PostgreSQL-specific coverage, endpoint/model
  collection dependencies, Ruff/Bandit availability, independent review, and final
  task closure. This task remains In Progress.

## Review round 1 progress

- [x] Hardened factory custody with registered-device RSA-OAEP wrapping and
  server-owned authority selection; added typed API bootstrap/completion routes.
- [x] Reserved Personal Context enrollment metadata/domains and made completion
  receipts device-specific so one device cannot unlock another's pushes.
- [ ] Dedicated canonical snapshot transaction and Sync-store compare-and-set
  receipt persistence remain under review. TASK-13148 remains In Progress.

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
