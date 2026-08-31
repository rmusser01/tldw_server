---
id: TASK-13147
title: Materialize Personal Context Sync domains
status: Done
assignee:
  - '@codex'
created_date: '2026-08-30 19:38'
updated_date: '2026-08-30 21:31'
labels:
  - personal-context
  - sync
  - security
dependencies:
  - TASK-13146
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
documentation:
  - Docs/Design/2026-08-30-personal-context-profile-server-design.md
  - IMPLEMENTATION_PLAN_personal_context_sync_transport.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate and materialize canonical Personal Context whole-object Sync envelopes through the server mutation authority so replicated records preserve identity, integrity, lineage, privacy, and purge fences.
<!-- SECTION:DESCRIPTION:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Server adapters validate all five canonical Personal Context transport domains, exact whole-object identity, schema v1, 16 KiB limit, HMAC integrity, purge generation, and base lineage with stable fail-closed outcomes; the delete-everywhere mutation lifecycle remains owned by approved plan Task 6.
- [x] #2 Accepted envelopes materialize through the authenticated `PersonalContextService` mutation boundary with optimistic concurrency; materializers never mutate canonical repository tables directly.
- [x] #3 Factory wiring advertises adapter/writable readiness only when all five adapters, materializers, key custody, schema, and server-trusted authorization are usable.
- [x] #4 Tombstones and purge barriers are content-free, device-only records are rejected, idempotent retries do not duplicate canonical versions, and logs/results expose no raw profile body.
- [x] #5 Targeted adapter, materializer, Personalization regression, static, security, diff, and independent review gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing adapter and materializer tests for every domain, canonical byte/HMAC validation, size/schema/profile/scope/purge fences, exact lineage, authorization-before-decrypt, idempotent replay, conflict outcomes, and log redaction.
2. Implement one parameterized Personal Context domain adapter that validates transport invariants without mutating canonical storage.
3. Add an authenticated sync-apply method to `PersonalContextService`, then implement one parameterized materializer that calls it with expected lineage and maps service outcomes into Sync results.
4. Register the complete five-domain set in the factory only when the Personal Context service and key resolver are ready; retain fail-closed Task-1 capability advertisement otherwise.
5. Run targeted Sync and existing Personalization regressions, Ruff, compilation, Bandit, diff hygiene, independent review, and commit.

ADR required: no (existing)
ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
Reason: ADR-002 already governs canonical whole-object transport, server mutation authority, key custody, integrity, and purge fencing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added strict adapters and service-owned materializers for all five canonical Personal Context Sync domains with exact schema, identity, integrity, size, purge, and optimistic-lineage validation.
- Added authenticated `PersonalContextService` sync application, complete factory/readiness wiring, generic Sync-history at-rest encryption, content-free failures, and idempotent replay/conflict handling.
- Derived the Sync-history storage key from canonical profile custody so ordinary profile encryption-key rotation preserves stored-history readability; missing or mismatched canonical profiles now fail closed without leaking `KeyError`.
- Verification: 87 targeted Personalization/Personal Context Sync tests and 8 capability/model tests passed; Ruff, Python compilation, Bandit, `git diff --check`, and independent review passed with no remaining actionable blocker.
- Known skips: the full repository suite was not run per repository policy. Wrapped-key bootstrap/distribution remains approved plan Task 3; delete-everywhere mutation lifecycle remains Task 6.
- ADR: existing `backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md` applies; no new ADR was required.
<!-- SECTION:NOTES:END -->
