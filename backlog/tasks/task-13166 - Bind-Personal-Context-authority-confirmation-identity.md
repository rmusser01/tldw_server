---
id: TASK-13166
title: Bind Personal Context authority confirmation identity
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 03:29'
updated_date: '2026-09-04 05:36'
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
- [x] #1 Deterministic authority-row reuse compares the complete normalized immutable envelope fingerprint including base lineage, object revision, stable identity, client metadata, dependency and mutation-group fields, schema and adapter versions, delete state, encryption and routing metadata, authority metadata, and canonical payload identity.
- [x] #2 The internal ingress-to-authority confirmation path accepts only a current pending or applied client_ingress envelope bound to the exact canonical receipt identity and rejects every mismatched lineage, device, envelope, digest, generation, object, version, manifest, batch, or sequence fact.
- [x] #3 Ordinary Personal Context lineage and current-head CAS validation remains unchanged outside the narrow trusted confirmation path.
- [x] #4 Real Personalization plus Sync SQLite tests prove new and updated record publication, repeated manifest publication, deterministic retry, and rejection of tampered persisted pending rows without source acknowledgement or durable poison.
- [x] #5 Errors, logs, persistence metadata, and test diagnostics remain content-free for protected profile data.
- [x] #6 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs authority identity and deterministic relay.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED identity and tamper tests. 2. Persist and read the exact ingress receipt. 3. Compare the complete immutable authority fingerprint and canonical receipt binding. 4. Run targeted security and regression checks. 5. Self-review and close the task. ADR required: no new ADR; ADR-002 governs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a canonical HMAC attestation over every stable stage-derived authority envelope field, the authenticated payload digest, source identity, and any originating cross-store ingress receipt. One complete verifier is reused for deterministic staging and finalization; final verification and pending/applied CAS now share one guarded Sync transaction. Randomized AEAD bytes remain verified by restore rather than regenerated for equality.
- Reads the Sync receipt by exact cursor first, then reads the Personalization receipt through its `(dataset_id, device_id, client_envelope_id)` primary key and verifies the exact batch, source, and manifest facts. Semantic, companion-manifest, and purge-barrier authorities all retain the correct originating receipt identity; purge wire versions compare in canonical string form.
- Added real temporary Personalization and Sync SQLite coverage for persisted authority-envelope mutations, first-stage and post-stage receipt mutations, staged relay-resume tampering, semantic and manifest companions, exact purge confirmation, duplicate same-batch receipts, deterministic retry, and the existing new/updated/repeated publication paths. Tamper failures remain retryable and neither apply nor acknowledge source rows or create durable poison.
- Review round 2 requires both receipts before first-stage companion-manifest signing whenever its acknowledged semantic/purge authority has a client-ingress base. The same existing exact-ingress predicate now validates the origin envelope during first stage, deterministic reuse, and finalization; missing or mismatched receipts fail retryably before insertion.
- Review round 3 classifies a companion manifest's exact origin base before receipt resolution. An absent base or a validated applied same-object `home_authority` base is a direct receipt-free publication; only `client_ingress` enters the strict two-store receipt verifier, while unknown or malformed authority/base relationships fail retryably.
- Verification: targeted pytest `71 passed, 7 warnings`; Ruff passed with `--no-cache` after the sandbox denied the worktree cache write; Bandit exited 0 with existing comment/nosec parser warnings; `git diff --check` passed. The pytest warnings/log noise are existing environment configuration issues (cache permission, isolated `USER_DB_BASE_DIR`, legacy test API-key format, and `system_log_buffer` permission noise), not new product warnings.
- Modified `service.py`, `store.py`, `Sync_DB.py`, `personal_context_publication.py`, and `test_sync_v2_personal_context_authority_identity.py`. Ordinary `_evaluate_envelope` and current-head CAS logic were not changed. No new ADR was required; ADR-002 governs. No full suite was run per repository policy, and no blockers remain.
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
