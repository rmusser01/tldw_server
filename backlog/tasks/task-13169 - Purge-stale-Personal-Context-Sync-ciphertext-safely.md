---
id: TASK-13169
title: Purge stale Personal Context Sync ciphertext safely
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 03:34'
updated_date: '2026-09-04 14:49'
labels:
  - personal-context
  - sync
  - security
  - retention
dependencies:
  - TASK-13168
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
Remediate TASK-13161 by binding cryptographic cleanup of stale Personal Context Sync rows to the authenticated explicit Delete Entire Profile path while keeping pull and relay recovery non-destructive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Authenticated explicit direct full-profile purge cryptographically shreds or otherwise makes unrecoverable every old-generation Personal Context Sync authority and ingress payload and wrapped data key, including pending, orphaned, superseded, and history-key-protected rows.
- [x] #2 Pull, relay recovery, compaction, listing, and ordinary mutation never trigger destructive profile cleanup.
- [x] #3 Content-free tombstone and generation-fence evidence remains sufficient for retry and audit while current-generation data, other profiles, and other datasets are unaffected.
- [x] #4 Cleanup is idempotent, restartable after partial failure, and covers retained backup artifacts within the documented profile-purge boundary.
- [x] #5 Before explicit purge, non-destructive scans skip stale-generation rows without exposing them or allowing an orphaned pending row to block eligible current-generation delivery.
- [x] #6 Real SQLite tests and DB, WAL, backup, and key-rotation canaries prove old ciphertext cannot be recovered after authorized purge.
- [x] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs direct-purge-only cryptographic shredding.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED direct-purge authorization and ciphertext-retention tests. 2. Journal cleanup authority only in the confirmed direct purge transaction. 3. Shred old-generation Sync payload/key material idempotently and repair barriers. 4. Keep pull and relay non-destructive. 5. Inventory application-owned backup boundaries and verify canaries. 6. Self-review and close the task. 7. Review round 2: reproduce direct store/database duck-type execution, capability field retargeting, subclass spoofing, and expired/completed lease loss at every destructive layer; replace caller-controlled validation with one exact-type, tamper-evident, live-journal validator used by service, store, and database; rerun focused retention/recovery/publication/service and static checks. ADR required: no new ADR; ADR-002 and the approved spec govern.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a content-free, constrained Personalization cleanup-intent journal minted atomically only by authenticated direct full-profile purge. Claim, release, and completion are retry-safe and owner-fenced. Broad cleanup now requires an exact internal, immutable repository-issued capability whose private provenance and HMAC tag bind the authenticated repository, source database, user, dataset, Sync store/database, intent, profile, generations, owner, and unexpired live journal claim. Service, store, and database all use the same module-private validator; duck types, subclasses, public intents, field retargeting, and caller-defined validation methods are not executable.
- Added exact dataset/profile/generation-scoped SQLite cryptographic shredding for old Personal Context authority, ingress, orphan/superseded state, conflicts, receipts, wrapped DEKs/nonces/ciphertext, and all associated device-wrapped integrity-key rotations. Applied acknowledgement/fence evidence remains content-free; stale heads are repaired; unrelated rows are preserved.
- Wired active and archived Sync datasets in both authenticated API and Sync factory assembly. Remote/client/future-signed purge, pull, relay, compaction, listing, and ordinary mutation never mint or execute this cleanup. After failure or restart, only the same authenticated endpoint request with exact `DELETE EVERYWHERE` and the prior expected generation can reclaim it; wrong/different requests reject.
- Deleted old canonical ingress digests in the purge transaction through exact publication batch/profile/generation/result/manifest lineage and checked one-row destructive predicates. Before logical deletion, both canonical and Sync SQLite stores require verified `secure_delete=ON` and WAL mode. Completion additionally requires `VACUUM`, an empty freelist, exact successful WAL truncation, and an absent/empty WAL for the canonical store and every affected Sync database.
- Inventoried and re-audited active DB/WAL/SHM, in-place migration, managed backup, generated snapshot/export, master-key, canonical profile-key, and Sync key-record custody. The application has no managed Personalization/Sync backup target; operator-created backups and previously exported recovery bundles are explicitly outside the guarantee. SQLite is proven; non-SQLite cleanup fails closed/pending and would require a separately reviewed retention policy.
- Real-SQLite RED reproductions covered all four round-one findings and the remaining round-two lower-layer capability bypass. Final focused purge-retention, relay-recovery, publication, and service verification passed: `121 passed, 6 warnings`. Ruff passed all touched Python files; Bandit exited 0 with only parser/accepted `nosec B608` warnings and no findings; `git diff --check` passed.
- Detailed authorization, artifact inventory, destructive-predicate review, and canary evidence are recorded in `.superpowers/sdd/2026-09-04-personal-context-relay-remediation/task-4-report.md`. ADR required: no new ADR; ADR-002 and the approved ongoing-sync specification govern. `ongoing_sync_version` remains unchanged. No full suite was run, per the task brief.
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
