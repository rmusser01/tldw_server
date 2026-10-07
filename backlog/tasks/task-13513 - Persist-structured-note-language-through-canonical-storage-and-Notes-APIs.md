---
id: TASK-13513
title: Persist structured note language through canonical storage and Notes APIs
status: In Progress
created_date: 2026-10-07 02:23
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-foundations.md
updated_date: 2026-10-07 04:05
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make a note's selected source language durable and portable without losing source text or weakening version conflicts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Nullable content_language survives create, read, list, duplicate, versions and backup restoration without changing note source.
- [x] #2 Language-only changes use optimistic note versioning; omission preserves, explicit null resets and bounded unknown future values survive.
- [x] #3 Fresh and existing SQLite schemas migrate correctly; server PostgreSQL DDL and lifecycle checks retain parity.
- [x] #4 An authenticated capability route truthfully reports complete language write support and rejects mutations unsafe under active Sync before persistence.
- [x] #5 Real storage and API tests pass with scoped static and security evidence; persisted-empty-note contract remains unchanged.
- [x] #6 Accepted bounded nonblank titles and source content retain their exact characters in canonical writers and API reads; validation may reject invalid values but never trim, escape or truncate accepted snapshots.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: Docs/ADR/065-portable-note-language-and-negotiated-structured-editing.md (extends locked ADR-031)
Reason: additive portable metadata and negotiated future clear/private snapshots under approved Chatbook ADR-194.
1. Trace canonical writers/readers and current copy/version/backup owners.
2. Add failing real storage/API omission/null, conflict and absent-column migration checks.
3. Extend C73/S74/PG78 schema heads additively and ordinary API contracts; keep active Sync safely gated until F3.
4. Verify targeted SQLite and owned PostgreSQL paths, generated API types, scoped static comparison and Bandit.
5. Self-review and commit F2 repo-local units; preserve In Progress status pending independent review.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Persisted nullable opaque content_language through canonical rows, ordinary APIs, exact SQLite sync_log snapshots and SQLite backups. Omission/null and optimistic version behavior are explicit; accepted bounded titles/content remain exact. Active Sync temporarily rejects incomplete metadata-bearing upserts/restores; locked notes.note v1 stays unchanged. ADR: Docs/ADR/065-portable-note-language-and-negotiated-structured-editing.md.

Verified genuine absent-column C73→74/S74→75/PG78→79 upgrades and old/current SQLite backup restoration. Required C feature/scope/client96 passed, targeted scaffold/atomicity/v7396 passed, corrected census6 passed. S required initial new/API45 passed; final SQLite feature26 passed. Owned PostgreSQL required run96 passed/one historical fixture seed incompatibility; corrected historical SQL and final storage/routes/upgrade/lifecycle rerun4 passed. Final explicit-null interop regression1 passed. Existing historical C failures were reproduced at immutable base and are documented separately, not presented as green.

Legacy static comparison introduces no Ruff findings (C1690 inherited; S0); touched server Bandit0; new files clean and changed-line formatter debt preserved. Generated API types via existing generator; frontend typecheck passed. Source bound is null or1..100 opaque characters, no normalization. Canonical backup support remains SQLite-only; fetched-record copy is proven, UI copy/history propagation remains its later owner. Full commands/results, actual files, baseline controls and limitations: .superpowers/sdd/2026-09-27-structured-document-editing/task-2-report.md in the Chatbook programme checkout. Remains In Progress for independent review; no push/merge.
F2 review round1: the cached language/version preflight could reject materialization only after an incomplete envelope had already been appended. ADR065 now records admission linearization at durable append. Ordinary, batch and inbound accepted-note capture share Sync→product locking, reread language and supplied expected versions, and commit Sync before releasing the read-only Notes lock. Existing materialization/replay remains outside; no distributed atomic-commit claim. PostgreSQL uses row locks and a short table write fence only for absent rows. The private expected-version condition is not serialized or hashed into locked v1 payloads. Detailed RED/GREEN and static evidence are appended to the programme task-2-report.md. Required original capture/batch selection69 passed; PUT/PATCH race + batch/inbound controls6 passed; real two-step failure retains applied prefix and durable failed envelope1 passed; owned PostgreSQL existing/absent-row lock controls2 passed. Keep In Progress for fresh review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
