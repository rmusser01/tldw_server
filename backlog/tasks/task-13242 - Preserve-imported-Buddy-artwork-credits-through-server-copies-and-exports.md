---
id: TASK-13242
title: Preserve imported Buddy artwork credits through server copies and exports
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-10 14:16'
updated_date: '2026-09-10 14:45'
labels:
  - buddy
  - persona
  - portability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live Trenchcoat import on dev 50c1f68957 succeeds but discards embedded creator, source URL and Apache-2.0 notices. Both the independent Buddy copy and a native re-export lose this metadata. Preserve authored artwork credits while treating them as untrusted content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A native pack import retains validated bounded artwork creator, source and license notices from the supported source_context.artwork carrier.
- [x] #2 Independent Buddy copying and native export/re-import preserve the artwork metadata without depending on the source Persona.
- [x] #3 Malformed, oversized and unknown metadata cannot become executable policy or leak unrelated source context; existing packs without artwork metadata remain compatible.
- [x] #4 A failing regression reproduces the observed loss, and real Trenchcoat import/copy/export verifies the repair on SQLite and supported PostgreSQL storage paths.
- [x] #5 PostgreSQL native export serializes stored timestamp values so the credited import/copy/export journey can complete on both supported database backends.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/006-buddy-artwork-credit-portability.md
Reason: Optional bounded metadata bridge extends ADR-005 ownership, with no migration.

1. Reproduce credit loss with real SQLite preview/import/copy/export and invalid metadata cases (done red).
2. Retain validated credits in existing manifest storage. Restore only the native carrier on export, remove the internal key from the exported copy, and include artwork in export fingerprints. Copy independent attribution; ignore unrelated context.
3. Verify the full SQLite journey and supported PostgreSQL snapshot/export paths. PostgreSQL import-job metadata is explicitly unsupported by the existing repository; do not expand that boundary. Normalize exported datetime fields uncovered by the PostgreSQL regression.
4. Run focused regression, lint/format and Bandit. Repeat public Trenchcoat through the authenticated HTTP worker and validate its export with actual Chatbook native code.
5. Record source-bound evidence, native/voice qualification gaps, review findings and resolution, and re-import recovery guidance; publish the focused dev PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced 2026-09-10 using authenticated HTTP plus the standard Persona portability worker in a disposable profile. Original archive SHA256620f06958112d9be1dce0d6592842f47ada9c8c4d9f359c0ed92d878af4ab80f. Import preview and commit completed; independent POST /api/v1/buddies returned201; exported archive downloaded200. Creator absent in copied attribution and all exported metadata. Evidence under /private/tmp/buddy-v1-followup-n4shbp3q/attribution-result.json. Related ADR005 requires independent artwork attribution ownership; persistence design must be checked before implementation.

Focused run: 111 passed; PostgreSQL journey reproduced a pre-existing native export failure because metadata pack/asset timestamp values are datetime objects. Repair will normalize only the exported timestamp fields to ISO text, preserving SQLite string behavior and the existing storage schema. SQLite published Trenchcoat HTTP import/copy/export/re-import now preserves the exact 11,654-byte notices and original PNG bytes.

PostgreSQL portability import-job storage explicitly raises NotImplementedError (SQLite-only). Revised qualification separates SQLite full import/commit journey from supported PostgreSQL Buddy snapshot/export/preview. Review also found strict native Chatbook manifest rejection of the internal credit key; export now strips that key from a copy, restores source_context.artwork, and explicitly fingerprints credits. No storage/runtime boundary expansion.

Implemented ADR-006: native credit validation/storage, independent attribution snapshots, native-compatible export with credit-sensitive fingerprints, and narrow PostgreSQL timestamp normalization. Resolved both independent-review findings (Chatbook strict-root compatibility and legacy credit-free context). Final portability:45 passed including supported PostgreSQL copy/export; adjacent ownership/manifest/asset tests:72 passed. Ruff/Black passed; Bandit zero findings. Final published Trenchcoat HTTP roundtrip and actual Chatbook importer preserve exact credits and PNG bytes. See Docs/Reviews/2026-09-10-buddy-followup.md and its source-hashed receipt. PostgreSQL import jobs remain unsupported; native terminal/installed extension and physical voice qualification stay open. PR review/merge pending.

Published PR https://github.com/rmusser01/tldw_server/pull/2940 against freshly fetched dev50c1f68957. Implementation commit0e72f25515, no behind-dev commits at publication. Collection installation guidance in tldw-stuff PR18 links the source-bound verification and recovery instructions. Task remains In Progress pending PR review/merge.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
