---
id: TASK-13150
title: Attest coherent Reading List snapshot pages
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-01 14:26'
updated_date: '2026-09-01 14:37'
labels:
  - collections
  - reading-list
  - pagination
dependencies: []
references:
  - TASK-18919 (tldw_chatbook)
documentation:
  - >-
    tldw_chatbook:Docs/superpowers/plans/2026-08-31-library-collections-capture-reader.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Reading List page totals, rows, and tag hydration come from one database snapshot so clients can rely on exact paging, then expose that shipped guarantee through docs-info.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A concurrent writer cannot produce a mixed total/page result.
- [ ] #2 Tag hydration uses the same snapshot as count and rows.
- [ ] #3 Docs-info exposes hasReadingSnapshotPagesV1=true in capabilities and supported_features.
- [ ] #4 The existing Reading List endpoint and response shape remain unchanged.
- [ ] #5 Focused tests and touched-scope security checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a controlled concurrent-writer regression and witness the pre-fix mismatch.
2. Reuse the existing database transaction and connection plumbing for count, rows, and tags.
3. Add a failing docs-info capability test, then one literal attestation entry.
4. Run focused regressions, Bandit, and diff checks; document evidence.

ADR required: no
ADR path: N/A
Reason: This is a bounded correctness fix and capability attestation for an existing Reading List service contract; it introduces no new endpoint, schema, storage owner, or runtime boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 1 complete. Baseline: existing list selection passed. RED evidence: controlled writer produced total 21 with rows beginning at newly committed ID 22; connection test observed None for count/page/tag; PostgreSQL-mode test observed no repeatable-read request. GREEN evidence: all 4 list/snapshot-focused tests pass after reusing the existing transaction, passing one connection through count/page/tag hydration, and requesting REPEATABLE READ READ ONLY for PostgreSQL. Plan correction: generic PostgreSQL transactions are READ COMMITTED, so the focused isolation statement was required before the capability can truthfully be server-wide.

Stage 2 complete. RED: the focused docs-info test failed with KeyError for hasReadingSnapshotPagesV1. GREEN: one literal capability entry now appears identically in capabilities, supported_features, and the endpoint response; the exact test passes, all 17 docs-info capability tests pass, and the 4 list/snapshot tests remain green. No endpoint or response shape changed.
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
