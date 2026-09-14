---
id: TASK-13150
title: Attest coherent Reading List snapshot pages
status: Done
assignee:
  - '@codex'
created_date: '2026-09-01 14:26'
updated_date: '2026-09-01 15:39'
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
- [x] #1 A concurrent writer cannot produce a mixed total/page result.
- [x] #2 Tag hydration uses the same snapshot as count and rows.
- [x] #3 Docs-info exposes hasReadingSnapshotPagesV1=true in capabilities and supported_features.
- [x] #4 The existing Reading List endpoint and response shape remain unchanged.
- [x] #5 Focused tests and touched-scope security checks pass.
- [x] #6 Punctuation-heavy Reading search is treated as natural language and does not surface an FTS parser error.
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

Stage 3 verification exposed a pre-existing punctuation-search defect: the complete touched Collections test file failed identically on this branch and a detached origin/dev worktree because `C++/Rust: Intro? [Guide]` reached SQLite FTS as raw syntax. The existing regression was the RED evidence. The Collections natural-language query builder now extracts Unicode word tokens, quotes each prefix term, and only permits raw mode for unmistakable FTS syntax; the regression is GREEN. Both focused files pass (42 tests), touched production Bandit passes, and both cumulative and working-tree diff checks pass. Whole-file Black/Ruff remain a documented baseline rather than an expanded rewrite: all four legacy files would be reformatted, and Ruff reports unrelated pre-existing issues including undefined `global_settings` in config_info.py and longstanding Collections warnings. Changed hunks were reviewed directly for style.

Independent review then found that the first snapshot implementation requested PostgreSQL isolation after pool checkout had already run scope-setting queries, used SQLite's writer-reserving `BEGIN IMMEDIATE`, left capability maps absent on docs-info fallback paths, and still allowed some punctuation into raw FTS mode. Each finding was verified against the concrete backend lifecycle and received a failing regression. The correction commits the PostgreSQL scope setup before beginning `REPEATABLE READ READ ONLY`, uses `BEGIN DEFERRED` for SQLite, always publishes the shipped capability, and treats all public search text as natural language. GREEN evidence: 47/47 touched-file tests and 21/22 adjacent FTS/watchlist consumer tests pass with one pre-existing optional skip.

Final PostgreSQL lifecycle review found that Psycopg could still emit a plain implicit `BEGIN` before the requested isolation statement. The final correction commits scope setup, temporarily enables autocommit, issues raw `BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY`, rolls back the snapshot, restores autocommit, and always returns the connection. Cleanup failures are logged without replacing a primary operation failure. Final evidence: 47/47 touched-file tests pass in a complete ephemeral dependency environment; selected Ruff, `py_compile`, touched production Bandit, working-tree diff, and cumulative `origin/dev...HEAD` diff checks pass. Independent re-review approved the cumulative branch with no findings. No new ADR was required because this remains a bounded correctness repair to an existing service contract.

The downstream TASK-18919 live walkthrough then exposed a separate SQLite bootstrap problem: the Collections schema-memo verifier treated SQLiteBackend mapping rows as positional rows, raised `KeyError`, and replayed the full schema on every adapter construction. A focused regression reproduced the extra bootstrap; reading the existing named `name` field makes the second adapter reuse the memo. The two-test bootstrap gate and scoped Ruff/diff checks pass. This bounded integration correction does not change the ADR determination or public API.

Post-rebase review verified and corrected two additional boundary cases. A list read nested inside a caller-owned SQLite write transaction now reuses that transaction without beginning or rolling it back, and docs-info derives the Reading snapshot flag from the enabled route while publishing dynamic capabilities atomically. The concurrent-writer regression explicitly establishes WAL, lifecycle cleanup is exercised through the public listing API, and the new tests carry the required unit classification, annotations, and docstrings. The complete three-file review gate passes 51 tests; changed tests and production rules outside documented `dev` baselines pass scoped Ruff and diff checks.
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
