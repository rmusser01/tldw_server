---
id: TASK-13004
title: Synchronize Notes relationships graph and restore lifecycle
status: Done
assignee: []
created_date: '2026-08-08 20:23'
updated_date: '2026-08-11 04:41'
labels:
  - notes
  - sync-v2
  - parity
  - graph
dependencies:
  - TASK-13003
references:
  - >-
    https://github.com/rmusser01/tldw_chatbook/blob/dev/backlog/decisions/046-synchronized-database-notes-parity.md
documentation:
  - >-
    https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs/Parity/2026-08-08-notes-server-capability-matrix.md
  - >-
    Docs/superpowers/specs/2026-08-10-notes-link-sync-and-graph-lifecycle-design.md
  - >-
    Docs/superpowers/plans/2026-08-10-notes-link-sync-and-graph-lifecycle-plan.md
  - Docs/ADR/037-canonical-notes-link-sync-and-derived-graph-projections.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add first-class Sync v2 ownership for explicit Notes relationships while keeping wikilink parsing, backlinks, and graph summaries deterministic projections, and complete the conflict-aware trash/restore lifecycle for synchronized notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Capabilities advertise a versioned `notes.link` domain with upsert and tombstone operations.
- [x] #2 Explicit relationship payloads preserve stable edge identity, source, target, type, label, properties, ownership, and optimistic base state across SQLite and PostgreSQL.
- [x] #3 Wikilinks, backlinks, orphan reports, and graph summaries are rebuilt deterministically from synchronized notes and explicit links rather than synchronized as mutable duplicates.
- [x] #4 Server-origin link mutations and note trash or restore mutations capture canonical envelopes when Sync v2 is active.
- [x] #5 Delete-versus-update, restore-versus-recreate, and concurrent edge edits yield idempotent outcomes or reviewable conflicts with restore-preview evidence.
- [x] #6 Graph and lifecycle endpoints remain paginated, bounded, and authorized for the selected dataset.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define the strict notes.link domain, payload, capability, provenance, and canonical identity contract.
2. After explicit v58 PostgreSQL lock/RLS approval, implement transactional schema migration and tenant isolation.
3. Implement the portable explicit-link lifecycle store and graph revision updates.
4. Add the notes.link adapter, materializer, lifecycle conflict rules, and exact repair semantics.
5. Add separate resumable notes_link_v1 enrollment/bootstrap for new and already-ready default-personal datasets.
6. Route active-Sync link APIs through canonical server-origin capture while preserving inactive legacy behavior.
7. Persist deterministic wikilink projections and bounded owner-scoped maintenance.
8. Make graph, backlink, orphan, cursor, and cache reads live-only, compatible, bounded, and revision-safe.
9. Integrate notes.link with restore preview, repair, and conflict resolution ordering.
10. Run boundedness, regression, security, static, documentation, review, and backlog-hygiene gates.

Operational approval: on 2026-08-10 the user explicitly approved the v58
PostgreSQL migration's `db_schema_version` row lock, transaction-held `ACCESS
EXCLUSIVE` locks on `notes`, `note_edges`, `chacha_keywords`, and
`note_keywords`, verified temporary suspension/restoration of `notes` FORCE RLS,
and fail-closed transactional rollback without product-row deletion or rekeying.

ADR required: yes
ADR path: Docs/ADR/037-canonical-notes-link-sync-and-derived-graph-projections.md
Reason: TASK-13004 changes durable Notes schema, Sync conflict/enrollment contracts, PostgreSQL ownership/RLS, public lifecycle APIs, and canonical-versus-derived graph authority.

Detailed executable plan: Docs/superpowers/plans/2026-08-10-notes-link-sync-and-graph-lifecycle-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented `notes.link` as the canonical Sync v2 representation for explicit
manual note-to-note relationships. Schema v58 adds strict portable link lifecycle
state, owner-scoped PostgreSQL RLS and migration validation, derived graph
projection state, and graph revisions. The Sync adapter, materializer, capture
coordinator, restore ordering, and separate resumable `notes_link_v1` bootstrap
preserve immutable edge identity and optimistic ancestry without coupling link
readiness to the six Notes organization domains. Active-Sync REST mutations use
canonical server-origin envelopes; inactive requests retain the legacy product
path. Incident links remain durable but hidden while either note is trashed.

Wikilinks and backlinks remain derived projections. Note writes update or dirty
bounded projection generations, graph reads use persisted projections and
revision-bound caches/cursors, and tag/source graph nodes remain compatible. Link,
backlink, orphan, projection-maintenance, bootstrap, and restore work is paginated
and capped. Read-only link detail/list authority remains available during resumable
bootstrap, while mutations fail closed until `notes_link_v1` is ready.

Documentation and decisions:

- Added ADR-037 and the approved design/executable plan.
- Updated the Sync v2 API guide and Notes graph README with lifecycle, dataset,
  restore, projection, cursor, cache, and compatibility contracts.
- Recorded the user-approved v58 PostgreSQL lock/FORCE-RLS rollback boundary in the
  plan and task.

Verification evidence:

- Pre-change graph/restore baseline: 66 passed.
- Restore/repair focused gate: 52 passed, 140 deselected.
- Boundedness/query-structure gate: 9 passed, 105 deselected.
- Final schema/product/graph/API/RBAC gate: 238 passed, 26 warnings.
- Final Sync/restore/bootstrap compatibility gate: 362 passed, 1 skipped, 3
  warnings. The skip is the optional live-PostgreSQL test because no live DSN was
  configured; server-free PostgreSQL DDL, RLS, locking, query-shape, and materializer
  contracts passed.
- Final read-authority regression: RED 1 failed, GREEN 1 passed (18 deselected).
- Ruff passed for every task Python file except the exact 17 inherited whole-file
  findings already present in `ChaChaNotes_DB.py` and `ChaCha_Notes_DB_Deps.py` at
  the base commit. All 21 new Python files pass `ruff format --check`.
- Bandit on touched production Python exited 0, `py_compile` exited 0, and both
  `git diff --check` and `git diff --cached --check` exited 0.

Final correctness/security review found and fixed one read-boundary issue: link
detail originally required mutation readiness and was unavailable during resumable
bootstrap. The endpoint now performs the same authorized canonical-dataset read as
link listing without weakening mutation fail-closed behavior. Review of dataset
authority, endpoint-pair ownership/RLS, protected-field privacy, revision-bound
cache/cursors, migration rollback, projection repair, and restore ordering found no
remaining task-scoped defect. No lesson file was added because no new reusable
repository-wide incident remained after the existing design and tests captured the
behavior.

PR review follow-up hardened optimistic link writes with SQL CAS predicates, moved
cursor encoding and projection persistence behind their owning service/store
boundaries, centralized the public validation exception and payload limits, and
made capability schemas advertise the canonical UUID, endpoint-order, and bounded
property contracts. Schema review fixes added cascading link foreign keys,
set-based PostgreSQL canonicalization, column-specific graph triggers, standalone
RLS schema ordering, named result columns, and explicit DSR cleanup for manual and
derived links. Graph reads now cap projection note queries at 1,000 and Sync restore
discovery includes `notes.link`. The approved inactive-Sync compatibility contract
still permits omitted `expected_version`; SQL CAS prevents lost updates there,
while active-Sync mutations continue to require optimistic ancestry.

Review verification recorded focused RED/GREEN coverage for cursor ownership,
CAS races, capability limits, projection abstraction/bounds, restore discovery,
RLS ordering, DSR erasure, trigger scope, and set-based migration behavior. The
complete affected pre-rebase and post-rebase gates each finished with 433 passed
and one optional live-PostgreSQL skip. The freshly fetched `origin/dev` was already
the branch ancestor (zero commits behind), so the requested rebase was a no-op.
Touched-file Ruff, Bandit, byte compilation, and `git diff --check` passed; the
documented whole-file `ChaChaNotes_DB.py` Ruff baseline remains unchanged.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused Notes relationship, graph, and lifecycle suites pass on supported database backends.
- [x] #8 Bandit and static checks pass for touched production files.
- [x] #9 Performance checks demonstrate bounded, paginated graph and orphan queries.
<!-- DOD:END -->
