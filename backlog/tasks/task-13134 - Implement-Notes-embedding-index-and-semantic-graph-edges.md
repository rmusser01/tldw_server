---
id: TASK-13134
title: Implement Notes embedding index and semantic graph edges
status: In Progress
created_date: 2026-08-27 02:20
labels:
- notes
- notes-graph
- embeddings
- second-brain
- backend
priority: Medium
dependencies:
- TASK-13138
updated_date: 2026-08-30 01:06
modified_files:
- Docs/superpowers/specs/2026-08-29-notes-semantic-index-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an opt-in, owner-scoped embedding lifecycle for Notes and expose bounded semantic-similarity relationships as a derived Notes Graph edge type. Preserve manual links as the canonical user-approved relationship model; semantic edges remain reproducible projections tied to an embedding model and note content version.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Notes embedding records are owner- and dataset-scoped, versioned by note content and embedding model, and updated through bounded Jobs-backed indexing and reindexing.
- [ ] #2 Create, edit, trash, restore, delete, model-change, and rebuild lifecycles invalidate or refresh embeddings deterministically without exposing another user's data.
- [ ] #3 The Notes Graph schema and existing /api/v1/notes graph routes support an optional semantic edge type with configurable top-k, threshold, hard caps, truncation metadata, and graceful feature-disable behavior.
- [ ] #4 Semantic edges expose model/version provenance and similarity evidence sufficient for UI explanation but are never persisted as manual links without explicit user acceptance.
- [ ] #5 SQLite and PostgreSQL behavior, RBAC, Sync compatibility, failure recovery, performance bounds, unit/integration/property tests, documentation, and Bandit verification are covered.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-08-29: Senior implementation-constraint review identified required spec corrections for user-scoped run APIs, semantic opt-in compatibility, evidence offsets, vector-only backend contracts, DSR erasure, graph admission/async composition, dimension and activation rules, cache freshness, and restore semantics.
- Design specification: `Docs/superpowers/specs/2026-08-29-notes-semantic-index-design.md`
- 2026-08-29: Addressed all implementation-constraint review findings in the design: nested run APIs, frozen legacy defaults, field-relative code-point evidence, dedicated ChromaDB/pgvector semantic storage, fail-closed DSR erasure, bounded first-page async projection, dimension preflight and activation gates, fresh status projection, and explicit Sync/full-restore behavior.
- Verification: `git diff --check` passed. This revision changes documentation and task metadata only, so runtime tests and Bandit are not applicable.
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
