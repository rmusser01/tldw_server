---
id: TASK-13134
title: Implement Notes embedding index and semantic graph edges
status: In Progress
assignee: []
created_date: 2026-08-27 02:20
updated_date: 2026-08-30 06:39
labels:
- notes
- notes-graph
- embeddings
- second-brain
- backend
dependencies:
- TASK-13138
documentation:
- Docs/superpowers/specs/2026-08-29-notes-semantic-index-design.md
- Docs/superpowers/plans/2026-08-29-notes-semantic-index-implementation-plan.md
priority: medium
modified_files:
- Docs/superpowers/specs/2026-08-29-notes-semantic-index-design.md
- Docs/superpowers/plans/2026-08-29-notes-semantic-index-implementation-plan.md
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/core/DB_Management/chacha/__init__.py
- tldw_Server_API/app/core/DB_Management/chacha/note_semantic_models.py
- tldw_Server_API/app/core/DB_Management/chacha/note_semantic_store.py
- tldw_Server_API/app/core/DB_Management/chacha/note_store.py
- tldw_Server_API/app/core/Sync/v2/materializers/notes.py
- tldw_Server_API/tests/DB_Management/test_chacha_migration_v64.py
- tldw_Server_API/tests/DB_Management/test_chacha_semantic_migration.py
- tldw_Server_API/tests/DB_Management/test_chacha_semantic_migration_postgres.py
- tldw_Server_API/tests/Notes_Graph/unit/test_semantic_store.py
- tldw_Server_API/tests/Notes_Graph/integration/test_semantic_note_lifecycle.py
- tldw_Server_API/tests/Notes_Graph/integration/test_semantic_note_lifecycle_postgres.py
- tldw_Server_API/tests/Sync/test_sync_v2_notes_semantic_lifecycle.py
- tldw_Server_API/app/core/Notes_Graph/semantic_capabilities.py
- tldw_Server_API/app/core/Notes_Graph/semantic_settings.py
- tldw_Server_API/app/core/AuthNZ/permissions.py
- tldw_Server_API/app/core/AuthNZ/settings.py
- tldw_Server_API/app/core/AuthNZ/rbac_seed.py
- tldw_Server_API/app/core/AuthNZ/migrations.py
- tldw_Server_API/tests/Notes_Graph/unit/test_semantic_capabilities.py
- tldw_Server_API/tests/Notes_Graph/unit/test_semantic_settings.py
- tldw_Server_API/tests/AuthNZ_Unit/test_notes_graph_semantic_permissions.py
- tldw_Server_API/tests/AuthNZ/integration/test_notes_graph_semantic_permissions_postgres.py
- tldw_Server_API/tests/AuthNZ_Unit/test_notes_graph_suggestion_permissions.py
- tldw_Server_API/app/core/Notes_Graph/semantic_content.py
- tldw_Server_API/app/core/Notes_Graph/semantic_embeddings.py
- tldw_Server_API/tests/Notes_Graph/unit/test_semantic_content.py
- tldw_Server_API/tests/Notes_Graph/unit/test_semantic_embeddings.py
- tldw_Server_API/tests/Embeddings_isolated/test_notes_semantic_policy.py
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Approved executable plan: `Docs/superpowers/plans/2026-08-29-notes-semantic-index-implementation-plan.md`. Execute its five stages and 13 TDD tasks sequentially, preserving the documented persistence, vector-only storage, Jobs, async graph projection, DSR, and shared UI boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-08-29: Senior implementation-constraint review identified required spec corrections for user-scoped run APIs, semantic opt-in compatibility, evidence offsets, vector-only backend contracts, DSR erasure, graph admission/async composition, dimension and activation rules, cache freshness, and restore semantics.
- Design specification: `Docs/superpowers/specs/2026-08-29-notes-semantic-index-design.md`
- 2026-08-29: Addressed all implementation-constraint review findings in the design: nested run APIs, frozen legacy defaults, field-relative code-point evidence, dedicated ChromaDB/pgvector semantic storage, fail-closed DSR erasure, bounded first-page async projection, dimension preflight and activation gates, fresh status projection, and explicit Sync/full-restore behavior.
- Verification: `git diff --check` passed. This revision changes documentation and task metadata only, so runtime tests and Bandit are not applicable.
- 2026-08-29: Written specification approved. Added the five-stage, 13-task TDD implementation plan at `Docs/superpowers/plans/2026-08-29-notes-semantic-index-implementation-plan.md`.
- Plan self-review closed background credential durability, typed operator settings, execution revalidation, semantic conversion audit provenance, DSR ordering, explicit legacy defaults, and runnable frontend command issues.
- 2026-08-29: Approved spec and executable implementation plan committed in `6e7f95cf8b`. Documentation verification passed: staged diff check, trailing-whitespace scan, balanced 90 code fences, five stages, 13 tasks, and no unresolved TODO/TBD/FIXME markers. Runtime tests and Bandit are not applicable to this docs/task-metadata-only planning revision.
- 2026-08-29 Fix Wave 5 (Task 1): Repaired the PostgreSQL race-test lifecycle so observer/worker pools are protected from creation, exact blocked worker PIDs are cancelled/terminated before pool close, both non-daemon threads must exit, and injected assertion cleanup preserves the primary failure. Verification: 5/5 fresh race processes and 63 focused+adjacent tests passed; Ruff, Bandit (test assertions excluded), and `git diff --check` passed.
- 2026-08-29 Task 1 complete: Added ChaChaNotes v65 semantic configuration, generation, note-state, chunk-manifest, and work-ledger persistence with owner/dataset scoping, forced PostgreSQL RLS, CAS publication/dimension fences, bounded cleanup, canonical digest checks, and deterministic live PostgreSQL concurrency coverage. Independent task review clean. Range: `14b4b6464e..6639f7acb6`.
- 2026-08-29 Task 2 complete: Canonical Note and Sync create/edit/restore/delete paths now update semantic dirty/tombstone work in the same transaction for the authoritative dataset, remain no-ops when disabled/unscoped, and retain generation cleanup identity across hard deletes. Real Notes JSON round-trip and live PostgreSQL product-lifecycle tests close restore, RLS, rollback, conflict-upsert, and cascade risks. Independent review clean. Verification: 26 tests passed; Ruff, Bandit, and diff checks clean. Range: `88d169beef..ac45d93fe3`.
- 2026-08-29 Task 3 complete: Added deterministic compatibility/disclosure identities with separate sanitized model and revision fields, durable-credential preflight, bounded semantic operator settings, and AuthNZ migration 096 for `notes.graph.semantic.manage`. Migration/backstop behavior preserves later revocations while granting approved roles on catalog creation. Independent runtime/migration review clean. Verification: 59 tests passed; Ruff, Bandit, and diff checks clean. Range: `c90625d842..37d96af28e`.
- 2026-08-29 Task 4 complete: Added version-bound canonical chunking and strict endpoint-neutral embedding execution; migrated NoteStore semantic fingerprints to bind `content_version`; extended typed semantic limits. Verification: Task 4 plus orchestrator 73 passed; Task 2 lifecycle 14 passed; PostgreSQL lifecycle 1 passed; Ruff, Bandit, and diff checks clean. Report: `.superpowers/sdd/2026-08-29-notes-semantic-index-implementation-plan/task-4-report.md`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-13134 remains in progress after completion of implementation Tasks 1 through 4.
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
