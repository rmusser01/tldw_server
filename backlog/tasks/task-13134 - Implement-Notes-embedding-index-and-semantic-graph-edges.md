---
id: TASK-13134
title: Implement Notes embedding index and semantic graph edges
status: In Progress
assignee: []
created_date: 2026-08-27 02:20
updated_date: 2026-08-30 16:18
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
- tldw_Server_API/app/core/LLM_Calls/payload_utils.py
- tldw_Server_API/app/core/LLM_Calls/providers/google_embeddings_adapter.py
- tldw_Server_API/app/core/LLM_Calls/providers/huggingface_embeddings_adapter.py
- tldw_Server_API/app/core/LLM_Calls/providers/openai_embeddings_adapter.py
- tldw_Server_API/app/core/http_client.py
- tldw_Server_API/tests/LLM_Adapters/unit/test_embeddings_google_native_http.py
- tldw_Server_API/tests/LLM_Adapters/unit/test_embeddings_huggingface_native_http.py
- tldw_Server_API/tests/LLM_Adapters/unit/test_openai_embeddings_adapter_batch_single.py
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

2026-08-30 Task 4 Fix Round 1 started from fcb7f2a207: correct runtime endpoint provenance and no-redirect adapter transport, pin discovered model revision per run, account for probe/failed/provider-cache outcomes, and reject contradictory semantic settings using strict TDD.

2026-08-30 Task 4 Fix Round 1 complete: added credential endpoint provenance and strict no-redirect native adapter execution, run-pinned discovered revisions, content-free probe/failure/cache-aware usage accounting, and semantic settings cross-limit validation. Verification: Task 4 plus orchestrator 80 passed; settings plus OpenAI/Google/HuggingFace adapter contracts 80 passed; full Task 2 lifecycle including live PostgreSQL 26 passed; Ruff passed with two unchanged http_client.py TRY203 baseline findings excluded; Bandit 0 findings/0 errors; git diff --check passed. Report: .superpowers/sdd/2026-08-29-notes-semantic-index-implementation-plan/task-4-report.md.

2026-08-30 Task 4 Fix Round 2 started from 06e2371e13: disable resolved Google query-key fallback, make resolved Google list batches one batchEmbedContents request, preserve cancellation usage accounting, pin the full run endpoint, and publish full ResolvedDimension through CAS using strict TDD.
2026-08-30 Task 4 Fix Round 2 takeover complete: independently audited the inherited six-file diff against 06e2371e13 with no corrective source/test edits required. Fresh verification: Task 4 plus orchestrator 86 passed; settings plus native adapters 83 passed; both cancellation regressions passed in 5 fresh processes (10/10); Ruff passed on all five touched Python files; Bandit scanned 998 lines with 0 findings and 0 skipped lines; git diff --check passed. One initial unchanged Hypothesis too_slow health-check failure passed with its exact seed and on the full-suite rerun. PostgreSQL was not rerun because no lifecycle/CAS persistence files changed. Report: .superpowers/sdd/2026-08-29-notes-semantic-index-implementation-plan/task-4-report.md.
2026-08-30 Task 4 Fix Round 3 complete from 584d8f9ade: routed every attempted provider usage outcome through one cancellation-draining, single-logger finalizer that preserves truthful success/failure status without duplicate append-only billing rows; added nested Google embedContentConfig.outputDimensionality for resolved scalar/batch requests, probe omission, strict direct-call validation, and a real Notes adapter seam. Verification: focused RED 11 failed/2 passed before implementation; final Task 4 plus orchestrator 89 passed; settings plus OpenAI/Google/HuggingFace adapters 92 passed; four cancellation selectors passed in 10 fresh processes (40/40); Ruff passed all five touched Python files; Bandit scanned 1,061 lines with 0 findings and 0 skipped lines; git diff --check passed. One initial unchanged Hypothesis too_slow health-check failure passed with its exact seed and on the clean full-suite rerun. PostgreSQL was not rerun because lifecycle/CAS persistence was untouched. Report: .superpowers/sdd/2026-08-29-notes-semantic-index-implementation-plan/task-4-report.md.
2026-08-30 Task 4 Fix Round 4 started from 1b0dcc0d91: preserve original cancellation when the single append-only usage logger raises after provider cancellation or accounting-phase cancellation, retain normal logger exception visibility, and make cancellation regressions Python 3.10 compatible without weakening cross-version status/call/leak assertions.
2026-08-30 Task 4 Fix Round 4 complete from 1b0dcc0d91: deferred ordinary shielded logger-task exceptions to the existing result-precedence block so original provider/accounting cancellation wins without retrying the append-only logger or changing truthful status; added two cancellation regressions, a visible non-cancelled logger exception control, and Python 3.10 capability guards limited to cancellation count/message details. Verification: focused RED 2 failed/1 passed; final Task 4 plus orchestrator 92 passed in 6.82s; settings plus OpenAI/Google/HuggingFace 92 passed in 5.54s; logger-exception/repeated-cancellation selectors 40/40 across 10 fresh processes (55.95s aggregate pytest runtime); Python 3.10.20 production-finalizer harness 3/3 with full pytest environment unavailable; Ruff passed both touched Python files; Bandit scanned 846 lines with 0 findings/0 skips; git diff --check passed. Report: .superpowers/sdd/2026-08-29-notes-semantic-index-implementation-plan/task-4-report.md. Residual risk is limited to the absence of a full dependency-backed Python 3.10 suite and live provider calls.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
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
