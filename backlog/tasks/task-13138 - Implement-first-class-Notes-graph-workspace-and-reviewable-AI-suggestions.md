---
id: TASK-13138
title: Implement first-class Notes graph workspace and reviewable AI suggestions
status: In Progress
assignee: []
created_date: 2026-08-27 03:40
updated_date: 2026-08-27 16:02
labels:
- notes
- knowledge-graph
- webui
- browser-extension
- llm
- jobs
dependencies: []
references:
- TASK-13134
- TASK-13135
- TASK-13136
- TASK-13137
documentation:
- Docs/superpowers/specs/2026-08-26-notes-second-brain-graph-suggestions-design.md
- Docs/superpowers/plans/2026-08-26-notes-second-brain-graph-suggestions.md
priority: high
modified_files:
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_models.py
- tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py
- tldw_Server_API/app/core/DB_Management/chacha/note_store.py
- tldw_Server_API/tests/DB_Management/test_chacha_migration_v64.py
- tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v64.py
- tldw_Server_API/app/core/Notes_Graph/suggestion_content.py
- tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_content.py
- tldw_Server_API/app/core/Notes_Graph/suggestion_retrieval.py
- tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_retrieval.py
- tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_retrieval_backends.py
- tldw_Server_API/app/core/Notes_Graph/suggestion_store.py
- tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_store.py
- tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_lifecycle.py
- tldw_Server_API/app/core/Notes_Graph/suggestion_capabilities.py
- tldw_Server_API/app/core/Notes_Graph/suggestion_generation.py
- tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_capabilities.py
- tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_generation.py
- tldw_Server_API/tests/LLM_Adapters/unit/test_notes_graph_suggestion_call_policy.py
- tldw_Server_API/app/core/Chat/chat_service.py
- tldw_Server_API/app/core/LLM_Calls/adapter_registry.py
- tldw_Server_API/app/core/LLM_Calls/capability_registry.py
- tldw_Server_API/app/core/LLM_Calls/providers/openai_adapter.py
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/app/core/Jobs/worker_sdk.py
- tldw_Server_API/app/core/Notes_Graph/suggestion_jobs.py
- tldw_Server_API/app/core/Notes_Graph/suggestion_maintenance.py
- tldw_Server_API/app/core/Notes_Graph/suggestion_observability.py
- tldw_Server_API/app/core/Notes_Graph/suggestion_service.py
- tldw_Server_API/app/services/notes_graph_suggestions_maintenance.py
- tldw_Server_API/app/services/notes_graph_suggestions_worker.py
- tldw_Server_API/app/services/startup_study_privilege_jobs_pollers.py
- tldw_Server_API/tests/Jobs/test_jobs_prune_postgres.py
- tldw_Server_API/tests/Jobs/test_jobs_prune_sqlite.py
- tldw_Server_API/tests/Jobs/test_worker_sdk.py
- tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_jobs.py
- tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_observability.py
- tldw_Server_API/tests/Services/test_notes_graph_suggestions_workers.py
- tldw_Server_API/tests/Services/test_startup_study_privilege_jobs_pollers.py
- tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py
- tldw_Server_API/app/core/Jobs/operations/postgres/idempotency.py
- tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py
- tldw_Server_API/app/core/Jobs/operations/sqlite/idempotency.py
- tldw_Server_API/tests/Jobs/test_jobs_manager.py
- tldw_Server_API/tests/Jobs/test_jobs_manager_postgres.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Graph as a first-class Notes view mode shared by the WebUI and browser extension, and add an on-demand, source-grounded suggestion workflow for the selected note. Suggestions use a whole-library Notes FTS shortlist plus one bounded configured LLM invocation to propose related-note links and tags. Suggestions remain provisional until explicitly accepted or rejected; acceptance uses the existing manual-link and tag mutation paths. This slice does not add embeddings, semantic edge types, background organization, library-wide themes, or saved layouts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Notes exposes Graph as a first-class view mode with a focused graph canvas, search, focus control, edge-type filters, layout/fit controls, and a responsive inspector rather than a routine modal.
- [ ] #2 The graph initially focuses on the selected or most-recent note, supports bounded interactive expansion, and offers all-notes mode only below the configured cap.
- [ ] #3 The inspector provides Details and Suggestions views, grounded evidence for both notes, Strong match/Possible match bands, accept/reject controls, provisional dashed edges, and tag suggestion chips.
- [ ] #4 Canvas and Relationships views provide equivalent access to graph relationships; keyboard, focus, non-color state, narrow-screen, overflow, and long-content behavior are covered.
- [ ] #5 Suggestion generation is exposed only beneath the existing Notes graph namespace and runs through Jobs with owner/dataset scoping, idempotent admission, max_retries=0, cancellation, bounded inputs/outputs, and no note content or credentials in Job payloads or logs.
- [ ] #6 FTS searches the active owner-scoped Notes library and excludes only the selected note, trash, and directly linked note pairs; shared tag/source membership does not exclude candidates.
- [ ] #7 One configured LLM invocation receives only a bounded allowlisted shortlist and tag catalog, treats notes as untrusted data, uses a strict output schema, and cannot introduce unknown note IDs, tools, provider settings, or unbounded fields.
- [ ] #8 Suggestion runs and provisional suggestions are durable, paginated, retention bounded, and keyed by content fingerprints; evidence is stored as fingerprint-bound canonical-text offsets and reconstructed on read rather than copied into suggestion records.
- [ ] #9 Relationship suggestions accept as ordinary undirected manual links with weight 1.0 and no model-selected semantic label/properties; tag suggestions use existing tag normalization and cap newly invented tags.
- [ ] #10 Accept/reject operations are idempotent and race safe. Acceptance uses compare-and-swap plus the existing mutation path and a bounded reconciliation lease; unchanged-version rejection suppresses the same pair/tag across model or prompt versions.
- [ ] #11 Accepting one tag does not stale sibling suggestions: title/body content fingerprints are independent of tag membership, and existing-tag acceptance resolves as an idempotent success.
- [ ] #12 Request-time validation uses HTTP conflict/validation/rate-limit/readiness responses; failures after 202 are represented by durable run status, stable error codes, and sanitized user guidance.
- [ ] #13 Generation validates the top-level response strictly, drops individually invalid or duplicate items, atomically persists the validated set, and records only aggregate validation counts; no invalid suggestion is exposed.
- [ ] #14 Current projection freshness is verified or the run reports degraded/unavailable discovery rather than claiming a complete current-library search.
- [ ] #15 A successful current-version run supersedes older pending suggestions while preserving current-version rejections; stale/obsolete records follow bounded retention and note/user deletion cascades.
- [ ] #16 Backend unit/integration/property tests, frontend component/contract/accessibility tests, Playwright desktop/mobile visual checks, and an offline suggestion-quality evaluation corpus cover the approved design.
- [ ] #17 Relevant Notes and API documentation is updated, touched code passes targeted tests and lint/type checks, and Bandit reports no new findings.
- [ ] #18 A nested capability preflight discloses and ETag-binds the effective provider/model, endpoint-origin revision, local/remote/unknown data boundary, outbound data categories, permissions, and limits; the worker revalidates the same revision immediately before provider invocation.
- [ ] #19 Publication verifies an immutable owner-scoped terminal Job receipt across active/archive Jobs and revalidates source/target freshness before a fenced ChaChaNotes activation transaction; abandoned active run states reconcile fail closed.
- [ ] #20 Suggestion preprocessing enforces explicit pre-transfer UTF-8 byte limits for selected and candidate notes, reports oversized selected notes with actionable 422 guidance, and never silently truncates analyzed content.
- [ ] #21 Suggestion reconciliation runs independently of provider-worker readiness, and acceptance uses a renewable lease plus an in-transaction coordinator guard that fences both canonical mutation and suggestion finalization.
- [ ] #22 Static reset routing, loaded-graph search scope, paginated Relationships accessibility behavior, and the documented 90-day idempotency replay horizon are covered by contract tests.
- [ ] #23 All mutating suggestion routes use durable operation receipts with request fingerprints and bounded replay envelopes; reset is revision-guarded so replay cannot delete later rejections.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the approved 12-task test-first plan in Docs/superpowers/plans/2026-08-26-notes-second-brain-graph-suggestions.md. It stages schema/store work, bounded retrieval and provider contracts, Jobs publication, guarded Sync decisions, nested API/RBAC, the shared Graph workspace, accessibility, E2E, documentation, and security verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Deferred work is tracked separately: TASK-13134 embeddings and semantic edges; TASK-13135 automatic background organization; TASK-13136 library-wide recurring themes; TASK-13137 saved graph views/layouts. This task remains review-first and on-demand.

Design approved in chat and written to Docs/superpowers/specs/2026-08-26-notes-second-brain-graph-suggestions-design.md. Self-review completed for incomplete markers, internal consistency, scope, asynchronous state transitions, privacy, cancellation, retention, and RBAC. The design-only change has no Bandit-applicable code scope.
Written-spec hardening pass incorporated provider capability preflight, active/archive publication receipts, Job-row owner authority, bounded note analysis, static route ordering, renewable acceptance fencing, provider-independent maintenance, loaded-graph search and accessible relationship pagination, and an explicit idempotency replay horizon.
Analysis limits were refined from post-load code-point counts to backend-portable pre-transfer UTF-8 byte predicates so oversized Notes cannot consume application memory before enforcement.
First independent spec review found five state-machine blockers. The spec now binds provider configuration/data categories through worker invocation, revalidates freshness at activation, reconciles every active run state, fences canonical mutations inside coordinator transactions, and persists 90-day idempotency receipts for every mutating route.
Second spec review found an interrupted enqueue/cancellation continuation ambiguity. Operation receipts now distinguish terminal replay from narrowly scoped in-progress continuation, maintenance resends the same idempotent cancellation command, and provider adapters must disable internal retries. Third review approved the written spec with no blocking issues; its advisory replay-wording clarification was also incorporated.
Verification before amendment commit: third independent review status Approved with no issues; advisory wording incorporated. `git diff --check` passed. The Backlog acceptance-criteria assertion reported sequential criteria 1-23. Scope is two documentation/tracking files. Runtime tests and Bandit are not applicable to this docs-only design amendment.

Implementation plan completed at Docs/superpowers/plans/2026-08-26-notes-second-brain-graph-suggestions.md. Three independent plan-review iterations were completed. Review findings addressed queue-specific 30-day Jobs receipt retention, graph-reader All-notes metadata, reload run discovery, non-vacuous observability coverage, complete Bandit scope, keyword lifecycle/RBAC tests, exact offline quality thresholds, and a baseline-aware Ruff gate. Final reviewer status: Approved with no issues.
Plan verification: 12 tasks and 82 explicit checkbox steps; all modification/deletion paths exist or are created by an earlier task; no TODO/TBD/placeholder markers; all Python commands use the repository virtual environment; `git diff --check` passed before staging. Runtime tests and Bandit are not applicable to this documentation-only planning change.

Implementation started with the subagent-driven-development workflow after confirming the isolated worktree and rebasing on current origin/dev.
Task 1 started: implementing ChaChaNotes schema v64 and the typed graph-suggestion store skeleton from the approved plan. Scope is SQLite/PostgreSQL migration parity, typed records/store initialization, focused migration tests, regression coverage, Bandit, and a task report.
Task 1 implementation completed locally: schema v64 adds SQLite/PostgreSQL Notes graph suggestion tables, indexes, forced PostgreSQL RLS, typed records, and an owner-bound store skeleton. Verification: required v61-v64 migration regression suite passed (25 tests); Ruff passed on touched Python files; Bandit scanned the production scope with 0 findings. Final report: .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-1-report.md. Parent task remains In Progress for later tasks.
Final Task 1 candidate verification: exact required v61-v64 migration regression suite passed 26 tests (including live PostgreSQL v63-to-v64 upgrade); Ruff passed; `git diff --check` passed; Bandit found 0 findings. Parent task remains In Progress for later tasks.
Task 1 Fix Round 1 completed locally: v64 now enforces composite owner/dataset foreign keys across graph tables, receipt source-note cascade on hard note deletion, and lifecycle-aware canonical unique identities for related pairs and tags. Verification: focused SQLite/PostgreSQL v64 suite passed 15 tests; required v61/v62/v64 migration regression suite passed 33 tests; Ruff passed; Bandit reported 0 findings. Report updated at .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-1-report.md. Parent task remains In Progress for later tasks.
Task 1 Fix Round 2: replaced composite receipt FK `ON DELETE SET NULL` with scope-preserving `NO ACTION` plus equivalent SQLite/PostgreSQL BEFORE DELETE triggers that clear only the nullable receipt IDs. Added SQLite/live PostgreSQL tests for referenced receipt deletion and preservation of pending/rejected current-fingerprint suggestions. Verification: 35 migration tests passed; Ruff passed; Bandit reported 0 findings. Report: `.superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-1-report.md`.

Task 2 completed in notes-second-brain-planning. Added deterministic bounded owner/dataset-scoped Notes graph retrieval, canonical source grounding, exact fingerprint suppression, backend-portable FTS, byte-limit enforcement, tag bounds, and SQLite/PostgreSQL integration coverage. Modified: tldw_Server_API/app/core/Notes_Graph/suggestion_content.py; tldw_Server_API/app/core/Notes_Graph/suggestion_retrieval.py; tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py; tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_content.py; tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_retrieval.py; tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_retrieval_backends.py; .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-2-report.md. Verification: Task 2 20 passed/0 skipped, directly affected Notes FTS 3 passed/57 deselected, Ruff clean, Bandit 0 findings. Report: .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-2-report.md.

Task 2 Fix Round 1 completed locally. Added fail-closed exact owner/dataset authority validation, exact SQLite/PostgreSQL FTS trigger/structure contract checks, bounded first-ranked-60 metadata shortlist before exclusions or text transfer, shortlist-only oversized aggregation, and strengthened independent evidence-offset property assertions. Modified: .gitignore; tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py; tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py; tldw_Server_API/app/core/Notes_Graph/suggestion_retrieval.py; tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_content.py; tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_retrieval.py; tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_retrieval_backends.py. Verification: Task 2 suites 25 passed/0 skipped (2 established warnings); focused Notes search 1 passed/57 deselected; PostgreSQL FTS 8 passed; Ruff clean; Bandit 0 results. The broader legacy v10 migration test remains an unrelated known failure: _migrate_from_v57_to_v58 queries absent legacy note_edges. Local report retained but ignored at .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-2-report.md and removed from git index.

Task 2 Fix Round 2 completed locally. Closed the remaining live PostgreSQL selected-source byte-guard coverage gap with a characterization integration test. It asserts NotesGraphSourceTooLargeError(notes_graph_source_too_large), verifies the only source title/content query includes the octet_length byte predicate, and confirms candidate FTS is not reached. Modified: tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_retrieval_backends.py. Verification: focused live PostgreSQL characterization 1 passed/0 skipped; Task 2 suites 26 passed/0 skipped (2 established warnings); focused Notes SQLite search 1 passed/57 deselected; PostgreSQL FTS 8 passed; Ruff clean. Bandit not applicable: no production code changes. Report remains local and ignored at .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-2-report.md.

Task 2 Fix Round 3 completed locally. Tightened the live PostgreSQL oversized-source characterization test from an unanchored pytest regex match to exact str(exc.value) == notes_graph_source_too_large equality while retaining the exception type and SQL byte-guard/no-FTS ordering assertions. Modified: tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_retrieval_backends.py. Verification: focused live PostgreSQL test 1 passed/0 skipped (2 established warnings); Task 2 suites 26 passed/0 skipped (2 established warnings); Ruff clean. Bandit not applicable: no production code changes. Report remains local and ignored at .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-2-report.md.
Task 3 started in notes-second-brain-planning at base 47f76468362839c3ace61df10a8e824362097eaf. Scope: durable ChaChaNotes suggestion runs/receipts/publication/retention plus same-transaction NoteStore invalidation hooks. Reused patterns inspected before edits: Jobs owner-scoped idempotency receipts and constant-time fingerprint replay checks; Slides/shared-workspace revision and lease-fence CAS transitions; Notes attachment/Sync bounded HMAC cursor encoding; NoteStore projection/sidecar lifecycle hooks inside product transactions; Jobs bounded retention candidate selection. Strict two-file RED will precede production edits.
Task 3 complete: implemented the durable ChaChaNotes suggestion run/receipt/decision state machine, atomic staged publication, five-minute fenced acceptance leases, signed pagination, bounded exact-horizon retention, and same-transaction NoteStore invalidation hooks. Corrected the unreleased v64 active-run index to the approved full tuple and split hidden staged uniqueness from visible canonical identity after focused SQLite/PostgreSQL RED demonstrations.

Verification:
- Required Task 3 GREEN: 21 passed, 0 skipped, 2 warnings.
- Task 1 v64 SQLite/PostgreSQL migration regressions: 17 passed, 0 skipped, 2 warnings.
- Task 2 retrieval regressions: 26 passed, 0 skipped, 2 warnings.
- Direct NoteStore/projection lifecycle regressions: 36 passed, 0 skipped, 2 warnings.
- Ruff all touched Python: clean.
- Bandit touched production Python: 0 findings, 0 errors.
- Live PostgreSQL exercised active-run identity, note lifecycle invalidation, publication, receipt replay, acceptance fencing, reject/reset, and retention; no established-fixture skips.
- Local ignored report: .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-3-report.md
Task 3 Fix Round 1 completed locally from reviewer base 5fdb9575256d888cc6a743318cc6112a1330aff8. Fixed terminal receipt replay after detail cleanup, explicit atomic admission failure, canonical undirected related identity, tag sync-id re-resolution and membership filtering, receipt-bound acceptance fences/release, unreleased-v64 non-null binding fields, closed transition/request/envelope contracts, exact bounded retention, and complete same-transaction NoteStore/Sync lifecycle rollback coverage. Verification: required Task 3 suite 43 passed/0 skipped; v64 SQLite/live-PostgreSQL 28 passed/0 skipped; Task 2 26 passed/0 skipped; direct NoteStore/projection/Sync 65 passed/0 skipped; final deduplicated 11-file run 154 passed/0 skipped (2 established warnings). Ruff clean; Bandit 0 findings/0 errors; git diff --check passed. Live PostgreSQL ran with no skips. Ignored report appended at .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-3-report.md.
Task 3 Fix Round 2 started from exact base 2cd3220c4533476d8c06e201354191ff62650d18. Scope is the two verified remaining findings: accepting-identity activation filtering for canonical tag/undirected related suggestions, and a closed Task-5-ready cancellation/publication/maintenance CAS persistence surface. Focused SQLite/live-PostgreSQL RED precedes production edits; no Jobs/provider authority is added.
Task 3 Fix Round 2 completed from exact base 2cd3220c4533476d8c06e201354191ff62650d18. Activation now filters staged canonical tag/undirected-related duplicates when an accepting row owns the visible identity, preserving that row's revision, lease, and decision receipt while publishing unrelated staged rows atomically. Added exact notes_graph_capabilities_changed_before_queue admission failure replay and closed operation-specific cancellation receipt plus Task-5 maintenance reconciliation CAS primitives; no Jobs/provider I/O or generic transition surface was added. Strict focused RED: 6 failed/8 deselected/0 skipped/2 warnings, followed by 6 passed/8 deselected/0 skipped/2 warnings. Review subcycles: queued-stale continuation RED 2 failed then GREEN 2 passed; missing in-progress resource RED 2 failed then GREEN 2 passed, both SQLite/live PostgreSQL. Verification: Task 3 49 passed/0 skipped; v64 migration 28 passed/0 skipped; Task 2 26 passed/0 skipped; direct NoteStore/projection/Sync 58 passed/0 skipped; final deduplicated 11-file run 161 passed/0 skipped/2 inherited warnings. One unchanged initial final run hit Hypothesis input-generation health-check timing (1 failed/160 passed); isolated rerun and unchanged full rerun passed. Ruff clean; Bandit 0 findings/0 errors; git diff --check passed. Ignored report appended at .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-3-report.md.
Task 3 Fix Round 3 started from exact base 4faf2e73c10b291055cb8c5ff1739fdfb2e6676f. Final ordinary fix scope is the ledger-approved unreleased-v64 maintenance lease omission and operation-specific error/guidance pair contracts. SQLite/live-PostgreSQL migration and store tests will be added and run RED before schema/model/store edits; Task 5 retains scheduling, Jobs lookup, and external effects.

Task 3 Fix Round 3 completed from exact base 4faf2e73c10b291055cb8c5ff1739fdfb2e6676f. Unreleased v64 now persists paired five-minute maintenance lease tokens/expiries and an owner/dataset active-run scan index. The store claims deterministic bounded batches (1..100), fences reconcile/release by owner, dataset, state, revision, exact non-expired token, clears leases on all competing run transitions, and keeps Jobs/provider I/O outside ChaChaNotes. Admission and worker error/guidance contracts are exact operation-specific mappings; job/publication outcomes are derived and run-admit replay rejects unsafe persisted cross-products. Strict RED: 6 failed/42 deselected/0 skipped/2 warnings; lease subcycles each failed on SQLite/live PostgreSQL before fixes. GREEN: focused 6 passed; Task 3 53 passed; v64 SQLite/live PostgreSQL 30 passed; Task 2 26 passed; NoteStore/projection/Sync 58 passed; final deduplicated run 167 passed/0 skipped/2 inherited warnings. Ruff clean; Bandit 0 findings; git diff --check passed. Report appended locally at .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-3-report.md and remains ignored.
Task 4 started from exact base 65d98c8624160152f8af7b87b7daf1dbd64c9aca. Scope is capability disclosure, deterministic opaque revisions, strict bounded local validation, deterministic match strength, and one provider call through ProviderCallPolicy/perform_chat_api_call_async. Inspected established prompt-improvement call-policy tests, structured-output negotiation, canonical endpoint scope, provider config resolution, and capability derivation before edits. The three required test files will be written and run RED before the two production modules.
Task 4 completed from exact base 65d98c8624160152f8af7b87b7daf1dbd64c9aca. Added deterministic provider capability disclosure with canonical endpoint-origin digest, exact stable limits/data categories, unknown-as-external boundary behavior, strict bounded one-call generation through ProviderCallPolicy/perform_chat_api_call_async, local JSON/schema validation, and server-computed Strong/Possible match strength. No Jobs, persistence, API, service, or shared adapter/policy changes were made. Final required plus existing no-retry suite: 56 passed/0 skipped/4 inherited warnings. Ruff check clean; Ruff format check reported 5 files formatted; Bandit 0 findings/0 errors/0 nosec/0 skipped across 794 production LOC; git diff --check passed. Prompt-improvement regressions were not required because shared policy code was unchanged. Privacy/one-call self-review found no sensitive logging or persisted transport material. Ignored report: .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-4-report.md.
Task 4 Fix Round 1/3 started from exact base b2af691006235d1d49492a25ad3ddb0aba9a92c1. Scope is the seven verified review findings and latest ledger ruling: derive one-attempt/same-origin safety from an audited adapter contract bound to the actual ConfiguredEndpointScope, enforce opt-in hard transport/wall-clock timeouts without changing ordinary or Prompt Improvement behavior, bind one validated effective-limits authority through disclosure/request/parse, reject ambiguous response candidates, normalize NFC before matching, reject duplicate/deep JSON safely, and record exact inherited warning evidence. Focused tests will be written and run RED before production edits; shared production changes are limited to the opt-in provider policy/adapter enforcement required by the ruling.

Task 4 Fix Round 1 completed from exact base b2af691006235d1d49492a25ad3ddb0aba9a92c1. Replaced caller-attested transport booleans with an exact-concrete-adapter registry contract and bound the disclosed ConfiguredEndpointScope to the OpenAI centralized fetch transport. Strict opt-in calls use one retry attempt, same-origin-only redirects, configured timeout clamping, and an outer async deadline; ordinary and Prompt Improvement behavior remains unchanged. One validated effective limits object now drives disclosure revision, prompt/request caps, parser caps, output size, candidate count, and timeout. Added exact-one-choice normalization, NFC matching, duplicate-key/depth-safe JSON rejection, and real counting-transport 307/308 coverage. Strict RED: 67 failed/7 passed/4 warnings. GREEN: Task 4 focused 74 passed/0 skipped/4 inherited warnings; required Task 4 plus no-retry, Prompt Improvement, and OpenAI adapter regressions 128 passed/0 skipped/4 inherited warnings. Ruff lint clean; five Task 4 files/tests Ruff-format clean; four pre-existing shared files intentionally retain baseline formatting to avoid mechanical churn; Bandit 0 findings; git diff --check passed. Ignored report appended at .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-4-report.md.

Task 5 NEEDS_CONTEXT at base 9be6076b18cc4865d572c1e85d887429977cc992. Blocking closed-contract gap: bind_admitted_run requires expected_completion_token before enqueue binding reaches queued, but the required completion token is the WorkerSDK-acquired lease ID, which does not exist until acquire_next_job; start_run and stage_suggestions expose no fenced way to bind it. Job UUID substitution would force receipt mismatch, direct SQL would bypass Task 3 CAS, and preallocating Jobs lease IDs broadens scope. Recommended ruling: authorize an operation-specific Task 3 extension that atomically binds the acquired lease ID during queued-to-running CAS. No production/tests changed; RED intentionally not authored because it would encode an unapproved architecture. Ignored report: .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-5-report.md.

Task 5 completed from base 9be6076b18cc4865d572c1e85d887429977cc992 after the approved lease-binding and authoritative-owner/run-lookup rulings. Added content-free idempotent Jobs admission, owner-row-authoritative one-call worker execution, queued placeholder-to-acquired-lease CAS, running-to-publishing Job/token fencing, active/archive immutable receipt publication, provider-independent bounded maintenance, closed local observability, opt-in WorkerSDK completion-token binding, exact 31-day forced-archive prune handling, and app-owned lifecycle registration. Final required Step 7: 102 passed/0 skipped/4 inherited warnings, including live PostgreSQL. Task 3 plus migrations: 85 passed/0 skipped/2 inherited warnings. Task 4 regressions: 128 passed/0 skipped/4 inherited warnings. Scoped Jobs/idempotency/startup: 167 passed/0 skipped/2 inherited warnings; focused updated startup expectations: 16 passed/0 skipped/2 inherited warnings. Three broad sidecar/inventory expectation failures are byte-identical inherited baseline defects against 9be6076 and were not changed. Ruff clean; Bandit 0 findings; git diff --check clean. Ignored report: .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-5-report.md.
Task 5 Review Fix Round 1/3 completed from exact base 53d084935a03eea2daaea790d842848eda340546. Hardened post-enqueue admission replay, durable owner/dataset maintenance discovery, exact leased publication activation, 10-minute missing-Job grace across active pre-publication states, adjacent capability revalidation, closed non-vacuous lifecycle observability, resumable identity-guarded cancellation, and shared production maintenance cadence. Verification: Task 5 Step 7 121 passed/0 skipped/4 inherited warnings; Task 3 plus live PostgreSQL 87 passed/0 skipped/2 warnings; Task 4 129 passed/0 skipped/4 warnings; Jobs SQLite/live PostgreSQL 71 passed/2 optional encryption skips/2 warnings. Affected startup run: 54 passed/3 inherited baseline failures/0 skipped/8 warnings; failing files are byte-identical to base. Ruff clean; Bandit 0 findings; git diff --check clean. Ignored report: .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-5-report.md.
Task 5 Fix Round 2/3 completed from exact base 5289ddc8b73ea4120e0549d9f7117d1e45d5c232. Moved provider_started after successful capability revision/availability validation and immediately before the sole provider call; cancellation-before-capability and capability-call adjacency remain unchanged. Added mismatch/unavailable regression proving no provider_started, no provider invocation, and closed failure telemetry. Dataset discovery is unchanged under the controller's one-authoritative-dataset-per-owner ruling. Focused RED: 2 failed/0 skipped/4 inherited warnings; GREEN on repair attempt 1: 2 passed/0 skipped/4 warnings. Task 5 Step 7: 123 passed/0 skipped/4 warnings. Task 4 call-policy: 20 passed/0 skipped/4 warnings. Ruff clean; Bandit 0 findings; diff check clean. Ignored report: .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-5-report.md.
Task 6 started from exact base 34e8e58ad5ab254bbadc0b4ff57541b6e8cd82ee. Scope is the narrow process-local GuardedProductMutation contract threaded only through synchronous server-origin Notes link/organization materialization, with identity validation, same-product-transaction before/after callbacks, exact-replay guarded finalization, and unchanged unguarded behavior. Strict focused RED will precede production edits; no callbacks will enter envelopes, Jobs, logs, or durable storage.
Task 6 completed from exact base 34e8e58ad5ab254bbadc0b4ff57541b6e8cd82ee. Added a closed process-local GuardedProductMutation for notes.link, notes.keyword, and notes.keyword_link; threaded it only through synchronous server-origin capture/materialization; validated one exact group identity before product writes; re-executed guarded exact replay; and ran before/after in the locked ChaCha product transaction around verified canonical postconditions. Guarded keyword creation runs only before and cannot finalize; guarded note-keyword relationship after is the finalization point. Callbacks are absent from Sync envelopes, Jobs, logs, and durable state. Strict RED: required test file failed collection with ModuleNotFoundError for the missing guarded_product_mutation contract (1 error, 5 inherited warnings). First GREEN attempt: 4 failed/2 passed because replay guard plumbing targeted manifest loading; repair attempt 1 moved it to replay materialization and focused GREEN passed 6/6. Self-review tightened keyword no-finalizer behavior; focused verification passed 8/8 with 33 deselected and 2 inherited warnings. Final required regression suite: 92 passed/0 skipped/2 inherited warnings. Ruff lint passed all 15 touched Python files; the two new files are Ruff-format clean. A whole-touched-file format check would reformat 11 legacy files, so it was not applied to avoid unrelated mechanical churn. Bandit scanned 13,666 production LOC with 0 findings/0 errors and 26 established skipped tests; git diff --check passed. Parent TASK-13138 remains In Progress for Tasks 7-12.
Task 6 Fix Round 1 completed locally from reviewed head a7b7c1607072af25015a485fd2d11db14e0f886a. Guard identities and complete guarded Notes plans are now validated before append; matching envelopes carry only a fixed content-free guard-required marker, and every replay entry point fails closed without the exact fresh process-local capability set. New-tag capture uses a before-only keyword guard plus the finalizing keyword-link guard. Added post-append SQLite barriers and a live PostgreSQL row-lock barrier; also corrected the PostgreSQL note_edges.directed 0/1 binding exposed by that test. Verification: required suite 98 passed/0 skipped/2 inherited warnings; Ruff clean; Bandit 0 findings/0 errors; git diff --check clean. Full Fix Round 1 evidence is in .superpowers/sdd/2026-08-26-notes-second-brain-graph-suggestions/task-6-report.md. Parent task remains In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
