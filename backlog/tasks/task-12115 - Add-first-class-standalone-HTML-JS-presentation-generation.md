---
id: TASK-12115
title: Add first-class standalone HTML-JS presentation generation
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-07-16 13:15
labels:
- slides
- presentation-studio
- backend
- frontend
- security
dependencies: []
documentation:
- Docs/superpowers/specs/2026-07-15-standalone-html-presentations-design.md
- Docs/superpowers/plans/2026-07-15-standalone-html-presentations-implementation-plan.md
priority: high
modified_files:
- tldw_Server_API/app/core/Slides/slides_migrations.py
- tldw_Server_API/app/core/Slides/slides_db.py
- tldw_Server_API/app/core/DB_Management/db_path_utils.py
- tldw_Server_API/app/core/Slides/standalone_html_contracts.py
- tldw_Server_API/app/core/Slides/standalone_html_validator.py
- tldw_Server_API/app/core/Slides/standalone_html_validation_pool.py
- tldw_Server_API/app/core/Slides/presentation_service.py
- tldw_Server_API/app/api/v1/schemas/slides_schemas.py
- tldw_Server_API/app/api/v1/endpoints/slides.py
- tldw_Server_API/app/core/Slides/slides_export.py
- tldw_Server_API/tests/Slides/test_standalone_html_db_migration.py
- tldw_Server_API/tests/Slides/test_standalone_html_domain.py
- tldw_Server_API/tests/Slides/test_standalone_html_validator.py
- tldw_Server_API/tests/Slides/test_standalone_html_validation_pool.py
- tldw_Server_API/tests/Slides/test_standalone_html_dependency_smoke.py
- tldw_Server_API/tests/Slides/test_standalone_html_api.py
- tldw_Server_API/tests/Slides/test_slides_db.py
- tldw_Server_API/tests/Slides/test_slides_export.py
- pyproject.toml
- backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md
- tldw_Server_API/app/core/Slides/__init__.py
- Docs/superpowers/plans/2026-07-15-standalone-html-presentations-implementation-plan.md
- tldw_Server_API/Config_Files/Prompts/README.md
- tldw_Server_API/Config_Files/Prompts/slides.prompts.md
- tldw_Server_API/Config_Files/config.txt
- tldw_Server_API/app/core/Slides/standalone_html_config.py
- tldw_Server_API/app/core/Slides/standalone_html_registry.py
- tldw_Server_API/app/core/Utils/prompt_loader.py
- tldw_Server_API/app/core/config_sections/__init__.py
- tldw_Server_API/app/core/config_sections/slides.py
- tldw_Server_API/tests/Config/test_config_sections_typed_loaders.py
- tldw_Server_API/tests/Slides/test_standalone_html_config.py
- tldw_Server_API/tests/Slides/test_standalone_html_registry.py
- tldw_Server_API/app/core/Slides/standalone_html_sources.py
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/core/DB_Management/backends/base.py
- tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py
- tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py
- tldw_Server_API/app/core/DB_Management/chacha/message_store.py
- tldw_Server_API/app/core/DB_Management/chacha/note_store.py
- tldw_Server_API/app/core/DB_Management/media_db/api.py
- tldw_Server_API/app/core/DB_Management/media_db/repositories/media_lookup_repository.py
- tldw_Server_API/app/core/DB_Management/media_db/runtime/execution_ops.py
- tldw_Server_API/app/core/DB_Management/media_db/schema/backends/postgres_helpers.py
- tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py
- tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py
- tldw_Server_API/app/core/RAG/rag_service/profiles.py
- tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py
- tldw_Server_API/tests/Slides/test_standalone_html_sources.py
- tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py
- tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py
- tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py
- tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py
- tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py
- tldw_Server_API/tests/DB_Management/unit/test_postgresql_error_redaction.py
- tldw_Server_API/tests/RAG_NEW/unit/test_rag_profiles.py
- tldw_Server_API/tests/RAG_NEW/unit/test_reranker_trust_remote_code.py
- tldw_Server_API/tests/RAG_NEW/unit/test_preinstalled_local_reranker.py
- tldw_Server_API/tests/RAG_NEW/unit/test_slides_source_retrieval_hardening.py
- tldw_Server_API/app/core/Slides/standalone_html_provider.py
- tldw_Server_API/tests/Slides/test_standalone_html_provider.py
- tldw_Server_API/tests/Slides/test_standalone_html_generation.py
- tldw_Server_API/app/core/Jobs/worker_sdk.py
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/app/core/Jobs/migrations.py
- tldw_Server_API/app/core/Jobs/pg_migrations.py
- tldw_Server_API/tests/Jobs/test_worker_sdk.py
- tldw_Server_API/tests/Jobs/test_jobs_finalize_idempotency_sqlite.py
- tldw_Server_API/tests/Jobs/test_jobs_slides_generation_coordination_sqlite.py
- tldw_Server_API/tests/Jobs/test_jobs_slides_generation_coordination_postgres.py
- tldw_Server_API/app/core/Slides/standalone_html_service.py
- tldw_Server_API/app/services/standalone_html_generation_jobs_worker.py
- tldw_Server_API/tests/Slides/test_standalone_html_generation_jobs.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement a hardened standalone HTML+JavaScript presentation mode shared across existing Slides source types, with a form-first Presentation Studio flow, strict content-kind invariants, bounded LLM output, explicit-save editing, a text-only safe outline, attachment-only file handoff, compatibility guards, tests, documentation, and a firm no-execution boundary across every tldw surface.
<!-- SECTION:DESCRIPTION:END -->

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

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An approved design spec and implementation plan document the architecture, no-execution security boundary, compatibility behavior, and deferred scope.
- [ ] #2 The Slides backend supports structured_slides and standalone_html as explicit, validated content kinds without permitting split-brain records.
- [ ] #3 Standalone HTML generation uses one shared mode-aware service across supported source kinds, submission-time immutable source snapshots, and one administrator-configured concrete allowlisted provider/model/adapter/endpoint target.
- [ ] #4 Presentation Studio exposes a form-first HTML+JavaScript generation flow and a dedicated code, text-only safe-outline, save, conflict, recovery, and attachment-download experience.
- [ ] #5 Generated HTML/JavaScript is never rendered or executed by a tldw server, WebUI, extension, worker, MCP path, or renderer; source is never served as text/html.
- [ ] #6 Legacy presentations and clients remain structured by default, schema-v2 and version migrations are covered, and capabilities fail closed without blocking existing HTML read/edit/export.
- [ ] #7 Focused backend, frontend, security, integration, and E2E tests pass, and Bandit reports no new findings in touched Python.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-15-standalone-html-presentations-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-15 requester-approved V1 direction: standalone HTML+JavaScript is generated, stored, edited, versioned, and downloaded as opaque source. Presentation Studio exposes the form first and shows only a trusted text safe outline. Every tldw execution or fidelity-render path remains prohibited in V1.

2026-07-15 final design hardening: the shared backend binds each accepted generation to an owner-scoped public generation UUID, an immutable internal Jobs UUID, domain-separated HMAC receipts, an immutable source snapshot, and one server-selected concrete allowlisted provider/model/adapter/endpoint target. The design also specifies retrieval-only owner-local RAG, bounded source adapters and provider reads, killable validation and outline workers, raw octet-stream save and draft attachment paths, explicit content-kind negotiation for compatibility, crash recovery, and an emergency standalone-HTML egress kill.

2026-07-15 fresh re-review: backend, security, and product reviewers all returned APPROVED with no remaining P0-P3 or blocking design findings. Four embedded JSON contracts parse, Markdown fences are balanced, the related link resolves, heading hierarchy is valid, required hardened contracts are present, and stale superseded contracts are absent. This revision changes only documentation and Backlog metadata, so Python tests, frontend builds, and Bandit are not applicable. Implementation has not started; the next step is the task-specific implementation plan.
<!-- SECTION:NOTES:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-15 implementation plan approved: the five-stage, 17-task TDD plan locks the closed provider adapter catalog, external-secret/shared-Jobs-store key and reconciliation metadata, fenced receipt/worker recovery, guarded per-request MCP discovery and execution, and the inert form/editor/download boundary. A fresh independent plan review returned APPROVED after correcting the HTML slides=[] persistence invariant, shared Jobs coordination, guarded Uvicorn/WebSocket pins, exact outer-fence parsing, dependency smoke gates, per-commit Backlog staging, and mechanically complete Bandit scope. Backend Slides baseline: 100 passed with 5 warnings. The isolated frontend worktree had no installed workspace dependencies, so no frontend product test ran or failed; Task 13 begins with a frozen clean install and pre-change regression gate. Implementation code has not started.

2026-07-15 Task 1 implementation evidence: assertion-level TDD RED was 36 collected, 24 failed, 12 passed, 5 warnings in 8.92s. The owner-isolation review test separately went RED with 1 failed, 5 warnings in 8.86s before adding the owner-scoped input projection.

Final GREEN: schema/domain/database suite 36 passed, 5 warnings in 12.12s; existing Slides API regression 76 passed, 5 warnings in 12.95s. Bandit over slides_migrations.py, slides_db.py, and db_path_utils.py reported 0 findings and 0 errors.

Migration verification covers new/v0/v1/empty/multirow version state, normalization, idempotent reopen, injected rollback, concurrent connections, legacy backfill, per-statement execution, and FTS synchronization. The legacy failure was traced to a newly created external-content FTS index with 1 content row and 0 docsize rows; one transactional rebuild before backfill restored synchronization.

Deferred for the mandated root spec gate, not hidden: unrelated pre-v2 compatibility check/ALTER helpers remain process-local before the v2 runner; future-version rejection follows base/unrelated initialization; normalized v2 reopen rewrites the single schema_version row. Task 1 did not expand these because the approved spec permits unrelated legacy helpers to remain and root review will adjudicate scope.

2026-07-15 Task 1 schema-initialization review fixes: TDD RED proved future schema v3 was rejected only after structural mutation (1 failed, 5 warnings in 7.92s; PRAGMA schema_version changed 3→32) and normalized schema-v2 reopen attempted a write lock under a competing writer (1 failed, 5 warnings in 12.78s; database is locked). The migration runner now performs a nonmutating completeness/future-version probe before and after BEGIN IMMEDIATE, while SlidesDatabase probes the full base+v2 schema before locking and rechecks under the SQLite lock before any DDL or compatibility helper runs. Base DDL is executed one complete statement at a time so trigger bodies remain intact.

Fresh GREEN after final edits: schema/domain/database suite 38 passed, 5 warnings in 7.65s; Slides API regression 76 passed, 5 warnings in 10.48s. Bandit exited 0 over slides_migrations.py, slides_db.py, and db_path_utils.py with only existing nosec notices; git diff --check passed. Regression coverage confirms future-version rejection is structurally read-only, normalized v2 reopen succeeds without a write lock or data-version change, and all three FTS triggers exist.

The prior compatibility-helper concern is no longer deferred: the process-local schema cache/lock was removed, and compatibility helpers are serialized inside BEGIN IMMEDIATE with a second completeness/future-version check before mutation.

2026-07-15 Task 1 closure: commits 285480902033bff715ba45e1eac18404f4385b2a and 73318e36ed8196d3bed82e4702b158d5e8bd881f. Fresh specification review returned ✅ Spec compliant. The quality re-review found no Critical or Important findings and explicitly approved proceeding to Task 2. Two nonblocking cleanup notes remain for a later relevant persistence touchpoint: refine generation-job uniqueness error translation before worker persistence and optionally consolidate duplicated summary SQL.

2026-07-15 Task 2 started: building the authoritative no-execution html5lib/tinycss2 validator and the bounded killable subprocess pool under assertion-level TDD.

2026-07-16 Task 2 validator/pool implementation evidence: initial assertion-level RED was 86 collected, 22 failed, 64 passed, 5 warnings. Security-review RED tranches then proved character-reference and semicolonless-CSS preflight gaps, semantic adjacency loss, malformed-attribute nontermination, namespace/script/style bypasses, template interpolation and bracket/regex sink gaps, worker diagnostic loss, malformed IPC capacity stranding, dead-worker provider admission, unused-reservation leakage, and readiness/reap races before each fix.

Implementation now provides a parser-only, source-redacted html5lib/tinycss2 validator with explicit byte/token/tree/CSS/text budgets, namespace-aware active/resource rejection, bounded diagnostic JavaScript sink lexing, and frozen derived metadata. A supervised maximum-four-worker process pool provides a bounded 24-item interactive queue, eight pre-provider generation reservations, weighted 3:1 scheduling, watchdog/cancellation replacement, confirmed reap-before-replace, closed bounded IPC, tracked reservation cleanup, total malformed-response handling, and a spawn readiness handshake. Slides package exports are a backward-compatible PEP 562 lazy facade so spawn imports do not initialize Slides DB/export/generator services. Direct dependencies are exactly pinned and installed: html5lib==1.1 and tinycss2==1.4.0.

Final verification: combined validator/pool/dependency smoke 140 passed, 3 warnings, exit 0 in 30.78s; isolated validator+smoke 121 passed; isolated pool 19 passed. Bounded existing Slides DB/generator/export regression coverage had 40 passes; two custom.css export assertions fail identically in the unchanged Task 1/base eager-import worktree, so they are recorded baseline/environment failures rather than Task 2 regressions. Production-only Bandit exited 0 with 0 findings across all four touched production Python files; the all-touched-Python Bandit run had 0 medium/high findings and only test-idiom LOW findings. Black and Ruff pass; git diff --check passes. pip check is honestly nonzero in the shared pre-existing venv for five unrelated dependency drifts: faster-whisper 1.2.0 vs >=1.2.1, sentence-transformers 5.2.3 vs >=5.4.0, torch 2.3.0 vs >=2.11.0, transformers 4.57.6 vs >=5.5.3, and typer 0.16.1 vs typer-slim's >=0.24.0 requirement. The two new exact dependencies themselves are installed at the required versions.
2026-07-16 Task 2 final audit closure: the URL-policy follow-up first ran RED with 14 selected validator cases (10 expected failures, 4 passes), proving missing RDFa about/resource/vocab/prefix coverage, SVG color-profile URL coverage, and generic unmistakable URL markers in arbitrary attributes. After the narrow fail-closed fix, the same tranche passed 14/14 while preserving normal SVG xmlns, hex colors, and benign #label or /label values in non-resource attributes. The independent URL-policy re-review returned APPROVED with no Critical or Important findings. Final focused validator/pool/dependency-smoke verification passed 154/154 with 3 existing warnings in 30.69s. Black completed and Ruff reported all checks passed. Production-only Bandit exited 0 with 0 findings, 0 skipped tests, and 2,191 LOC scanned across Slides/__init__.py, standalone_html_contracts.py, standalone_html_validator.py, and standalone_html_validation_pool.py. Tracked git diff --check exited 0, and every untracked Task 2 file had empty no-index whitespace diagnostics. Exact html5lib==1.1 and tinycss2==1.4.0 dependency pins remain installed. Task 2 is ready for its authorized commit; overall TASK-12115 remains In Progress for later implementation stages.
2026-07-16 Task 2 post-commit specification review closure: review of f860f1aa26351520ee8df503ac2b5c2a5955c668 reported seven Important blockers and no broader Task 3+ scope: (1) CSS preflight split oversized raw URL/function tokens before the 65,536-byte check; (2) slide discovery was ancestor-blind for template, notes, deck chrome, excluded subtrees, and nested slides, allowing count/text leakage; (3) the bounded JavaScript diagnostic matcher missed qualified constructors and simple aliases; (4) the watchdog timed only poll(), leaving send()/recv() outside the bound; (5) result projection/serialization exceptions could print attacker-controlled child tracebacks; (6) cancellation during close could strand a closed slot with _closing=True; and (7) response decoding accepted noncanonical titles, source-mismatched bytes/digests, and incompatible error status/retry/reason combinations.

Assertion-level validator RED for findings 1-3 selected 24 cases: 16 failed and 8 passed in 7.96s. The failures proved function/URL raw-token max+1, all hidden-only and hidden/nested slide count/text cases, qualified constructors, simple aliases, and alias-source redaction. Existing comment/string boundaries, ordinary wrapper-contained slides, safe aliases, and already-recognized qualified fetch behavior were positive controls. Minimal GREEN checkpoints were CSS raw lexical boundaries 4/4, excluded/nested slide discovery and extraction 9/9, and qualified/simple-alias diagnostic handling plus redaction 11/11; the exact combined tranche then passed 24/24 with 3 existing warnings in 7.08s.

Assertion-level pool RED for findings 4-7 selected 14 cases and failed 14/14 in 9.25s. A corrected serialization readiness assertion was rerun separately and failed on the intended TOP-SECRET-POSTRETURN traceback leak, not setup. The blocked-send and framed-partial-receive cases each remained bounded by a one-second external test timeout and initially surfaced TimeoutError, proving that the internal watchdog missed the I/O phase without hanging the suite. GREEN checkpoints: child result projection/serialization redaction 1/1; double-cancel terminal close plus existing failed-reap retry 2/2 with _closed=True, _closing=False, empty slots/PIDs, zero counters, and idempotent retry; malformed semantic response matrix 10/10 with PID replacement, capacity release, and recovery; whole-roundtrip blocked-send/partial-receive 2/2 with fixed timeout errors, old-PID reap/replacement, successful recovery, and a tracked _rpc_sync finally event set before resolution, proving no blocked I/O work remained. The exact combined pool follow-up tranche passed 14/14 with 3 existing warnings in 7.48s.

Final verification after formatting/lint cleanup: complete pool suite including default spawn passed 33/33 with 3 warnings in 11.49s. Fresh full validator/pool/dependency-smoke verification passed 192/192 with 3 warnings in 16.60s. Black --check left all four touched Python files unchanged and Ruff reported all checks passed. Production-only Bandit exited 0 with 0 findings, 0 skipped tests, and 2,444 LOC scanned across Slides/__init__.py, standalone_html_contracts.py, standalone_html_validator.py, and standalone_html_validation_pool.py. git diff --check exited 0. A narrowly filtered pgrep check found no remaining standalone-HTML pytest or validator process; the IPC regressions independently assert the tracked executor work finishes before timeout resolution. The bounded neighboring Slides DB/generator/export suite remained 40 passed and the same two assets/custom.css failures already reproduced on unchanged base, so no new adjacent regression was introduced.

The seven Important findings are implemented and their focused/full gates are green. Per root coordination, a fresh independent reviewer will inspect the follow-up commit after it is created because the prior audit child was explicitly closed and declined a new turn; this note does not claim that future review in advance. Overall TASK-12115 remains In Progress for Tasks 3+.
2026-07-16 Task 2 third review closure: fresh review of ce5ddc97bc7dcc8e4c5ef571c406538358bb5597 identified the remaining standalone trust-boundary gaps: CSS preflight split at-keyword, hash/name, signed/decimal/exponent number, percentage/dimension, non-ASCII and escaped identifier, function/URL, string/comment token families; successful IPC tuples lacked the immutable 1,048,576-byte ceiling; error locations were not closed by reason; root-level chrome and hidden/nested slide accounting disagreed; qualified/aliased popup and navigation diagnostics were incomplete; and a terminal CSS backslash could fail to advance. No Task 3+ scope was changed.

Assertion-level RED was captured before production edits. The expanded validator tranche selected 50 cases and produced 28 expected failures / 22 positive-control passes in 7.90s; a quoted-url function/string separation control then failed 1/1 on the intended over-combination. The pool tranche selected 19 cases and produced 4 expected failures / 15 positive-control passes in 7.79s: internally consistent 1,048,577-byte success, non-parser locations, and parser half-pairs were accepted, while exact-ceiling success, bounded parser pairs, and out-of-range rejection were controls.

The minimal fix now uses CSS Syntax-aligned name/escape/number span accounting with guaranteed linear progress, distinct quoted-function/string handling, and conservative unquoted URL spans; counts every XHTML section.slide toward the hard 30-element limit while deriving only eligible non-chrome slides; and recognizes closed obvious qualified/alias sink token sequences for open, location navigation, beacon, history, cache, and service-worker registration. Pool decoding rejects oversized successes before metadata construction and permits a bounded line+column pair only for html_parse_error/css_parse_error, with all other locations absent.

Targeted GREEN: validator 51/51 passed with 143 deselected and 3 warnings in 7.01s; pool 19/19 passed with 23 deselected and 3 warnings in 7.19s. Fresh final validator/pool/dependency-smoke verification passed 237/237 with 3 warnings in 16.74s. Black --check left all four touched Python files unchanged and Ruff reported all checks passed. Production-only Bandit exited 0 with 0 findings, 0 skipped tests, and 2,562 LOC across Slides/__init__.py, standalone_html_contracts.py, standalone_html_validator.py, and standalone_html_validation_pool.py. git diff --check passed. Narrow pgrep found no residual standalone validator or pytest processes; the full pool regressions also assert tracked RPC executor work terminates before timeout resolution.

The bounded neighboring Slides DB/generator/export suite, rerun alone to avoid cross-pytest SQLite contention, produced 40 passes and exactly the two documented unchanged-base failures: test_export_bundle_includes_assets and test_export_bundle_stamps_style_hooks_and_includes_builtin_pack_css, both because assets/custom.css is absent. Overall TASK-12115 remains In Progress for Tasks 3+.
2026-07-16 Task 2 direct-navigation sink closure: final specification review of 936a33ec6f1aa678fdc574a4fb57a1cabb86d195 found one remaining Important gap in the bounded JavaScript diagnostic catalog: plainly identifiable direct open(), history.go/back/forward(), location.reload(), and Navigation API navigate/reload/traverseTo/back/forward() calls were accepted. Qualified and simple-alias forms, including window/self/globalThis, top/parent location navigation, and document.location, also needed canonical coverage without turning the diagnostic into a JavaScript parser.

Assertion-level RED was captured before production edits: 53 selected cases produced 36 expected failures and 17 positive-control passes, with 193 deselected and 3 warnings in 16.19s. The failures covered direct, qualified, document/top/parent, Navigation API, and alias paths plus the new source-redaction path. The positive controls proved already-recognized popup cases and navigation words in strings, comments, unrelated object methods, and unrelated aliases remained inert.

The minimal fix keeps shallow token diagnostics: window/self/globalThis/top/parent qualifiers canonicalize through the existing global normalizer; the closed direct and alias tables now include exact popup/history/location/document.location/Navigation API members; and direct-pattern matching requires that a candidate not be a property of an unrelated object. No general parsing, scope resolution, or dataflow was added. The redundant explicit window.location assignment pattern was removed because normalization already supplies the canonical location form.

Targeted GREEN passed 53/53 with 193 deselected and 3 warnings in 6.91s. Fresh final validator/pool/dependency-smoke verification passed 289/289 with 3 warnings in 16.50s. Black --check left both scoped Python files unchanged and Ruff reported all checks passed. Production-only Bandit exited 0 with 0 findings, 0 skipped tests, and 2,588 LOC across Slides/__init__.py, standalone_html_contracts.py, standalone_html_validator.py, and standalone_html_validation_pool.py. git diff --check passed, and narrowly filtered pgrep found no residual standalone validator or pytest processes. No Task 3+ scope was touched; overall TASK-12115 remains In Progress.
2026-07-16 Task 2 bounded-input/diagnostic closure: the latest code-quality review identified three Important trust-boundary defects and one narrow title-control gap: Unicode-expanding lowercasing corrupted HTML preflight offsets; the JavaScript diagnostic lexer silently truncated after 50,000 tokens; and document byte/UTF-8 limits were enforced only after full conversion and after pool startup/queue admission. No Task 3+ behavior was changed.

Assertion-level TDD RED was captured before production edits in three bounded groups. Unicode offsets/title bidi produced 7 failures and 8 positive-control passes (U+0130 drift plus U+206A-U+206F acceptance). Script diagnostics produced 3/3 expected failures (max+1 truncation, no-sink overage, and a late fetch sink). Size/pre-IPC produced 7 failures and 3 positive-control passes (oversized str/bytes subclasses invoked conversion, four public invalid inputs attempted worker startup, and a reserved invalid input entered RPC).

The minimal fix adds one shared bounded document-input preflight used before validator copies and before pool startup/queue/IPC. Bytes are length-gated before strict decoding; strings use an O(1) code-point ceiling followed by bounded strict UTF-8 chunk measurement; built-in base methods avoid attacker-controlled subclass conversion hooks. HTML syntax now uses length-preserving ASCII-only A-Z folding, script tokens raise the stable html_tokens budget error immediately at max+1, generation reservations remain unconsumed on rejected source, and title validation includes U+206A-U+206F.

Targeted GREEN passed 28/28. The complete validator/pool suite passed 307/307. After Black formatting, the bounded dependency-smoke, migration, domain, generator, validator, and pool suite passed 344/344 with only existing warnings. Ruff passed all four touched files. Production-only Bandit scanned 2,565 LOC with 0 findings, 0 nosec suppressions, and 0 skipped tests. git diff --check passed; only standalone_html_validator.py, standalone_html_validation_pool.py, and their two test files are modified. A filtered process audit found no remaining validator, pytest, or child Python process.

Nonblocking debt deliberately not expanded in this security patch: pool/validator policy constants remain duplicated, and existing unused internal response/state fields remain available for a later relevant simplification. Overall TASK-12115 remains In Progress for Tasks 3+.
Backlog scope clarification: the final commit contains the four scoped validator/pool code and test files plus this required TASK-12115 record; no other repository file is modified.

2026-07-16 Task 2 closure: commits f860f1aa26351520ee8df503ac2b5c2a5955c668, ce5ddc97bc7dcc8e4c5ef571c406538358bb5597, 936a33ec6f1aa678fdc574a4fb57a1cabb86d195, 6bb58289538f0f256ac7bd9f8336f4f7da8052d0, and 788255d18639698a9a16cceeeaf539dda62b3c5f. Final specification review returned ✅ Spec compliant. The quality re-review found no Critical or Important findings and explicitly approved proceeding to Task 3. Final focused and neighboring verification passed 344 tests; Black, Ruff, Bandit, git diff, and residual-process gates were clean. Two nonblocking quality debts remain for a later relevant cleanup: duplicated closed validator/pool policy constants and unused _HtmlPreflight, queue_kind, and RPC argument fields.
2026-07-16 Task 3 domain/REST/export guard implementation evidence: assertion-level TDD started with 56 collected, 26 expected failures, 30 passes, and 5 warnings. The shared PresentationService now owns closed content-kind negotiation, source-free guards/projections, worker-only standalone creation, raw octet-stream source save, same-kind restore, source-free deletion, and operation allowlisting. SQLite persistence enforces immutable kind/generation identity, authoritative standalone validation and derived metadata, compact UTF-8 active-kind snapshots with a standalone-only storage ceiling and bounded retention, generation-job conflict translation, pre-pagination kind filtering for list/search, and a single alias-aware summary projection builder. The standalone restore DB path is explicitly standalone-only; structured restore preserves the legacy API normalization/update pipeline including image alt-text indexing.

The REST schema and routes preserve exact legacy structured response shapes unless standalone_html is explicitly accepted, emit Vary on negotiated success/error paths, use strong standalone and weak structured ETags, guard detail/version/search/render/artifact/export routes before source loading or dispatch, reject explicit kind changes with stable content_kind_immutable ordering, expose explicit raw HTML source save, return source-free version/delete metadata, support UTF-8 discriminated JSON export, and keep HTML attachment transport plus all interactive execution/render/preview behavior deferred. Generic create/mutation remains fail-closed for standalone fields.

Final focused Task 3 suite: 63 collected, 61 passed, with exactly two unchanged-base export fixture failures because assets/custom.css is absent (test_export_bundle_includes_assets and test_export_bundle_stamps_style_hooks_and_includes_builtin_pack_css); no Task 3 test failed. Final structured regression with the previously failing random seed: 108 collected, 106 passed, with only those same two baseline failures. A one-off schema-concurrency SQLite lock was investigated without code changes: isolated 1/1, three bounded repetitions 3/3, the preceding rollback+concurrency slice 2/2, and the seeded full rerun all passed. Targeted cleanup/error-order regressions passed 5/5.

Black --check left all eight scoped Python files unchanged after formatting; Ruff reported all checks passed. Production-only Bandit exited 0 with 0 findings across five production files (5,703 LOC; eight justified skipped SQL checks). git diff --check passed and no residual pytest/validator/Bandit process remained. Overall TASK-12115 remains In Progress for Task 4+; Task 3 adds no execution, rendering, preview, generation worker, or attachment transport.
2026-07-16 Task 3 review-fix closure: assertion-level follow-ups hardened the domain/REST/export boundary without expanding into Task 4. Generic persistence mutations now reject every standalone_html change except exact delete/restore state transitions; standalone save/restore accept only immutable pool validation results whose bytes, digest, and derived metadata are rechecked before a short atomic write. Restore and JSON export also verify exact stored payload/provenance/metadata, preserve structured missing-version precedence, and avoid parsing irrelevant structured fields for standalone responses. The single app-owned StandaloneHtmlValidationPool is lazily shared through app.state for interactive requests and the future Task 8 worker, validated outside writer transactions, asynchronously closed at lifespan teardown, and fully removed with its lock; no worker-local pool is introduced.

Negotiated routes now establish target/operation compatibility before If-Match parsing and centrally add Vary: X-Slides-Accept-Content-Kinds on all success and downstream HTTP-error paths. Closed-operation errors expose bounded operation plus the actual bounded content-kind token while retaining the compatibility detail string. Version-payload and encoding failures discard attacker-controlled exception context before returning stable source-free errors. Regression coverage includes retention=25, rollback at the standalone snapshot ceiling, corrupt/mismatched validation artifacts, lifecycle teardown, all seven negotiated mutations, structured legacy precedence, downstream Vary propagation, source-free error chains, and future-kind metadata.

Fresh focused verification after the final fixes: standalone domain/API 66 passed with 5 existing warnings. The earlier combined export and structured suites remain 81/83 and 106/108 respectively, with only the two unchanged-base assets/custom.css fixture failures already reproduced on base. Black --check leaves all five scoped files unchanged; Ruff passes; git diff --check passes. Production-only Bandit scanned slides.py, presentation_service.py, and slides_db.py (4,634 LOC) with 0 findings and eight existing justified SQL skips; report /tmp/bandit_task12115_task3_review_fixes.json. Overall TASK-12115 remains In Progress for Task 4+.

2026-07-16 Task 3 final correction closure: follow-up TDD fixed four independent review blockers and one migration-audit blocker without expanding Task 3 scope. Validator-pool teardown now tolerates an unused app-owned pool; negotiated FastAPI request-validation errors retain the framework's exact 422 body while adding `Vary: X-Slides-Accept-Content-Kinds`; standalone soft-delete restore revalidates the exact stored source outside the SQLite writer and atomically rechecks source, version, deletion state, and all derived metadata before restoring; and source-free version projections preserve legacy title/deleted values through denormalized snapshot metadata. Schema-v2 completeness, partial-v2 migration, rollback, concurrent migration, and idempotent reopen cover the added version metadata columns. Backfill uses nested `json_valid`/`json_type` guards so malformed or type-confused legacy title/deleted values become NULL rather than leaking nested payload text or coercing invalid truthy state.

Post-correction verification passed 88/88 migration/domain/API tests. The structured DB/API/export suite passed 106/108, with only the same two base-reproduced failures caused by the absent `assets/custom.css` fixture; no Task 3 regression failed. Exact-source race, FTS, version snapshot, and sync-log restore checks pass. Black and Ruff are clean; `git diff --check` is clean. Production-only Bandit scanned 5,051 LOC with 0 findings and eight existing justified SQL skips. The independent migration audit reports no remaining Critical or Important findings. Overall TASK-12115 remains In Progress for Task 4+ and interactive HTML execution remains disabled.

2026-07-16 Task 3 final follow-up closure: two fresh reviewers found that successful standalone soft-undelete lacked the source-bearing `private, no-store`/`nosniff` response policy and that saved JSON export checked only validator-derived metadata rather than the complete persisted standalone invariant. Test-first fixes now apply those headers only to standalone restore responses and reject malformed/nonempty/non-list `slides`, missing/blank generation job identity, and missing/malformed/empty/nonobject/oversized provenance before validator-pool admission with a fixed chain-free `standalone_html_response_invalid`. Valid rows still run through the authoritative subprocess pool. Routine Slides and Research Workspace health checks now use a source-free `SELECT 1` probe with fixed SQLite failure normalization instead of materializing detail rows. Deeply nested stored version JSON now maps `RecursionError` to the same chain-free `version_payload_invalid` behavior for domain decode, version GET, and restore. The stale structured sanitization fixture was aligned with Task 3's discriminated response contract.

Final follow-up evidence: assertion-level RED was 15/15 for the initial review findings, 5/5 for boundedness/health hardening, and 3/3 for recursive snapshot handling. Focused GREEN passed 20/20, and the combined standalone/health/capability/sanitization suite passed 446/446. Structured regressions passed 106 tests with only the same two base-reproduced `assets/custom.css` failures. Black and Ruff are clean on the touched Slides scope; existing Research Workspace whole-file formatter/lint debt reproduces unchanged at HEAD. Production-only Bandit exits 0 with no findings, and `git diff --check` is clean. Fresh specification and quality re-reviews both returned READY with 0 Critical, 0 Important, and 0 Minor findings. Task 3 is complete; Task 4 may begin. Interactive HTML execution remains disabled.

2026-07-16 Task 4 closure: assertion-level TDD began with two collection errors because the closed standalone configuration and typed Slides section did not exist; later RED tranches reproduced each prompt, configuration, keyring, rotation, and retirement hardening issue before its fix. The implementation now provides a default-off standalone generation gate with an independent egress kill, six fixed application-owned provider adapters, exact case-sensitive provider/model/adapter allowlisting with derived endpoints, bounded typed limits and timeouts, a deterministic canonical UTF-8 generation revision, and strict fail-closed loading of the maintained 128 KiB `slides.standalone_html.v1` prompt. The adapted prompt requires self-contained HTML, the normative twelve narrative flows, keyboard navigation, reduced motion, current-slide speaker notes, accessibility, no autoplay, and no external or executable tldw-side resource path.

The environment-only HMAC keyring accepts one to four canonical 32-byte secrets, pins five versioned domains and a known-answer vector, hides secret and digest material from representations, and fails the full generation gate when any shared current/retiring secret is absent. Its injected source-free Jobs-store interface supports explicit current-key activation, distinct local-versus-identical-winner CAS outcomes, exact transition validation, bounded rotation, and removal only after the 32-day floor plus a complete same-epoch fenced dormant-database sweep. No source, prompt, provider credential, secret, or HMAC value enters registry store calls. Interactive execution remains disabled.

Final verification: the combined Task 4 configuration, registry, and typed-loader suite passed 193/193 with 3 existing warnings; legacy prompt-loader regressions passed 10/10 with 3 existing warnings. Black left all eight touched Python files unchanged, Ruff passed, `git diff --check` passed, and production-only Bandit reported 0 findings and 0 errors across 1,765 LOC. Fresh independent specification and quality reviews both returned READY with 0 Critical, 0 Important, and 0 Minor findings. Task 4 is complete; overall TASK-12115 remains In Progress for Task 5+.
2026-07-16 Task 5 closure: implemented one bounded, immutable source-snapshot path for prompt, chat, media, notes, and retrieval-only RAG. Chat projections use one statement snapshot; notes and media use owner-scoped max+1 projections with explicit invalid/truncation markers; media preserves normalized-transcript/document/media fallback behavior on SQLite and PostgreSQL; and database failures cross source-redacted boundaries. The closed slides_source_retrieval_v1 profile disables generation, rewriting, decomposition, adaptive reruns, remote/web fallbacks, and request-time downloads; it formats only bounded local documents and permits only preinstalled local flashrank, cross_encoder, or none reranking. Raw RAG query bounds are enforced before trimming, all source families participate fairly, and generated_answer is never consumed. Interactive HTML execution remains disabled.

Final verification: the 14-file affected matrix passed 589/589 with 11 existing warnings. Ruff passed the scoped files; Black left all five entirely new files unchanged after formatting; git diff --check passed. Production-only Bandit reported 0 findings across the full touched scope (report /tmp/bandit_task12115_task5_final.json). Fresh independent specification/security review returned APPROVED with no remaining P0-P2 findings. Task 5 is complete; overall TASK-12115 remains In Progress for Task 6+.
2026-07-16 Task 6 closure: implemented one isolated asynchronous provider transport shared by the six closed standalone-HTML adapters. The path uses exact fixed endpoint identities and closed provider codecs, local HTTPX with trust_env/redirects disabled, identity-only raw streaming, fixed response/JSON/document budgets, source-free typed failures, exactly one call with no fallback, and immediate post-client-entry feature/egress/tuple revalidation. Fresh connect/read/overall/token/output limits come from one attempt snapshot; Python-3.10-compatible AnyIO timeout scoping enters HTTPX's lazy stream in the verified task without an extra scheduling turn. Interactive HTML execution remains disabled. Assertion-level TDD began with 60 failures, and later RED tests reproduced both client-entry and lazy-stream races plus stale overall-timeout behavior. Final root verification passed 123 focused provider/generation tests and 761 full standalone-HTML tests. Black, Ruff, py_compile, and git diff --check passed. Production-only Bandit scanned 518 LOC with 0 findings and 0 errors. Fresh independent full-range specification/quality review reported no Critical, Important, or Minor findings and READY. Commits: 6e1220fe3b and a37a484718. Overall TASK-12115 remains In Progress for Task 7+.
2026-07-16 Task 7 closure: implemented UUID-authoritative Slides generation coordination across active/archive Jobs, shared HMAC-key registry reconciliation, fenced lease and terminal CAS behavior, exact-once counters/events, fail-closed migration/readiness audits, immutable generation scope, serialized replay before every admission rejection, and PostgreSQL/SQLite pruning and archive parity. Interactive execution remains disabled.

TDD and verification: the final queue-policy winner race reproduced first for active and archive rows (2 failed) and passed after rejection-path serialized replay (2 passed). The exact four-file Task 7 suite passed 94 tests with 21 repository-managed PostgreSQL fixture/runtime skips and 3 warnings across 115 collected tests. Ruff, Black, py_compile, git diff checks, and production Bandit were clean; Bandit reported 0 findings and 0 errors across 12,025 LOC at /tmp/bandit_task12115_root_final5.json. Immutable package review (SHA-256 d5feeba87cf09072b73a939ee9a59cb7c3c3f7a120705db95ac30c78f4517b41) returned Ready to merge: Yes with no blockers.

Task 7 commits: 30fabc2bfd, 178ef50ebb, f2aad07ec7, 076742ba27, d2533d9d81, 7495834658. Overall TASK-12115 remains In Progress for Tasks 8+.
2026-07-18 Task 8 implementation evidence: receipt-backed standalone HTML admission and worker commit paths are implemented. Jobs payloads remain receipt-only; owner-scoped receipt/input claims precede enqueue; active/archive recovery binds the immutable Jobs UUID and paired numeric hint; nonterminal replay verifies immutable source, options, prompt, target, provenance, and the canonical +24-hour input deadline; terminal CAS deletes input and retains receipts for +30 days. The worker acquires validator capacity before egress, rebuilds exact option-bearing user content, enforces the stored-target allowlist/kill/key gates, checks exhaustion/quarantine and live Jobs state before provider and commit, and atomically commits one presentation with completed-winner precedence. Interactive execution remains disabled and Task 9 lifecycle wiring was not added.

TDD evidence: expanded RED collected 73 cases with 38 failed, 35 passed, 5 warnings (seed 3066732774); focused GREEN passed 73/73 with 5 warnings (seed 636255254). Final post-fix Task 8 plus structured-render regression passed 80/80 with 5 warnings in 10.64s (seed 381783820). Black --check left all four scoped files unchanged, Ruff passed, py_compile passed, and git diff --check passed. Production-only Bandit exited 0 with 0 findings and 0 errors across 4,049 LOC; 13 existing narrow SQL checks were skipped and no nosec suppressions were counted (report /tmp/bandit_task_8.json).

Cohesion/ponytail review consolidated duplicated pre-provider and pre-commit Jobs checks into _fence_job. Remaining explicit receipt/CAS branches preserve distinct retry, terminal, completed-winner, and commit-conflict semantics. Overall TASK-12115 remains In Progress for Task 9+; root review gates and final plan/DoD checkboxes remain pending.

2026-07-18 Task 8 review-hardening closure: preserved the receipt-backed no-execution boundary while hardening terminal-first worker races, fresh digest-key snapshot fences, deterministic admission errors, exact UUID/HMAC correlation, archived-job authority, transactional expiry/provenance checks, retry/commit CAS winner handling, and nullable archived numeric IDs. Distinct archived UUIDs sharing an idempotency scope are now valid; duplicate archive UUIDs and malformed legacy identity still fail closed. Task 9 lifecycle wiring remains deferred and interactive execution remains disabled.

Fresh verification on the final semantic-only diff: focused Jobs regressions passed 93/93 with 3 warnings (seed 338737102). The affected Task 7/Task 8/render matrix passed 210 tests with 24 repository-managed PostgreSQL environment skips and 5 warnings in 33.44s (seed 389374523); no failures or errors. py_compile passed all 12 touched files, git diff --check passed, and production-only Bandit scanned 15,588 LOC with 0 findings and 0 errors. Ruff is clean on all new/changed semantics; the full touched-file check reports only restored baseline B023/F401/F841 debt in worker_sdk.py/test_worker_sdk.py. Black --check leaves seven changed semantic files unchanged and reports only the deliberately unformatted baseline in manager.py, migrations.py, pg_migrations.py, worker_sdk.py, and test_worker_sdk.py; broad formatter churn was removed after review. Overall TASK-12115 remains In Progress for Task 9+.
2026-07-18 Task 8 final review-fix evidence: closed every confirmed lifecycle/security review issue without starting Task 9. The worker now loads its final digest snapshot before a post-lookup lease/cancel fence and has no await before synchronous commit; provider admission rechecks the absolute input deadline after digest/config/credential work and after the Jobs lookup. Worker/reconciler terminal races preserve reconciler winners while exact worker replays remain conflict-safe. Archive replay queries are bounded, canonical provenance is authenticated by the execution HMAC, worker terminalization clears stale last_error, and PostgreSQL/SQLite quarantine status, streak, availability, timestamps, and counters all use one effective same-error result (including omitted error_code).

TDD reproduced the defects before fixes: cancellation and lease loss during the fourth digest load committed output; reconciler-first CAS returned CONFLICT; archive reads lacked LIMIT; changed error codes quarantined early; provenance source_ref tampering reached egress; last_error stayed stale; omitted error_code never quarantined; and digest/Jobs lookup clock advances allowed expired provider admission or stale lease commit. New regressions cover those cases plus a real two-connection atomic claim race, exact HMAC comparison pairs, deterministic 422-before-413 payload validation, and a real retry-then-success WorkerSDK path.

Fresh verification: standalone generation 113/113 passed. The affected Task 7/Task 8/render/quarantine matrix passed 226 tests with 26 repository-managed PostgreSQL skips and 5 warnings (seed 2544557896). py_compile and git diff --check passed. Ruff passes all changed semantics; the full touched-file run reports only five unchanged baseline F401/F841 findings in legacy tests. Black leaves every new/changed hunk unchanged; its only output is surrounding legacy manager/quarantine formatting outside this diff. Production Bandit scanned 11,234 LOC with 0 findings and 0 errors at /private/tmp/bandit_task12115_task8_review_fixes.json. Independent SQL and security re-audits report no remaining Critical or Important findings. Interactive execution remains disabled, Task 9 lifecycle wiring remains deferred, and TASK-12115 remains In Progress.

2026-07-18 Task 8 final immutable review closure: commits 169bdf9b20, 56644c7f3b, and fb470bb0b6 implement and harden the receipt-backed standalone HTML generation worker path. Fresh specification review of fb470bb0b6 found no Critical or Important blockers and independently passed 212 tests; fresh quality review of 56644c7f3b..fb470bb0b6 found no Critical or Important blockers and independently passed 211 tests. Final local evidence remains 113 focused tests passed, 226 affected tests passed with 26 repository-managed PostgreSQL skips, py_compile/diff/scoped Ruff/changed-hunk Black clean, and Bandit 0 findings across 11,234 lines. Task 8 is complete. TASK-12115 remains In Progress for Task 9 and later work; Task 9 was not started and interactive execution remains disabled.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
