---
id: TASK-12115
title: Add first-class standalone HTML-JS presentation generation
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-07-16 01:50
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
- tldw_Server_API/tests/Slides/test_standalone_html_db_migration.py
- tldw_Server_API/tests/Slides/test_standalone_html_domain.py
- tldw_Server_API/tests/Slides/test_slides_db.py
- tldw_Server_API/tests/Slides/test_standalone_html_validator.py
- tldw_Server_API/tests/Slides/test_standalone_html_validation_pool.py
- tldw_Server_API/tests/Slides/test_standalone_html_dependency_smoke.py
- pyproject.toml
- backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md
- tldw_Server_API/app/core/Slides/__init__.py
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
