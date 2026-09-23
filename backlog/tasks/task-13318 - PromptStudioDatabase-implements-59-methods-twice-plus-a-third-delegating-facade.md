---
id: TASK-13318
title: >-
  PromptStudioDatabase implements 59 methods twice plus a third delegating
  facade
status: In Progress
assignee: []
created_date: '2026-09-22 04:55'
updated_date: '2026-09-23 15:30'
labels:
  - duplication
  - db
  - dual-backend
dependencies: []
references:
  - 'tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py:722'
  - 'tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py:3848'
  - 'tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py:7144'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A 7,426-line file where _BackendPromptStudioDatabase (722-3843, 3,121 lines) and _SQLitePromptStudioDatabase (3848-7141, 3,293 lines) implement 59 identically-named methods over parallel SQL, and PromptStudioDatabase (7144-7426) is a third *args/**kwargs delegating facade redeclaring 43 of them.

Observed drift rate already: 1 missing method (TASK-13290), 7 signature mismatches, and a retry policy on one side only. Three signature mismatches reproduced at runtime; _format_test_case differs in ARITY, (row) vs (cursor, row), so any shared helper calling it polymorphically is wrong on one backend by construction. The facade *args/**kwargs erases all of this from mypy and every IDE.

No cross-backend parity test exists over the 59 methods. Of 34 test files, 9 touch PostgreSQL and all are plumbing-level.

Destination: a prompt_studio_db/ package, one module per aggregate, ONE backend-neutral implementation over DatabaseBackend with dialect SQL isolated - following the already-shipped core/DB_Management/media_db/ split of Media_DB_v2.py. Needs design doc + ADR + staged plan.

Source: synthesis F20
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design doc and ADR recorded before code changes
- [x] #2 A signature-parity test covers all paired methods
- [x] #3 Business logic exists once, with only SQL differing per backend
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC2 done in 0205612478 (signature-parity ratchet, 7 known mismatches frozen, verified to bite). AC1 done: Docs/Design/2026-09-23-prompt-studio-db-consolidation-design.md + ADR-051 (Proposed). AC3 (logic exists once) is the multi-stage refactor the design stages; not started, pending the owner's choice of option. Notable finding: _BackendPromptStudioDatabase already branches on backend_type and could run on a SQLite DatabaseBackend -- a faster but riskier route, documented as Option A.

2026-09-23: decision 3 done — four latent public signature mismatches aligned (get_prompt include_deleted, create_bulk_test_cases client_id on SQLite; delete_signature hard_delete and list_evaluations filters keyword-only on both). Parity drift list down to 3 private entries. prompt_studio suite 1145 passed. Next: Stage 1 package skeleton.

2026-09-23: Stage 2 done — tests/prompt_studio/test_backend_behaviour_parity.py (scenarios: test_runs, prompt_versions, evaluations, reads) on SQLite + live PG. Found and fixed PG leaking *_tsv columns from 8 read paths. Stage 1 folded into Stage 3 (session = legacy object, as in media_db). Next: Stage 3 — move test runs, prompt versions, evaluations into prompt_studio_db/repositories/.

2026-09-23 (d27f06a143): Stage 3 done — test runs, prompt versions, evaluations in prompt_studio_db/repositories/; 13 duplicated methods removed; writes retry via retry_policy on both backends; update_evaluation allowlists columns. prompt_studio+Evaluations: 2019 passed, 5 failures pre-existing on HEAD (route mounting). DB_Management: 22 failures, all pre-existing (ChaCha/Media). Next: Stage 4 signatures, projects, prompts.

2026-09-23: Stage 4 done (4ec9f9c515 signatures, b50dbf10dc projects, this: prompts). Harness found & fixed on PG: uniqueness conflicts surfacing as DatabaseError (cursor wrapper matched redacted message; now typed -> ConflictError, repairs all PG writes); get_signature/get_prompt TypeError on missing rows; ensure_prompt_stub not advancing the id sequence (false ConflictError on next create). SQLite aligned: project name validation, InputError on missing-row updates, no swallowed DB errors in get_prompt*. Next: Stage 5 test cases.

2026-09-23: AC3 met. Stages 5-6 + helpers: 32c01e4105 test cases, dee169a794 optimizations, 689fb6d072 jobs (+ SQLite transaction() override that never issued BEGIN, found by the multiprocess acquisition test), 87e1cb21a2 sync-log/idempotency once. PromptStudioDatabase.py 7426 -> ~1700 lines; the two classes hold only connection/schema/execution/row-decoding infrastructure. ADR-051 Accepted. Open: Stage 7 typed facade (facade still forwards *args/**kwargs to repositories) - ergonomics, not correctness, since nothing is duplicated to drift.

2026-09-23: Stage 7 done (typed facade, signatures pinned to repositories). All stages complete.
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
