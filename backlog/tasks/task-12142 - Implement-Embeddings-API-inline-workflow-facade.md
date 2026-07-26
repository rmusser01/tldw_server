---
id: TASK-12142
title: Implement Embeddings API inline workflow facade
status: Done
assignee: []
created_date: 2026-07-04 01:19
updated_date: 2026-07-18 18:26
labels:
- embeddings
- implementation
- workflow
dependencies: []
references:
- Docs/superpowers/specs/2026-07-03-embeddings-workflow-architecture-design.md
- Docs/superpowers/plans/2026-07-03-embeddings-api-inline-workflow-facade-implementation-plan.md
- https://github.com/rmusser01/tldw_server/pull/2733
priority: high
modified_files:
- tldw_Server_API/app/core/Embeddings/workflow_types.py
- tldw_Server_API/app/core/Embeddings/workflow_runner.py
- tldw_Server_API/app/core/exceptions.py
- tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
- tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py
- tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py
- tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
- tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py
- tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_postgres.py
- tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v52_integration.py
- Docs/superpowers/plans/2026-07-03-embeddings-api-inline-workflow-facade-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the Stage 1 implementation plan for the canonical Embeddings workflow architecture. Scope: add workflow type contracts, no-op/in-memory trace collectors, inline workflow runner, endpoint pre-execute RG boundary hook, feature-flagged endpoint integration, tests, verification, and Bandit. No durable Jobs runner, schema changes, media/vector-store migration, or public API trace exposure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workflow type contracts and safe bounded trace collectors are implemented with isolated tests.
- [x] #2 Inline workflow runner wraps the existing EmbeddingRequestOrchestrator and preserves pre-execute RG reservation ordering.
- [x] #3 Feature-flagged endpoint path uses the inline runner without changing public response behavior, headers, metrics, logs, schemas, or legacy shims.
- [x] #4 Focused workflow, orchestrator, endpoint parity, compile, Bandit, and diff checks pass.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Stage 1 API inline workflow facade.

Verification completed:
- Focused Stage 1 suite: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py` -> 74 passed.
- Broader focused Embeddings regression suite: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_request_types.py tldw_Server_API/tests/Embeddings_isolated/test_input_normalizer.py tldw_Server_API/tests/Embeddings_isolated/test_provider_resolution.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_policy.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_characterization.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py tldw_Server_API/tests/Embeddings/test_embeddings_fallback_model_map.py tldw_Server_API/tests/Embeddings/test_batch_rate_headers.py` -> 200 passed.
- Compile: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m compileall -q tldw_Server_API/app/core/Embeddings/workflow_types.py tldw_Server_API/app/core/Embeddings/workflow_runner.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py` -> passed.
- Bandit: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Embeddings/workflow_types.py tldw_Server_API/app/core/Embeddings/workflow_runner.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py -f json -o /tmp/bandit_embeddings_inline_workflow_final.json` -> errors [], results [].
- `git diff --check` -> passed.

Review note: Task 1 and Task 2 had subagent spec/quality review loops. The final Task 2 quality re-review and Task 3 review were performed locally because subagent usage limits were reached; validated findings were fixed before final verification.
PR-readiness review on current origin/dev identified four validated follow-up areas to address before push: trace string value safety, immutable event metadata, explicit planning phase transitions, and ResourceGovernor commit/failure parity coverage. Applying fixes with test-first verification.
PR-readiness review remediation completed. Trace contracts now use immutable snapshots, explicit planning-phase emission, field-specific metadata allowlists, fixed enum strings, aggregate-only execution data, generated and constructor-validated workflow ids, and no caller-controlled provider/model/cache/fallback/request/user/error/class/header identifiers. ResourceGovernor parity coverage now includes reserve/commit success, execute failure, and reservation denial. Independent final review found no remaining Critical or Important findings. Focused workflow tests: 38 passed; endpoint parity tests: 37 passed; targeted mypy and compile checks passed.
Final rebased verification on origin/dev f05fe296db68ffccc411cc1c97bcdb5a123a9b03: 210 in-scope Embeddings tests passed; targeted workflow suite 38 passed; mypy reported no issues; compileall passed; Bandit errors/results were empty; git diff --check passed; branch is 0 behind and 14 commits ahead. The broader 218-test run produced 214 passed and 4 token-array failures. A detached untouched origin/dev worktree reproduced the same four failures exactly (400 Invalid token array input during tokenizer decoding before workflow execution), so they are recorded as an upstream baseline issue outside this PR. Draft PR: https://github.com/rmusser01/tldw_server/pull/2733
PR #2733 follow-up review pass started. Branch confirmed current with origin/dev f05fe296db. Reviewing four Gemini inline comments: validate numeric-cast suggestions against typed domain contracts; simplify duplicate forbidden-field checks where behavior remains unchanged; reply to each thread with technical disposition; rerun focused tests, mypy, compile, Bandit, and PR checks.
PR #2733 review follow-up completed. Validated all seven inline comments. Implemented five valid items: consolidated the duplicate forbidden-field sets and removed the redundant exact-match branch; documented all new workflow symbols; typed the ResourceGovernor pre-execute hook; centralized EmbeddingWorkflowTraceError in app/core/exceptions.py with a compatibility re-export and regression test. Rejected two numeric-cast suggestions because PreparedEmbeddingRequest and EmbeddingExecutionResult expose strict int counters; coercing floats or None would silently truncate or fabricate accounting and weaken the trace contract's fail-closed validation. Fresh verification: 211 in-scope Embeddings tests passed, focused workflow suite 39 passed, workflow Ruff and mypy checks passed, compileall passed, Bandit reported zero errors/findings, and git diff --check passed. Branch remains 0 commits behind origin/dev f05fe296db.
PR #2733 CI follow-up reopened on 2026-07-18. User requested rebase onto latest dev and remediation of all failed checks from run 29392257054. Initial triage: four Ubuntu full-suite shards failed at approximately 1 hour (Python 3.12/3.13 db-privileges and chacha-content-persona); gathering job-step and log evidence before changes.
CI follow-up root cause and remediation (2026-07-18): rebased cleanly onto origin/dev 668b0fce5707134768f880b5d064ccc5b0cc4691. Run 29392257054 had four cancelled Ubuntu jobs (Python 3.12/3.13 db-privileges and chacha-content-persona), all ending at the one-hour job limit without an assertion failure. Local reproduction against isolated PostgreSQL identified two deterministic test self-locks: test_workspace_sub_resources_postgres issued DROP TABLE from a second connection while CharactersRAGDB retained a read transaction; test_chacha_postgres_migration_v52_integration reran schema/FTS DDL while its verification read transaction remained open. Corrected the tests to release the CharactersRAGDB connection before cross-connection DDL, grouped downgrade DDL/seed writes in one backend transaction, and added 30-second per-test timeout guards. Updated the migration test to assert CharactersRAGDB._CURRENT_SCHEMA_VERSION after dev advanced to schema v54. Also made the adjacent SQLite review-transition test deterministic after validating that millisecond timestamps are not guaranteed unique. Verification: both live PostgreSQL tests passed together (2 passed); exact predecessor/file-boundary sequence passed (60 passed); focused workflow/endpoint suite passed (76 passed); broad Embeddings run produced 215 passed plus the same four token-array baseline failures previously reproduced on untouched dev before workflow execution; compileall passed; changed/new-file Ruff passed; targeted mypy with legacy imports skipped passed; Bandit reported no findings; git diff --check passed. Broad legacy endpoint Ruff/mypy findings remain baseline debt and are not introduced by this PR.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and reviewed Stage 1 of the Embeddings workflow architecture. Added immutable typed workflow contracts, bounded no-op/in-memory collectors, aggregate-only allowlisted trace metadata, generated and validated workflow IDs, an inline prepare/execute runner with failure preservation, explicit planning transitions, and feature-flagged endpoint integration that keeps ResourceGovernor reservation/commit ordering intact. Added endpoint parity coverage for success, execute failure, and reservation denial. Rebased onto current dev and opened draft PR #2733. No durable runner, schema change, public trace exposure, or flag promotion is included.
PR #2733 review remediation preserves strict integer accounting while resolving all validated maintainability and architecture findings. All seven inline comments have a documented technical disposition, and fresh workflow, endpoint, lint, type, compile, and security checks pass.
CI follow-up rebased PR #2733 onto latest dev and corrected both PostgreSQL integration-test self-locks that caused four Ubuntu shards to reach the one-hour limit. The migration test now follows the current schema version, both PostgreSQL tests release conflicting read connections before DDL and have bounded timeouts, and the adjacent millisecond-sensitive SQLite assertion uses a controlled clock. The exact failed file transitions pass against isolated PostgreSQL.
<!-- SECTION:FINAL_SUMMARY:END -->
