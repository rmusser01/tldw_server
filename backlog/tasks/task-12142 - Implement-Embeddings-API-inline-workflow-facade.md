---
id: TASK-12142
title: Implement Embeddings API inline workflow facade
status: Done
assignee: []
created_date: 2026-07-04 01:19
updated_date: 2026-07-15 00:35
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
- tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
- tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py
- tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py
- tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and reviewed Stage 1 of the Embeddings workflow architecture. Added immutable typed workflow contracts, bounded no-op/in-memory collectors, aggregate-only allowlisted trace metadata, generated and validated workflow IDs, an inline prepare/execute runner with failure preservation, explicit planning transitions, and feature-flagged endpoint integration that keeps ResourceGovernor reservation/commit ordering intact. Added endpoint parity coverage for success, execute failure, and reservation denial. Rebased onto current dev and opened draft PR #2733. No durable runner, schema change, public trace exposure, or flag promotion is included.
<!-- SECTION:FINAL_SUMMARY:END -->
