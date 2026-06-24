---
id: TASK-12016
title: Implement Embeddings request orchestrator
status: In Progress
created_date: 2026-06-24 18:29
labels:
- embeddings
- implementation
- refactor
priority: High
modified_files:
- tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_characterization.py
- backlog/tasks/task-12016 - Implement-Embeddings-request-orchestrator.md
updated_date: 2026-06-24 18:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved Embeddings request orchestrator implementation plan using subagent-driven development. Scope includes characterization tests, pure core module extraction, compatibility endpoint shims, feature-flagged orchestrator path, endpoint parity tests, verification, and security scan. Base plan: Docs/superpowers/plans/2026-06-24-embeddings-request-orchestrator-implementation-plan.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Characterization tests capture current endpoint/batch cache, dimensions, RG, fallback, and error behavior before production extraction.
- [ ] #2 Core Embeddings modules are added for request types, input normalization, provider resolution, policy, and orchestrator prepare/execute phases.
- [ ] #3 Endpoint preserves legacy behavior by default and exposes the orchestrator path only behind EMBEDDINGS_ORCHESTRATOR_ENABLED.
- [ ] #4 Compatibility shims remain for existing endpoint helper symbols and tests while delegating to new owners where extracted.
- [ ] #5 Dual-path parity tests cover representative success, cache, dimensions, base64, fallback, and error cases.
- [ ] #6 Focused Embeddings tests, compile checks, git diff checks, and Bandit verification are recorded before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-06-24-embeddings-request-orchestrator-implementation-plan.md with subagent-driven development. Each task must use TDD: write tests, verify expected failure, implement minimal code, rerun tests, then review for spec compliance and code quality before moving to the next task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created after baseline validation in the implementation worktree. Baseline command passed: source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py::test_batch_length_mismatch_raises tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py::test_batch_rate_limit_maps_to_429 tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py::test_batch_generic_provider_error_is_sanitized tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py::test_resolve_model_and_provider_strips_prefix tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py::test_resolve_model_and_provider_rejects_mismatch. Result: 5 passed, 19 warnings in 0.73s.
Task 1 characterization tests completed and reviewed. Added test_embeddings_orchestrator_characterization.py covering full cache hit provider skip/order, partial cache hit miss-only execution/cache write, base64 response cache value, legacy dimension-adjustment cache write order, RG reserve/commit on full cache hit, and vector-count mismatch 502 behavior. Observed compatibility behavior: the legacy endpoint returns dimension-adjusted vectors but writes pre-adjustment provider vectors to cache. Verification: requested Task 1 pytest command passed with 11 passed, 174 warnings in 6.82s. Bandit on the touched test file passed. Spec-compliance review approved. Code-quality review initially requested fixture/app-state/assertion strengthening; worker fixed all issues; code-quality re-review approved.
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
