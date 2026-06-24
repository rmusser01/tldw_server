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
- tldw_Server_API/app/core/Embeddings/request_types.py
- tldw_Server_API/app/core/Embeddings/input_normalizer.py
- tldw_Server_API/app/core/Embeddings/provider_resolution.py
- tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
- tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_characterization.py
- tldw_Server_API/tests/Embeddings_isolated/test_request_types.py
- tldw_Server_API/tests/Embeddings_isolated/test_input_normalizer.py
- tldw_Server_API/tests/Embeddings_isolated/test_provider_resolution.py
updated_date: 2026-06-24 20:13
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
Task 2 request types completed and reviewed. Added dependency-light request_types.py with Embeddings domain errors, sanitized HTTP payloads, request/normalization/provider/policy/execution dataclasses, safe detail/tag scalar contracts, and runtime redaction for secret-bearing detail/tag values while preserving safe numeric token-count details for input_too_long. Added isolated tests for forbidden raw/secret context attributes, plan serialization, contract type hints, observability tag sanitization, domain error payload sanitization, detail redaction, post-construction detail mutation re-sanitization, token-count preservation, and mutable default isolation. Verification: initial red import failure observed; later red checks caught contract/safety regressions; final pytest passed with 9 passed, 30 warnings; compileall passed; Bandit passed. Spec compliance approved after fixes. Code-quality review approved after token-count preservation fix.
Task 3 input normalizer completed and reviewed. Added pure input_normalizer.py with normalize_embedding_input using injected token counters/decoders and no endpoint/FastAPI/settings dependencies. Covered string, list[str], token-array, batch token-array, list-size limits, blank string/list-specific blank messages, decode failures, domain-error propagation, decoded text validation, decoder output count validation, strict token_lengths validation, raw-length accounting, absent third-return fallback, and token-limit details. Verification included multiple red checks for missing module and later shape/quality regressions; final pytest passed with 24 passed, 60 warnings; compileall passed; Bandit passed. Spec compliance approved after decoder-shape and token-accounting fixes. Code-quality review approved after list-specific empty-string and domain-error propagation fixes.
Task 4 completed: extracted provider/model resolution into `app/core/Embeddings/provider_resolution.py`, added isolated coverage in `tests/Embeddings_isolated/test_provider_resolution.py`, and converted endpoint `_split_provider_model` / `_resolve_model_and_provider` into compatibility shims. Spec review approved with no required changes. Quality review approved; added suggested assertions for absent-model default provider/model resolution.

Task 4 verification:
- `python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_provider_resolution.py tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py::test_resolve_model_and_provider_strips_prefix tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py::test_resolve_model_and_provider_rejects_mismatch` -> 34 passed, 77 warnings.
- `python -m compileall -q tldw_Server_API/app/core/Embeddings/provider_resolution.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py tldw_Server_API/tests/Embeddings_isolated/test_provider_resolution.py` -> passed.
- `python -m bandit -r tldw_Server_API/app/core/Embeddings/provider_resolution.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py -f json -o /tmp/bandit_embeddings_provider_resolution_coord.json` -> 0 findings.
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
