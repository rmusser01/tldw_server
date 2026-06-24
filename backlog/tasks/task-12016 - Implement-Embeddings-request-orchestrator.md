---
id: TASK-12016
title: Implement Embeddings request orchestrator
status: Done
created_date: 2026-06-24 18:29
labels:
- embeddings
- implementation
- refactor
priority: High
modified_files:
- Docs/superpowers/plans/2026-06-24-embeddings-request-orchestrator-implementation-plan.md
- tldw_Server_API/app/core/Embeddings/request_types.py
- tldw_Server_API/app/core/Embeddings/input_normalizer.py
- tldw_Server_API/app/core/Embeddings/provider_resolution.py
- tldw_Server_API/app/core/Embeddings/embedding_policy.py
- tldw_Server_API/app/core/Embeddings/orchestrator.py
- tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
- tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_characterization.py
- tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py
- tldw_Server_API/tests/Embeddings/test_embeddings_policy.py
- tldw_Server_API/tests/Embeddings/test_embeddings_unsupported_provider.py
- tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
- tldw_Server_API/tests/Embeddings_isolated/test_request_types.py
- tldw_Server_API/tests/Embeddings_isolated/test_input_normalizer.py
- tldw_Server_API/tests/Embeddings_isolated/test_provider_resolution.py
- tldw_Server_API/tests/Embeddings_isolated/test_embedding_policy.py
- tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py
updated_date: 2026-06-24 22:43
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved Embeddings request orchestrator implementation plan using subagent-driven development. Scope includes characterization tests, pure core module extraction, compatibility endpoint shims, feature-flagged orchestrator path, endpoint parity tests, verification, and security scan. Base plan: Docs/superpowers/plans/2026-06-24-embeddings-request-orchestrator-implementation-plan.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Characterization tests capture current endpoint/batch cache, dimensions, RG, fallback, and error behavior before production extraction.
- [x] #2 Core Embeddings modules are added for request types, input normalization, provider resolution, policy, and orchestrator prepare/execute phases.
- [x] #3 Endpoint preserves legacy behavior by default and exposes the orchestrator path only behind EMBEDDINGS_ORCHESTRATOR_ENABLED.
- [x] #4 Compatibility shims remain for existing endpoint helper symbols and tests while delegating to new owners where extracted.
- [x] #5 Dual-path parity tests cover representative success, cache, dimensions, base64, fallback, and error cases.
- [x] #6 Focused Embeddings tests, compile checks, git diff checks, and Bandit verification are recorded before completion.
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
Task 5 completed: extracted embedding policy decisions into `app/core/Embeddings/embedding_policy.py`; endpoint compatibility shims now delegate dimension policy, dimensions validation, L2 normalization, fallback chains/model mapping, allowlist checks, unsupported-provider handling, and fallback-chain decisions to the policy module. Added centralized `_enforce_embedding_policy_decision(...)` wrapper in the endpoint and endpoint regression tests for unknown/unsupported providers with invalid dimensions in both create and batch paths. Spec review approved after unknown-provider classification was added; quality review approved after endpoint pre-policy provider/dimension checks were removed.

Task 5 verification:
- Red check before final fix: `python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_unsupported_provider.py` failed with 3 failures exposing provider/dimensions ordering.
- Focused endpoint check after fix: `python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_unsupported_provider.py` -> 5 passed, 195 warnings.
- Policy suite: `python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_embedding_policy.py tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py tldw_Server_API/tests/Embeddings/test_embeddings_fallback_model_map.py tldw_Server_API/tests/Embeddings/test_l2_normalization_policy.py tldw_Server_API/tests/Embeddings/test_embeddings_batch_dimensions.py tldw_Server_API/tests/Embeddings/test_embeddings_policy.py tldw_Server_API/tests/Embeddings/test_embeddings_policy_toggle.py tldw_Server_API/tests/Embeddings/test_embeddings_policy_strict_mode.py tldw_Server_API/tests/Embeddings/test_embeddings_unsupported_provider.py` -> 45 passed, 754 warnings.
- Compile: touched policy, endpoint, and test files -> exit 0.
- Bandit: `python -m bandit -r tldw_Server_API/app/core/Embeddings/embedding_policy.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py -f json -o /tmp/bandit_embeddings_policy_coord.json` -> 0 findings, no errors.
Task 6 completed: added dependency-light `app/core/Embeddings/orchestrator.py` with EmbeddingCache/EmbeddingExecutor protocols, PreparedEmbeddingRequest, prepare() normalization/provider-resolution/policy/planning, and execute() cache read-through, miss-only executor calls, fallback model mapping, dimension postprocessing before cache writeback, vector-count validation, canonical float cache values, response headers, and redacted execution-plan boundaries. Added isolated orchestrator tests for prepare token accounting, full cache hit, partial cache hit, vector-count mismatch, redacted plan repr, fallback model mapping, and base64-independent cache values.

Task 6 red evidence: initial `python -m pytest -q --tb=short tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py` failed during collection with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Embeddings.orchestrator'`.

Task 6 verification:
- Direct import check: importing `tldw_Server_API.app.core.Embeddings.orchestrator` reported `fastapi` absent from `sys.modules`.
- Compile: `python -m compileall -q tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py` passed.
- Bandit: `python -m bandit -r tldw_Server_API/app/core/Embeddings/orchestrator.py -f json -o /tmp/bandit_embedding_orchestrator.json` passed with 0 results.
- Requested test command: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py` passed with 7 passed, 26 warnings in 8.74s.
Task 6 quality review fixes completed: fallback after partial primary cache hits now rebuilds a coherent full response under the fallback provider/model; fallback continues only for eligible retryable rate-limit/unavailable domain errors; non-retryable provider errors raise as-is; rate-limit exhaustion preserves the original retry_after; base64 requests with dimensions force effective dimension policy `reduce`; malformed non-list/tuple vector containers are rejected before cache writeback.

Task 6 post-review verification:
- `python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py` -> 12 passed, 36 warnings.
- `python -m compileall -q tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py` -> passed.
- `python -m bandit -r tldw_Server_API/app/core/Embeddings/orchestrator.py -f json -o /tmp/bandit_embedding_orchestrator_coord.json` -> 0 findings, no errors.
- Direct import check: importing `tldw_Server_API.app.core.Embeddings.orchestrator` reported `fastapi` absent from `sys.modules`.
Task 7 completed: wired `/api/v1/embeddings` through `EMBEDDINGS_ORCHESTRATOR_ENABLED` while preserving the legacy path as the default. Extracted the existing create-handler body to `_create_embedding_legacy`, added `_create_embedding_with_orchestrator` for endpoint-boundary responsibilities, added domain-error-to-HTTP mapping, applied orchestrator response headers, and added endpoint parity tests in `test_embeddings_orchestrator_endpoint_parity.py`.

Task 7 red evidence: initial focused flag-routing red run `python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py -k 'flag_'` failed with 2 failures because the public route still returned the legacy synthetic model (`text-embedding-3-small`) instead of calling the patched legacy/orchestrator seams. An earlier full red run also collected 7 tests and began failing, then was interrupted after a missing seam fell through to slow real provider setup.

Task 7 verification:
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py` -> 7 passed, 262 warnings.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m compileall -q tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py` -> passed.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py -f json -o /tmp/bandit_embeddings_orchestrator_endpoint_task7.json` -> 0 findings, no errors.
Task 7 quality review fixes completed. Preserved legacy BYOK OAuth 401 refresh/retry and final credential `touch_last_used` in `_EndpointEmbeddingExecutor`; preserved provider batching by `MAX_BATCH_SIZE` with combined vector-count validation; restored adapter-first execution when `LLM_EMBEDDINGS_ADAPTERS_ENABLED` is truthy; added `circuit_breaker_open` and `internal_execution_failure` to `EmbeddingErrorCode`; added a route-level real-orchestrator fallback-header parity test.

Task 7 review red evidence: after adding tests, `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py` failed with 3 expected failures: OpenAI OAuth 401 was mapped to `EmbeddingExecutionError` instead of forcing refresh/retry; provider batching sent all five texts in one provider call instead of 2/2/1 with `MAX_BATCH_SIZE=2`; adapter-enabled execution still called `create_embeddings_with_circuit_breaker`.

Task 7 review verification:
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py` -> 13 passed, 306 warnings.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m compileall -q tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py tldw_Server_API/app/core/Embeddings/request_types.py` -> passed.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py tldw_Server_API/app/core/Embeddings/request_types.py -f json -o /tmp/bandit_embeddings_orchestrator_endpoint_task7_review.json` -> 0 findings, no errors.
Task 7 endpoint wiring review loop completed.

Spec review: SPEC APPROVED after feature-flagged endpoint path, legacy handler extraction, ResourceGovernor boundary flow, domain error mapping, response headers, and endpoint parity tests were implemented.

Quality review findings addressed:
- Added provider preflight before primary and fallback cache reads so required provider credentials cannot be bypassed by full cache hits.
- Reused endpoint executor credential cache and touched resolved credentials after cached successes.
- Adapter-enabled orchestrator execution now skips provider-cache reads so adapter execution cannot be bypassed by stale provider cache, and adapter-produced vectors are not written to provider cache.
- Preserved adapter vector scale via EmbeddingExecutorOutput / embeddings_from_adapter propagation.
- Preserved non-429 provider HTTP 4xx behavior, including OpenAI OAuth second-401 propagation.
- Preserved legacy fallback behavior by skipping missing credentials for non-requested fallback providers so later fallback candidates can run.

Quality review: QUALITY APPROVED - prior credential/cache and adapter provenance findings resolved; no new Critical/Important Task 7 issues found.

Fresh verification:
- /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py::test_orchestrator_full_cache_hit_touches_resolved_provider_credentials -> 1 passed, 58 warnings
- /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py -> 31 passed, 470 warnings
- /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m compileall -q touched endpoint/core/test files -> exit 0
- /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r touched production files -f json -o /tmp/bandit_embeddings_task7_after_quality_fixes2.json -> 0 results, 0 errors
Task 8 completed: extended `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py` with a dual-path helper that runs each request once with `EMBEDDINGS_ORCHESTRATOR_ENABLED` unset and once with it enabled. The helper installs deterministic fakes for provider execution, cache get/set, BYOK credential resolution, metrics, token decoding/counting, fallback policy/model mapping, and ResourceGovernor reserve/commit for both runs. Added parity coverage for single string numeric embeddings, batch string index order, single token-array input, batch token-array base64 with dimensions, HuggingFace reduce/pad/ignore dimension policies, full cache hit provider skip, partial cache hit miss-only provider calls, OpenAI fallback to HuggingFace headers, explicit x-provider fallback suppression, and provider vector-count mismatch to 502. Parity assertions compare status, JSON, usage, provider/fallback/dimensions/rate-limit headers, and cache writes as float vectors.

Task 8 red evidence: after adding the first helper-using test before implementing the helper, `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py::test_task8_parity_helper_red_seed` failed with `NameError: name '_run_dual_path_embedding_request' is not defined`.

Validated production drifts fixed during Task 8:
- `app/core/Embeddings/orchestrator.py`: orchestrator response metadata now always includes `X-Embeddings-Provider`, matching the legacy endpoint for non-fallback provider execution.
- `app/api/v1/endpoints/embeddings_v5_production_enhanced.py`: endpoint executor provider vector-count mismatch detail now matches legacy batch-helper text (`... expected N for batch`) while preserving 502 mapping.

Task 8 verification:
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py` -> 29 passed, 851 warnings in 37.99s.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_characterization.py tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py tldw_Server_API/tests/Embeddings/test_embeddings_fallback_model_map.py` -> 49 passed, 1434 warnings in 62.02s.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m compileall -q tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py tldw_Server_API/app/core/Embeddings/orchestrator.py` -> passed with no output.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py tldw_Server_API/app/core/Embeddings/orchestrator.py -f json -o /tmp/bandit_embeddings_task8.json` -> exit 0; JSON results count 0 and errors `[]`. Stdout included Bandit comment-parser warnings for existing comment words (`non`, `cryptographic`, `retry`, `jitter`).
- `git diff --check` -> passed with no output.

Task 8 touched tracked files: `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py`, `tldw_Server_API/app/core/Embeddings/orchestrator.py`, `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`. The unrelated untracked watchlist template files were left untouched and unstaged.
Task 9 completed: added short endpoint compatibility migration comments for `_split_provider_model`, `_resolve_model_and_provider`, `_validate_dimensions_request`, `adjust_dimensions`, `decide_and_apply_l2`, `resolve_fallback_chain`, `map_model_for_provider`, and `create_embeddings_batch_async`. Comments name the new owner module and note removal after endpoint/legacy caller migration without user-facing deprecation messaging.

Final focused verification:
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_request_types.py tldw_Server_API/tests/Embeddings_isolated/test_input_normalizer.py tldw_Server_API/tests/Embeddings_isolated/test_provider_resolution.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_policy.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_characterization.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py tldw_Server_API/tests/Embeddings/test_embeddings_fallback_model_map.py tldw_Server_API/tests/Embeddings/test_batch_rate_headers.py` -> 168 passed, 1768 warnings.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m compileall -q tldw_Server_API/app/core/Embeddings/request_types.py tldw_Server_API/app/core/Embeddings/input_normalizer.py tldw_Server_API/app/core/Embeddings/provider_resolution.py tldw_Server_API/app/core/Embeddings/embedding_policy.py tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py` -> passed with no output.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Embeddings/request_types.py tldw_Server_API/app/core/Embeddings/input_normalizer.py tldw_Server_API/app/core/Embeddings/provider_resolution.py tldw_Server_API/app/core/Embeddings/embedding_policy.py tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py -f json -o /tmp/bandit_embeddings_orchestrator.json` -> 0 results, no errors. Stdout included Bandit comment-parser warnings for existing comment words (`non`, `cryptographic`, `retry`, `jitter`).
- `git diff --check` -> passed with no output.

Known skips/blockers: none for the focused Embeddings scope. Existing warnings remain in the selected test suite. Unrelated untracked watchlist template files remain untouched and unstaged.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Embeddings request orchestrator refactor behind `EMBEDDINGS_ORCHESTRATOR_ENABLED`. Added pure request types, input normalization, provider resolution, policy, and orchestrator modules; preserved endpoint compatibility shims and legacy default behavior; wired a feature-flagged orchestrator endpoint path; added characterization, isolated core, endpoint parity, policy, fallback, cache, BYOK, adapter, RG, and error-shape coverage; and completed focused verification plus Bandit with zero findings.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
