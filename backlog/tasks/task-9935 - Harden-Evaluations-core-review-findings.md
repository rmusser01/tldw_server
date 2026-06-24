---
id: TASK-9935
title: Harden Evaluations core review findings
status: Done
assignee: []
created_date: 2026-06-23 21:41
updated_date: 2026-06-24 03:54
labels:
- evaluations
- security
- hardening
dependencies: []
priority: high
modified_files:
- tldw_Server_API/app/core/Evaluations/eval_runner.py
- tldw_Server_API/app/core/Evaluations/recipe_runs_jobs_worker.py
- tldw_Server_API/app/core/Evaluations/recipe_runs_service.py
- tldw_Server_API/app/core/Evaluations/recipes/rag_answer_quality.py
- tldw_Server_API/app/core/Evaluations/synthetic_eval_repository.py
- tldw_Server_API/app/core/Evaluations/synthetic_eval_service.py
- tldw_Server_API/app/core/Evaluations/user_rate_limiter.py
- tldw_Server_API/app/core/Evaluations/webhook_manager.py
- tldw_Server_API/app/core/Evaluations/webhook_security.py
- tldw_Server_API/tests/Evaluations/test_evaluations_core_hardening.py
- tldw_Server_API/tests/Evaluations/test_recipe_runs_service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address validated current-code review findings in tldw_Server_API/app/core/Evaluations, including dataset ownership, webhook SSRF, persistence failure handling, recipe-run secret storage, synthetic eval queue ownership, cost limit atomicity, webhook domain matching, and stale adapter registration. This replaces the earlier collided TASK-2414 allocation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated findings have regression tests that fail before fixes and pass after fixes.
- [x] #2 Cross-user data access paths are owner-scoped for evaluations datasets and synthetic eval drafts.
- [x] #3 Webhook delivery paths use a single hardened validation/delivery flow or reject unsafe direct URLs.
- [x] #4 Sensitive per-run API keys are not persisted plaintext in recipe run metadata.
- [x] #5 Touched scope passes focused tests and Bandit scan.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Starting verified, test-first remediation for Evaluations core review findings. Earlier TASK-2414 output was ignored because that ID resolves to an unrelated Image Generation task in this worktree.

Verification complete. Focused tests: source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Evaluations/test_evaluations_core_hardening.py tldw_Server_API/tests/Evaluations/test_recipe_runs_service.py::test_recipe_service_redacts_sensitive_run_config_in_public_metadata tldw_Server_API/tests/Evaluations/unit/test_user_rate_limiter_minute_exact_and_reset.py -q -> 13 passed. Bandit: source .venv/bin/activate && python -m bandit -r <touched Evaluations core files> -f json -o /tmp/bandit_evaluations_9935.json -> 0 results. Documentation was not updated because the fixes are internal hardening behavior with regression tests.

Clean worktree PR verification on codex/evaluations-core-hardening-9935: focused pytest set passed with 13 passed; Bandit JSON report /tmp/bandit_evaluations_9935_clean_worktree.json reported 0 results; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated and fixed the Evaluations core review findings. Added regression coverage for cross-user dataset/synthetic draft ownership, webhook SSRF and DNS rebinding-safe delivery, persistence failure surfacing, recipe-run secret retention, domain boundary matching, stale adapter registration, and atomic daily cost reservation. Runtime changes scope dataset and synthetic sample lookups by owner, resolve webhook delivery to a validated IP target with Host preservation, avoid durable plaintext recipe-run secrets, raise on failed evaluation result persistence, stop registering the unimplemented PostgreSQL adapter, and make rate-limit cost reservation/write atomic. Verification: 13 focused tests passed and Bandit reported 0 findings on touched core files.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased PR #2460 branch on latest origin/dev and addressed validated Gemini/Qodo review comments: removed in-memory recipe run secret config storage in favor of BYOK-encrypted durable metadata, preserved candidate_api_keys through rag_answer_quality normalization for worker execution, moved webhook DNS resolution and rate-limit SQLite writes off the async event loop, preserved HTTPS webhook hostnames for TLS verification, fixed mixed IPv4/IPv6 private-network checks, normalized leading-dot domains, required synthetic workflow user scope for owner-filtered operations, and explicitly closed SQLite connections in the rate limiter. Verification: compileall on touched files passed; targeted review regressions passed (5 passed); expanded hardening/recipe/rate-limiter tests passed (32 passed); recipe service + worker suites passed (43 passed); Bandit on touched Evaluations core files reported 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
