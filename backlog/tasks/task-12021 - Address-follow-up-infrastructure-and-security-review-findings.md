---
id: TASK-12021
title: Address follow-up infrastructure and security review findings
status: Done
created_date: 2026-06-26 18:18
labels:
- backend
- security
- infrastructure
- review-fix
priority: high
references:
- tldw_Server_API/app/core/Infrastructure/distributed_lock.py
- tldw_Server_API/app/core/Security/egress.py
- tldw_Server_API/app/core/Infrastructure/redis_factory.py
- tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
modified_files:
- tldw_Server_API/app/core/Infrastructure/distributed_lock.py
- tldw_Server_API/app/core/Security/egress.py
- tldw_Server_API/app/core/Infrastructure/redis_factory.py
- tldw_Server_API/app/core/Infrastructure/README.md
- tldw_Server_API/app/api/v1/API_Deps/backpressure.py
- tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
- tldw_Server_API/tests/Infrastructure/test_distributed_lock.py
- tldw_Server_API/tests/Security/test_egress.py
- tldw_Server_API/tests/Infrastructure/test_redis_factory.py
- tldw_Server_API/tests/Infrastructure/test_redis_factory_metrics.py
- tldw_Server_API/tests/Embeddings/test_backpressure_and_quotas.py
updated_date: 2026-06-26 22:22
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix follow-up code review findings in infrastructure and security hardening: file lock ownership races, bounded DNS resolver behavior, explicit Redis fake fallback policy, and embeddings/ingest tenant quota fail-closed behavior. Verify the existing Redis migration-lock exception handling on the clean base.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FileLock release cannot delete a newer active lock and has focused regression coverage.
- [x] #2 FileLock residual files do not break native lock ownership and stale-file unlinking is not used for active locks.
- [x] #3 acquire_migration_lock preserves exceptions raised inside the protected body while still failing closed on Redis setup errors.
- [x] #4 Egress DNS timeout handling has bounded outstanding resolver work and focused regression coverage.
- [x] #5 Redis factory fake fallback is explicit for production-sensitive callers and missing-package behavior has focused regression coverage.
- [x] #6 Embeddings tenant RPS quota fails closed when Redis quota state is unavailable.
- [x] #7 Focused tests, diff check, py_compile, and Bandit touched-scope scan are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Port the reviewed fixes into a clean branch based on origin/dev. 2. Run focused regression tests and related suites in the clean worktree. 3. Run py_compile, diff hygiene, and Bandit on touched production files. 4. Mark the task Done with verification evidence before opening the PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Clean PR branch created from origin/dev at 6fe09bb972 in .worktrees/task-12021-followup-review-fixes.

Implemented fixes:
- FileLock keeps lock files in place on normal release so a prior owner cannot unlink a newer active lock.
- FileLock no longer attempts stale-file unlinking on failed acquisition; native platform locks determine ownership.
- FileLock acquisition no longer rewrites PID/UUID metadata or fsyncs the lock file because that metadata is not used for ownership after stale unlinking was removed.
- acquire_migration_lock already separated Redis setup errors from protected-body exceptions on the clean base; this task verified that existing behavior.
- Redis factory fake fallback is disabled by default for async and sync factories; callers must opt in explicitly.
- Egress DNS resolution bounds outstanding resolver work, defaults the cap to 64, allows WORKFLOWS_EGRESS_DNS_MAX_OUTSTANDING and WORKFLOWS_EGRESS_DNS_SLOT_WAIT_SECONDS overrides, waits briefly for a slot, and logs slot exhaustion, timeouts, worker-start failures, and resolver errors.
- Embeddings and ingest tenant RPS quota paths fail closed with HTTP 503 and Retry-After when Redis quota state is unavailable, with safe tenant/user/request context and exception details in warning logs.
- New/modified tests were adjusted to use approved unit markers and typed signatures where review comments flagged policy issues.

Verification in clean worktree:
- Baseline before patch: `python -m pytest tldw_Server_API/tests/Infrastructure/test_distributed_lock.py tldw_Server_API/tests/Security/test_egress.py tldw_Server_API/tests/Infrastructure/test_redis_factory.py tldw_Server_API/tests/Infrastructure/test_redis_factory_metrics.py tldw_Server_API/tests/test_minimal_deploy.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py -q` -> 49 passed.
- Initial clean-branch patch: focused suite -> 60 passed.
- Self-review follow-up: targeted red run for ingest quota outage and README contract -> 2 failed for expected reasons before fixes; green run -> 2 passed; full focused suite -> 62 passed.
- Latest PR comment pass targeted tests: `python -m pytest ...review-comment-focused test selection... -q` -> 8 passed.
- Latest PR comment pass full focused suite: `python -m pytest tldw_Server_API/tests/Infrastructure/test_distributed_lock.py tldw_Server_API/tests/Embeddings/test_backpressure_and_quotas.py tldw_Server_API/tests/Security/test_egress.py tldw_Server_API/tests/Infrastructure/test_redis_factory.py tldw_Server_API/tests/Infrastructure/test_redis_factory_metrics.py tldw_Server_API/tests/test_minimal_deploy.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py -q` -> 63 passed.
- `python -m py_compile tldw_Server_API/app/core/Infrastructure/distributed_lock.py tldw_Server_API/app/core/Security/egress.py tldw_Server_API/app/core/Infrastructure/redis_factory.py tldw_Server_API/app/api/v1/API_Deps/backpressure.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py` -> passed.
- `git diff --check -- ...touched files...` -> passed.
- `python -m bandit -r tldw_Server_API/app/core/Infrastructure/distributed_lock.py tldw_Server_API/app/core/Security/egress.py tldw_Server_API/app/core/Infrastructure/redis_factory.py tldw_Server_API/app/api/v1/API_Deps/backpressure.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py -f json -o /tmp/bandit_task_12021_pr_comments.json` -> 0 errors, 0 findings.

Known skips/blockers: None.
Reopened for latest PR issue pass: Qodo still flags ingest quota warning as insufficiently structured, and added a README contract-test brittleness recommendation. Also restore PR draft state after pushing because the human-authored change summary gate remains.
Latest PR issue pass addressed after user request to address all PR issues:
- Changed ingest tenant quota Redis-unavailable warning to use Loguru bound structured fields for tenant_id, exception_type, and event, while retaining exception traceback context via logger.opt(exception=exc).
- Mirrored the structured warning pattern on embeddings tenant quota Redis-unavailable logging for consistency.
- Made the Redis factory README contract test resilient to harmless formatting changes by using regex checks for documented explicit fallback examples rather than exact snippet strings.
- Extended the ingest quota fail-closed regression test to assert structured log context and exception capture.

Latest verification:
- `python -m pytest tldw_Server_API/tests/Embeddings/test_backpressure_and_quotas.py::test_ingest_tenant_quota_fails_closed_when_redis_unavailable tldw_Server_API/tests/Infrastructure/test_redis_factory.py::test_redis_factory_readme_documents_fail_closed_defaults -q` -> 2 passed.
- `python -m pytest tldw_Server_API/tests/Infrastructure/test_distributed_lock.py tldw_Server_API/tests/Embeddings/test_backpressure_and_quotas.py tldw_Server_API/tests/Security/test_egress.py tldw_Server_API/tests/Infrastructure/test_redis_factory.py tldw_Server_API/tests/Infrastructure/test_redis_factory_metrics.py tldw_Server_API/tests/test_minimal_deploy.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py -q` -> 63 passed.
- `python -m py_compile tldw_Server_API/app/api/v1/API_Deps/backpressure.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py tldw_Server_API/app/core/Infrastructure/redis_factory.py` -> passed.
- `git diff --check -- ...latest touched files...` -> passed.
- `python -m bandit -r tldw_Server_API/app/api/v1/API_Deps/backpressure.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py -f json -o /tmp/bandit_task_12021_all_pr_issues.json` -> 0 errors, 0 findings.

Remaining external state at start of this pass: GitHub full-suite checks had many pending shards; no local code change was made for pending-only checks. CodeRabbit docstring coverage warning appears global/non-actionable for this narrow PR. PR must remain draft until the human-authored change summary gate is satisfied.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created a clean PR branch from origin/dev, ported the TASK-12021 review fixes, addressed the latest PR review comments, and completed the follow-up all-issues pass. The branch keeps file locks from unlinking active successors, removes now-unused FileLock metadata writes, verifies the existing clean-base migration-lock body-exception behavior, defaults Redis helper fallbacks to fail-closed behavior unless explicitly enabled, makes egress DNS saturation observable/configurable, fails closed for embeddings and ingest tenant quota Redis outages with structured contextual warning logs, and updates Redis factory docs/tests to match the fallback contract without brittle exact-snippet assertions. Focused tests, compile, diff hygiene, and Bandit all passed in the clean worktree.
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
